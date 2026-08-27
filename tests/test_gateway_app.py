from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion

from branchpilot.gateway.app import _MAX_BODY_BYTES, _BoundaryMiddleware, _run_guarded, create_app
from branchpilot.gateway.config import GatewayConfig, InboundAPIKeys, ModelRoute, UpstreamConfig
from branchpilot.gateway.upstream import GatewayError, UpstreamChoice, UpstreamSample, Usage
from branchpilot.schema import Sample
from branchpilot.strategies import FixedStrategy


class FakeUpstream:
    def __init__(self, scripted: list[UpstreamSample | BaseException]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0
        self.live = 0
        self.max_live = 0

    async def sample(self, body: dict[str, Any], **kwargs: Any) -> UpstreamSample:
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        try:
            self.calls.append({"body": body, **kwargs})
            value = self.scripted.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        finally:
            self.live -= 1

    async def close(self) -> None:
        self.close_count += 1


def _sample(
    text: str,
    answer: str | None,
    *,
    prompt: int = 10,
    completion: int = 2,
    logprob: float | None = None,
    finish: str = "stop",
    prompt_details: dict[str, int] | None = None,
    completion_details: dict[str, int] | None = None,
) -> UpstreamSample:
    parse_status = "parsed" if answer is not None else "unparsed"
    if finish == "length":
        parse_status = "truncated"
        answer = None
    return UpstreamSample(
        sample=Sample(
            text=text,
            answer=answer,
            token_count=completion,
            mean_logprob=logprob,
            finish_reason=finish,
            parse_status=parse_status,
        ),
        choice=UpstreamChoice(
            content=text,
            finish_reason=finish,
            refusal=None,
            logprobs=None,
        ),
        usage=Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
        ),
        upstream_request_id="must-not-leak",
    )


def _config(
    strategy: Any,
    *,
    route_max: int | None = None,
    allow_overrides: bool = False,
    allowed_strategies: frozenset[str] = frozenset(),
    request_timeout: float = 1.0,
    sessions: int = 2,
    queue_timeout: float = 0.05,
    extractor: str = "numeric-strict",
) -> GatewayConfig:
    deployment = SimpleNamespace(
        strategy=strategy,
        cost=0.0,
        spec={
            "type": "fixed",
            "samples": strategy.max_samples,
            "max_samples": strategy.max_samples,
        },
    )
    plan = SimpleNamespace(policy=f"fixed-{strategy.max_samples}", family="fixed")
    upstream = UpstreamConfig(
        name="local", base_url="http://private.internal/v1", api_key="provider-key"
    )
    route = ModelRoute(
        alias="public-math",
        upstream="local",
        upstream_model="private/model",
        deployment=deployment,
        plan=plan,
        extractor=extractor,
        max_samples=route_max or strategy.max_samples,
        max_completion_tokens=64,
        options={"temperature": 0.8},
        allow_client_overrides=allow_overrides,
        allowed_strategies=allowed_strategies,
    )
    return GatewayConfig(
        inbound_api_keys=("inbound-key",),
        request_timeout_s=request_timeout,
        queue_timeout_s=queue_timeout,
        max_concurrent_sessions=sessions,
        upstreams={"local": upstream},
        models={"public-math": route},
    )


def _body(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": "public-math",
        "messages": [{"role": "user", "content": "question"}],
    }
    value.update(changes)
    return value


def _headers(token: str = "inbound-key") -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_chunked_body_limit_counts_all_chunks_before_endpoint_completion() -> None:
    async def scenario() -> None:
        messages = iter(
            [
                {
                    "type": "http.request",
                    "body": b"x" * 700_000,
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": b"x" * (_MAX_BODY_BYTES - 699_999),
                    "more_body": False,
                },
            ]
        )
        sent: list[dict[str, Any]] = []
        completed = 0

        async def receive() -> dict[str, Any]:
            return next(messages)

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def endpoint(scope: object, bounded_receive: Any, endpoint_send: Any) -> None:
            nonlocal completed
            del scope
            while True:
                message = await bounded_receive()
                if not message.get("more_body", False):
                    break
            completed += 1
            await endpoint_send({"type": "http.response.start", "status": 204, "headers": []})

        middleware = _BoundaryMiddleware(endpoint, InboundAPIKeys(("secret",)), 1.0)
        await middleware(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "headers": [(b"authorization", b"Bearer secret")],
            },
            receive,
            send,
        )

        assert sent[0]["status"] == 413
        assert completed == 0

    asyncio.run(scenario())


def test_body_read_deadline_fails_closed_before_endpoint_completion() -> None:
    async def scenario() -> None:
        sent: list[dict[str, Any]] = []
        completed = 0

        async def receive() -> dict[str, Any]:
            await asyncio.sleep(0.05)
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def endpoint(scope: object, bounded_receive: Any, endpoint_send: Any) -> None:
            nonlocal completed
            del scope
            await bounded_receive()
            completed += 1
            await endpoint_send({"type": "http.response.start", "status": 204, "headers": []})

        middleware = _BoundaryMiddleware(endpoint, InboundAPIKeys(("secret",)), 0.001)
        await middleware(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "headers": [(b"authorization", b"Bearer secret")],
            },
            receive,
            send,
        )

        assert sent[0]["status"] == 408
        assert completed == 0

    asyncio.run(scenario())


def test_unauthenticated_request_reads_zero_body_messages_and_redacts_middleware() -> None:
    async def scenario() -> None:
        sent: list[dict[str, Any]] = []
        body_reads = 0
        endpoint_calls = 0

        async def receive() -> dict[str, Any]:
            nonlocal body_reads
            body_reads += 1
            return {"type": "http.request", "body": b"secret body", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def endpoint(scope: object, endpoint_receive: Any, send: Any) -> None:
            nonlocal endpoint_calls
            del scope, endpoint_receive, send
            endpoint_calls += 1

        middleware = _BoundaryMiddleware(endpoint, InboundAPIKeys(("inbound-secret",)), 1.0)
        await middleware(
            {"type": "http", "path": "/v1/chat/completions", "headers": []},
            receive,
            send,
        )

        assert sent[0]["status"] == 401
        assert body_reads == 0
        assert endpoint_calls == 0
        assert "inbound-secret" not in repr(middleware)

    asyncio.run(scenario())


@pytest.mark.parametrize("content_length", [b"-1", b"invalid", b"1, 2"])
def test_malformed_content_length_fails_closed_without_body_read(
    content_length: bytes,
) -> None:
    async def scenario() -> None:
        sent: list[dict[str, Any]] = []
        body_reads = 0

        async def receive() -> dict[str, Any]:
            nonlocal body_reads
            body_reads += 1
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def endpoint(scope: object, endpoint_receive: Any, endpoint_send: Any) -> None:
            raise AssertionError("endpoint must not run")

        middleware = _BoundaryMiddleware(endpoint, InboundAPIKeys(("secret",)), 1.0)
        await middleware(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "headers": [
                    (b"authorization", b"Bearer secret"),
                    (b"content-length", content_length),
                ],
            },
            receive,
            send,
        )

        assert sent[0]["status"] == 400
        assert body_reads == 0

    asyncio.run(scenario())


def test_authentication_precedes_body_parsing_and_separates_credentials() -> None:
    upstream = FakeUpstream([_sample("answer", "42")])
    config = _config(FixedStrategy(1, 1))
    app = create_app(config, upstreams={"local": upstream})
    middleware_repr = repr(app.user_middleware)
    assert "inbound-key" not in middleware_repr
    assert "provider-key" not in middleware_repr
    assert "inbound-key" not in repr(config.inbound_api_keys)
    with TestClient(app) as client:
        for headers in ({}, {"authorization": "Basic x"}, _headers("wrong")):
            response = client.post("/v1/chat/completions", content="{not-json", headers=headers)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_api_key"
            assert response.headers["www-authenticate"] == "Bearer"
        assert upstream.calls == []
        malformed = client.post("/v1/chat/completions", content="{not-json", headers=_headers())
        assert malformed.status_code == 400
        assert upstream.calls == []
        assert client.get("/healthz").json() == {"status": "ok"}


def test_routes_alias_forces_structure_and_returns_sdk_compatible_response() -> None:
    upstream = FakeUpstream([_sample("work\n\\boxed{42}", "42", prompt=7, completion=3)])
    app = create_app(_config(FixedStrategy(1, 1)), upstreams={"local": upstream})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={**_headers(), "x-request-id": "client-controlled", "cookie": "secret=1"},
            json=_body(temperature=0.2, n=1, stream=False),
        )
    assert response.status_code == 200
    parsed = ChatCompletion.model_validate(response.json())
    assert parsed.model == "public-math"
    assert parsed.choices[0].message.content == "work\n\\boxed{42}"
    assert parsed.usage is not None and parsed.usage.total_tokens == 10
    call = upstream.calls[0]
    assert call["body"]["upstream_model"] == "private/model"
    assert call["body"]["temperature"] == 0.8
    assert call["body"]["n"] == 1
    assert call["body"]["stream"] is False
    assert call["body"]["max_completion_tokens"] == 64
    assert call["public_request_id"] != "client-controlled"
    rendered = response.text
    assert "private/model" not in rendered
    assert "must-not-leak" not in rendered
    assert "private.internal" not in rendered


def test_exact_sequential_early_stop_and_aggregate_repeated_usage() -> None:
    upstream = FakeUpstream(
        [
            _sample("first 42", "42", prompt=10, completion=2, prompt_details={"cached_tokens": 1}),
            _sample(
                "different 7", "7", prompt=11, completion=3, prompt_details={"cached_tokens": 2}
            ),
            _sample(
                "representative 42",
                "42",
                prompt=12,
                completion=4,
                prompt_details={"cached_tokens": 3},
            ),
        ]
    )
    app = create_app(_config(FixedStrategy(3, 4), route_max=4), upstreams={"local": upstream})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.status_code == 200
    assert len(upstream.calls) == 3
    assert [call["sample_index"] for call in upstream.calls] == [1, 2, 3]
    assert upstream.max_live == 1
    assert response.json()["choices"][0]["message"]["content"] == "first 42"
    assert response.json()["usage"] == {
        "prompt_tokens": 33,
        "completion_tokens": 9,
        "total_tokens": 42,
        "prompt_tokens_details": {"cached_tokens": 6},
    }
    assert response.headers["x-branchpilot-samples"] == "3"
    assert response.headers["x-branchpilot-selection"] == "majority"


def test_policy_tie_representative_and_unparsed_fallback() -> None:
    tie = FakeUpstream(
        [
            _sample("low confidence", "42", logprob=-5.0),
            _sample("high confidence", "7", logprob=-1.0),
        ]
    )
    app = create_app(_config(FixedStrategy(2, 2)), upstreams={"local": tie})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.json()["choices"][0]["message"]["content"] == "high confidence"

    unparsed = FakeUpstream([_sample("first", None), _sample("second", None)])
    app = create_app(_config(FixedStrategy(2, 2)), upstreams={"local": unparsed})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.json()["choices"][0]["message"]["content"] == "first"
    assert response.headers["x-branchpilot-selection"] == "unparsed-fallback"


def test_parsed_sentinel_like_answer_selects_its_true_majority() -> None:
    parsed = "<unparsed:0>"
    upstream = FakeUpstream(
        [
            _sample("minority", "minority"),
            _sample(parsed, parsed),
            _sample(parsed, parsed),
        ]
    )
    app = create_app(
        _config(FixedStrategy(3, 3), extractor="exact-content"),
        upstreams={"local": upstream},
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == parsed
    assert response.headers["x-branchpilot-selection"] == "majority"


def test_horizon_override_is_bounded_and_cost_is_never_client_controlled() -> None:
    upstream = FakeUpstream([_sample("one", "1"), _sample("two", "2")])
    config = _config(FixedStrategy(3, 3), allow_overrides=True)
    app = create_app(config, upstreams={"local": upstream})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=_body(branchpilot={"max_samples": 2}),
        )
        assert response.status_code == 200
        assert len(upstream.calls) == 2

    untouched = FakeUpstream([_sample("must not run", "1")])
    app = create_app(config, upstreams={"local": untouched})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=_body(branchpilot={"cost": 0.1}),
        )
    assert response.status_code == 400
    assert untouched.calls == []


def test_route_completion_cap_is_sent_when_omitted_and_cap_plus_one_is_rejected() -> None:
    allowed = FakeUpstream([_sample("answer", "42")])
    config = _config(FixedStrategy(1, 1))
    app = create_app(config, upstreams={"local": allowed})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.status_code == 200
    assert allowed.calls[0]["body"]["max_completion_tokens"] == 64

    smaller = FakeUpstream([_sample("answer", "42")])
    app = create_app(config, upstreams={"local": smaller})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=_body(max_tokens=32),
        )
    assert response.status_code == 200
    assert smaller.calls[0]["body"]["max_tokens"] == 32
    assert "max_completion_tokens" not in smaller.calls[0]["body"]

    rejected = FakeUpstream([_sample("must not run", "1")])
    app = create_app(config, upstreams={"local": rejected})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=_body(max_completion_tokens=65),
        )
    assert response.status_code == 400
    assert rejected.calls == []


def test_unknown_model_schema_and_unsupported_fields_make_zero_calls() -> None:
    upstream = FakeUpstream([_sample("unused", "1")])
    app = create_app(_config(FixedStrategy(1, 1)), upstreams={"local": upstream})
    with TestClient(app) as client:
        unknown = client.post(
            "/v1/chat/completions", headers=_headers(), json=_body(model="unknown")
        )
        tools = client.post("/v1/chat/completions", headers=_headers(), json=_body(tools=[]))
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert tools.status_code == 400
    assert upstream.calls == []


def test_later_upstream_failure_fails_closed_and_is_sanitized() -> None:
    error = GatewayError(502, "api_error", "upstream_error", "The upstream is unavailable.")
    upstream = FakeUpstream([_sample("paid first sample", "1"), error])
    app = create_app(_config(FixedStrategy(2, 2)), upstreams={"local": upstream})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "The upstream is unavailable.",
            "type": "api_error",
            "param": None,
            "code": "upstream_error",
        }
    }
    assert "paid first sample" not in response.text
    assert len(upstream.calls) == 2


def test_total_timeout_cancels_active_generation_and_lifespan_closes_once() -> None:
    class SlowUpstream(FakeUpstream):
        def __init__(self) -> None:
            super().__init__([])
            self.cancelled = False

        async def sample(self, body: dict[str, Any], **kwargs: Any) -> UpstreamSample:
            self.calls.append({"body": body, **kwargs})
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            raise AssertionError("unreachable")

    upstream = SlowUpstream()
    app = create_app(
        _config(FixedStrategy(1, 1), request_timeout=0.02), upstreams={"local": upstream}
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "upstream_timeout"
    assert upstream.cancelled
    assert upstream.close_count == 1


def test_disconnect_watcher_cancels_active_operation_without_background_work() -> None:
    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def scenario() -> None:
        cancelled = False

        async def operation() -> None:
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled = True
                raise

        with pytest.raises(asyncio.CancelledError):
            await _run_guarded(DisconnectedRequest(), operation(), 1.0)  # type: ignore[arg-type]
        assert cancelled

    asyncio.run(scenario())


def test_lifespan_reuses_one_upstream_across_requests_and_closes_it() -> None:
    upstream = FakeUpstream([_sample("one", "1"), _sample("two", "2")])
    app = create_app(_config(FixedStrategy(1, 1)), upstreams={"local": upstream})
    with TestClient(app) as client:
        assert (
            client.post("/v1/chat/completions", headers=_headers(), json=_body()).status_code == 200
        )
        assert (
            client.post("/v1/chat/completions", headers=_headers(), json=_body()).status_code == 200
        )
        assert upstream.close_count == 0
    assert len(upstream.calls) == 2
    assert upstream.close_count == 1


@pytest.mark.parametrize(
    "error,status,code",
    [
        (
            GatewayError(400, "invalid_request_error", "upstream_rejected_request", "Rejected."),
            400,
            "upstream_rejected_request",
        ),
        (
            GatewayError(429, "rate_limit_error", "gateway_overloaded", "Overloaded."),
            429,
            "gateway_overloaded",
        ),
        (GatewayError(504, "api_error", "upstream_timeout", "Timed out."), 504, "upstream_timeout"),
    ],
)
def test_error_status_matrix_uses_openai_envelopes(
    error: GatewayError, status: int, code: str
) -> None:
    upstream = FakeUpstream([error])
    app = create_app(_config(FixedStrategy(1, 1)), upstreams={"local": upstream})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=_body())
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == code
    assert response.headers["x-request-id"]
