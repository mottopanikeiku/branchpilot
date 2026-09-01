"""Sentinel-injection leakage suite: credentials and user content never escape.

Every secret and every content position in the stack is filled with an unmistakable
sentinel string. The stack is then exercised end to end -- one success path and every
error path -- and every artifact a client, an operator log, or a Python ``repr`` can
observe is scanned for those sentinels.

The generalized failure this suite closes: a provider adapter refusal of the form

    provider 'anthropic' does not support request field(s) seed;
    fix: remove seed from ... models.<alias>.options ...

used to be returned verbatim in the client's HTTP error body, handing an unauthenticated
caller the provider identity and the operator's config paths. It is now sanitized in
``branchpilot.gateway.upstream.OpenAIUpstream._bounded_response`` (constant client
message, actionable detail logged server-side only). One fixed call site is not a
guarantee, so this suite asserts the *class* of leak is impossible: no surface may carry
a credential, and no client-visible surface may carry operator infrastructure.

Two directions are enforced, because a sanitized client body is worthless if the operator
loses the diagnostic:

* negative -- the client sees constant text and nothing else;
* positive -- the server-side log still names the provider and the ``fix:`` clause.

No test in this module performs a network call: every upstream is an
``httpx.MockTransport`` or an in-process double, and the ASGI app is driven by
``TestClient``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from branchpilot.gateway.app import _MAX_BODY_BYTES, _BoundaryMiddleware, create_app
from branchpilot.gateway.config import (
    ConfigError,
    GatewayConfig,
    InboundAPIKeys,
    ModelRoute,
    UpstreamConfig,
    load_gateway_config,
)
from branchpilot.gateway.upstream import GatewayError, OpenAIUpstream
from branchpilot.ingest import read_requests
from branchpilot.ingest.formats import MalformedRecordError
from branchpilot.strategies import FixedStrategy

# --------------------------------------------------------------------------------------
# surface registry
# --------------------------------------------------------------------------------------

#: Every artifact this module injects sentinels into and scans. A new surface belongs
#: here *and* in a test; a surface named here without a scanner is a silent hole.
SCANNED_SURFACES = (
    "http.response.body.success",
    "http.response.body.error",
    "http.response.headers.success",
    "http.response.headers.error",
    "http.response.body.healthz",
    "logging.record.message",
    "logging.record.__dict__",
    "repr.GatewayConfig",
    "repr.UpstreamConfig",
    "repr.ModelRoute",
    "repr.InboundAPIKeys",
    "repr._BoundaryMiddleware",
    "repr.app.user_middleware",
    "str/repr.GatewayError",
    "str/repr.ConfigError",
    "repr.RequestRecord",
    "repr.IngestReport",
    "str/repr.ingest-exception-mid-iteration",
)

#: Surfaces that do not exist in the tree yet. When a card lands one, extend this module
#: rather than rediscovering the sentinel harness:
#:
#: * ``metrics.prometheus`` -- no ``/metrics`` endpoint and no collector exists today.
#:   ``test_metrics_surface_is_still_absent`` is the tripwire that fails when one lands,
#:   naming this list. Scan the exposition text: label *values* are the leak (a model
#:   alias is fine, an upstream base URL or an API key is not).
#: * ``ledger.record`` -- the audit ledger rows (savings/opportunity persistence) are not
#:   written by any module yet. Scan the serialized row and its ``repr``.
#: * ``cockpit.api.response`` -- no cockpit HTTP surface exists yet. Scan every JSON
#:   response body and header, exactly like the gateway cases below.
FUTURE_SURFACES = (
    "metrics.prometheus",
    "ledger.record",
    "cockpit.api.response",
)

# --------------------------------------------------------------------------------------
# sentinels
# --------------------------------------------------------------------------------------

INBOUND_KEY = "SENTINEL-INBOUND-KEY-7f3a"
UPSTREAM_KEY = "SENTINEL-UPSTREAM-KEY-b21c"
UPSTREAM_HOST = "SENTINEL-UPSTREAM-HOST-9de4"
UPSTREAM_MODEL = "SENTINEL-UPSTREAM-MODEL-4c17"
UPSTREAM_NAME = "SENTINEL-UPSTREAM-NAME-05ca"
PROMPT = "SENTINEL-PROMPT-a83b"
SYSTEM_PROMPT = "SENTINEL-SYSTEM-c74d"
COMPLETION = "SENTINEL-COMPLETION-e59f"
CLIENT_MODEL = "SENTINEL-CLIENT-MODEL-1d05"

#: Secrets. Forbidden on every surface, without exception, forever.
CREDENTIALS = (INBOUND_KEY, UPSTREAM_KEY)

#: Operator infrastructure. Legitimate in a server-side log, never client-visible.
INFRASTRUCTURE = (UPSTREAM_HOST, UPSTREAM_MODEL, UPSTREAM_NAME)

#: End-user text. Legitimate in the client's own response, never in a log or a repr.
CONTENT = (PROMPT, SYSTEM_PROMPT, COMPLETION)

#: Nothing a client receives may carry a credential or operator infrastructure.
CLIENT_FORBIDDEN = CREDENTIALS + INFRASTRUCTURE

#: Nothing a log record carries may carry a credential or end-user text.
LOG_FORBIDDEN = CREDENTIALS + CONTENT

#: The Batch 1 refusal leak, verbatim: provider identity plus operator config paths.
REFUSAL_TELLS = (
    "anthropic",
    "fix:",
    "seed",
    "models.<alias>.options",
    "fixed_extra_body",
    "upstreams.<name>",
)

PUBLIC_ALIAS = "public-audit-model"
ANTHROPIC_ALIAS = "public-anthropic-model"
OPENAI_UPSTREAM = f"{UPSTREAM_NAME}-openai"
ANTHROPIC_UPSTREAM = f"{UPSTREAM_NAME}-anthropic"
ABSENT_UPSTREAM = f"{UPSTREAM_NAME}-absent"
BROKEN_ALIAS = "public-broken-route"


# --------------------------------------------------------------------------------------
# scanner
# --------------------------------------------------------------------------------------


def _assert_clean(text: str, forbidden: tuple[str, ...], *, surface: str) -> None:
    """Fail naming the surface and every sentinel that reached it.

    Matching is case-insensitive on purpose: URL and header positions lowercase what they
    carry (httpx logs ``https://sentinel-upstream-host-9de4.internal/...``), so a
    case-sensitive scan would walk straight past a credential leaked through one.
    """
    haystack = text.lower()
    leaked = sorted({sentinel for sentinel in forbidden if sentinel.lower() in haystack})
    assert not leaked, f"{surface} leaked {leaked}"


def _response_body_text(response: httpx.Response) -> str:
    return response.text


def _response_header_text(response: httpx.Response) -> str:
    parts = [repr(response.headers), repr(dict(response.headers))]
    parts.extend(f"{name}: {value}" for name, value in response.headers.multi_items())
    return "\n".join(parts)


def _record_text(records: list[logging.LogRecord]) -> str:
    parts: list[str] = []
    for record in records:
        parts.append(record.getMessage())
        parts.append(logging.Formatter("%(name)s %(levelname)s %(message)s").format(record))
        # ``extra`` fields land on the record instance, not in the formatted message: a
        # structured logger ships them and a message-only scan would never see them.
        parts.append(repr(record.__dict__))
        parts.append(repr(record.args))
    return "\n".join(parts)


def _exception_text(exc: BaseException) -> str:
    """Every string form of an exception, including its ``args`` and the chained cause.

    ``__cause__`` is deliberately excluded from client-facing assertions -- the cause is
    the server-side diagnostic -- but ``str``/``repr`` of the raised exception itself must
    be clean, since that is what handlers and generic error reporters serialize.
    """
    return "\n".join([str(exc), repr(exc), repr(exc.args)])


# --------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------

_STRATEGY = FixedStrategy(samples=1, max_samples=1)


def _deployment() -> SimpleNamespace:
    return SimpleNamespace(
        strategy=_STRATEGY,
        cost=0.0,
        spec={"type": "fixed", "samples": 1, "max_samples": 1},
    )


def _upstream_config(name: str, provider: str) -> UpstreamConfig:
    return UpstreamConfig(
        name=name,
        base_url=f"https://{UPSTREAM_HOST}.internal/v1",
        api_key=UPSTREAM_KEY,
        max_connections=2,
        max_response_bytes=65_536,
        provider=provider,
    )


def _route(alias: str, upstream: str) -> ModelRoute:
    return ModelRoute(
        alias=alias,
        upstream=upstream,
        upstream_model=UPSTREAM_MODEL,
        deployment=_deployment(),
        plan=SimpleNamespace(policy="fixed-1", family="fixed"),
        extractor="exact-content",
        max_samples=1,
        max_completion_tokens=64,
        options={"temperature": 0.5},
        allow_client_overrides=False,
        allowed_strategies=frozenset(),
    )


def _gateway_config(
    *,
    request_timeout: float = 5.0,
    queue_timeout: float = 1.0,
    sessions: int = 2,
) -> GatewayConfig:
    return GatewayConfig(
        inbound_api_keys=(INBOUND_KEY,),
        request_timeout_s=request_timeout,
        queue_timeout_s=queue_timeout,
        max_concurrent_sessions=sessions,
        upstreams={
            OPENAI_UPSTREAM: _upstream_config(OPENAI_UPSTREAM, "openai"),
            ANTHROPIC_UPSTREAM: _upstream_config(ANTHROPIC_UPSTREAM, "anthropic"),
        },
        models={
            PUBLIC_ALIAS: _route(PUBLIC_ALIAS, OPENAI_UPSTREAM),
            ANTHROPIC_ALIAS: _route(ANTHROPIC_ALIAS, ANTHROPIC_UPSTREAM),
            # Routes an upstream the running app never bound: reaching it raises a
            # KeyError carrying the operator's upstream id straight into the generic
            # 500 handler.
            BROKEN_ALIAS: _route(BROKEN_ALIAS, ABSENT_UPSTREAM),
        },
    )


def _upstream_payload(**changes: Any) -> dict[str, Any]:
    """A valid OpenAI-shaped upstream response whose every field is a sentinel."""
    payload: dict[str, Any] = {
        # The upstream's own request id is a credential-shaped value on purpose: it must
        # not be relayed to the client.
        "id": f"upstream-req-{UPSTREAM_KEY}",
        "model": UPSTREAM_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": COMPLETION, "refusal": None},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    payload.update(changes)
    return payload


#: Upstream response headers are sentinel-laden: none may be relayed to the client.
_UPSTREAM_HEADERS = {
    "x-upstream-account": UPSTREAM_KEY,
    "x-upstream-host": UPSTREAM_HOST,
}


def _mock_upstreams(
    handler: Any,
    *,
    queue_timeout: float = 1.0,
) -> dict[str, OpenAIUpstream]:
    transport = httpx.MockTransport(handler)
    config = _gateway_config()
    return {
        name: OpenAIUpstream(item, queue_timeout, transport=transport)
        for name, item in config.upstreams.items()
    }


class _DoubleUpstream:
    """An in-process upstream double for paths that never reach the wire."""

    def __init__(self, *, raises: BaseException | None = None, sleep_s: float = 0.0) -> None:
        self.raises = raises
        self.sleep_s = sleep_s
        self.calls = 0

    async def sample(self, body: Any, **kwargs: Any) -> Any:
        del body, kwargs
        self.calls += 1
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.raises is not None:
            raise self.raises
        raise AssertionError("the double must be scripted")

    async def close(self) -> None:
        return None


@contextlib.contextmanager
def _client(
    config: GatewayConfig,
    upstreams: dict[str, Any],
    *,
    raise_server_exceptions: bool = True,
) -> Iterator[tuple[TestClient, FastAPI]]:
    app = create_app(config, strategies={"fixed": _STRATEGY}, upstreams=upstreams)
    with TestClient(app, raise_server_exceptions=raise_server_exceptions) as client:
        yield client, app


def _body(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": PUBLIC_ALIAS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
    }
    payload.update(changes)
    return payload


def _headers(token: str = INBOUND_KEY) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def _post(client: TestClient, payload: dict[str, Any], **kwargs: Any) -> httpx.Response:
    return client.post("/v1/chat/completions", json=payload, headers=_headers(), **kwargs)


# --------------------------------------------------------------------------------------
# exercised paths
# --------------------------------------------------------------------------------------


def _ok_handler(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(200, json=_upstream_payload(), headers=_UPSTREAM_HEADERS)


def _case_success() -> httpx.Response:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        return _post(client, _body())


def _case_401_bad_credential() -> httpx.Response:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        # A near-miss of the configured key: the presented credential must not be echoed
        # back either, so the sentinel appears on both sides of the comparison.
        return client.post(
            "/v1/chat/completions",
            json=_body(),
            headers=_headers(f"{INBOUND_KEY}-wrong"),
        )


def _case_404_unknown_model() -> httpx.Response:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        return _post(client, _body(model=CLIENT_MODEL))


def _case_404_unknown_route() -> httpx.Response:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        return client.post(f"/v1/{CLIENT_MODEL}", json=_body(), headers=_headers())


def _case_400_invalid_request() -> httpx.Response:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        # A schema violation whose rejected input is itself a sentinel: pydantic's
        # validation error names the offending value, and the handler must drop it.
        return _post(client, _body(temperature=PROMPT))


def _case_400_adapter_refusal() -> httpx.Response:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_upstream_payload(), headers=_UPSTREAM_HEADERS)

    with _client(_gateway_config(), _mock_upstreams(handler)) as (client, _):
        response = _post(client, _body(model=ANTHROPIC_ALIAS, seed=7))
    assert calls == 0, "a refused translation must never reach the upstream"
    return response


def _case_413_oversized_body() -> httpx.Response:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        filler = PROMPT * ((_MAX_BODY_BYTES // len(PROMPT)) + 16)
        return _post(client, _body(messages=[{"role": "user", "content": filler}]))


def _case_429_gateway_overloaded() -> httpx.Response:
    # A zero admission budget refuses deterministically without any timing race.
    config = _gateway_config(queue_timeout=0.0)
    upstreams = {
        OPENAI_UPSTREAM: _DoubleUpstream(raises=AssertionError("must not be reached")),
        ANTHROPIC_UPSTREAM: _DoubleUpstream(raises=AssertionError("must not be reached")),
    }
    with _client(config, upstreams) as (client, _):
        return _post(client, _body())


def _case_429_upstream_overloaded() -> httpx.Response:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            429,
            json={"error": {"message": f"{UPSTREAM_KEY} over quota on {UPSTREAM_HOST}"}},
            # A sentinel Retry-After must be rejected, not relayed into a client header.
            headers={"retry-after": UPSTREAM_KEY},
        )

    with _client(_gateway_config(), _mock_upstreams(handler)) as (client, _):
        return _post(client, _body())


def _case_502_invalid_upstream_response() -> httpx.Response:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        broken = _upstream_payload(usage={"prompt_tokens": 11, "total_tokens": 14})
        broken["detail"] = f"{UPSTREAM_KEY} at {UPSTREAM_HOST} serving {UPSTREAM_MODEL}"
        return httpx.Response(200, json=broken, headers=_UPSTREAM_HEADERS)

    with _client(_gateway_config(), _mock_upstreams(handler)) as (client, _):
        return _post(client, _body())


def _case_502_upstream_rejected() -> httpx.Response:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            401,
            json={"error": {"message": f"invalid api key {UPSTREAM_KEY}"}},
            headers=_UPSTREAM_HEADERS,
        )

    with _client(_gateway_config(), _mock_upstreams(handler)) as (client, _):
        return _post(client, _body())


def _case_504_upstream_timeout() -> httpx.Response:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            f"timed out reading from {UPSTREAM_HOST} with key {UPSTREAM_KEY}",
            request=request,
        )

    with _client(_gateway_config(), _mock_upstreams(handler)) as (client, _):
        return _post(client, _body())


def _case_504_gateway_timeout() -> httpx.Response:
    config = _gateway_config(request_timeout=0.01)
    upstreams = {
        OPENAI_UPSTREAM: _DoubleUpstream(sleep_s=30.0),
        ANTHROPIC_UPSTREAM: _DoubleUpstream(sleep_s=30.0),
    }
    with _client(config, upstreams) as (client, _):
        return _post(client, _body())


def _case_500_internal() -> httpx.Response:
    # An unexpected upstream failure carrying every sentinel it could possibly carry.
    detail = f"{UPSTREAM_KEY} {UPSTREAM_HOST} {UPSTREAM_MODEL} {PROMPT} {COMPLETION}"
    upstreams = {
        OPENAI_UPSTREAM: _DoubleUpstream(raises=RuntimeError(detail)),
        ANTHROPIC_UPSTREAM: _DoubleUpstream(raises=RuntimeError(detail)),
    }
    with _client(_gateway_config(), upstreams) as (client, _):
        return _post(client, _body())


def _case_500_unhandled() -> httpx.Response:
    # Routes a never-bound upstream: the KeyError names the operator's upstream id and
    # escapes to the generic handler, the one path that is not a GatewayError.
    with _client(
        _gateway_config(),
        _mock_upstreams(_ok_handler),
        raise_server_exceptions=False,
    ) as (client, _):
        return _post(client, _body(model=BROKEN_ALIAS))


#: Every client-visible path, by the status it must produce.
ERROR_CASES: dict[str, tuple[int, Any]] = {
    "401-invalid-credential": (401, _case_401_bad_credential),
    "404-unknown-model": (404, _case_404_unknown_model),
    "404-unknown-route": (404, _case_404_unknown_route),
    "400-invalid-request": (400, _case_400_invalid_request),
    "400-adapter-refusal": (400, _case_400_adapter_refusal),
    "413-oversized-body": (413, _case_413_oversized_body),
    "429-gateway-overloaded": (429, _case_429_gateway_overloaded),
    "429-upstream-overloaded": (429, _case_429_upstream_overloaded),
    "502-invalid-upstream-response": (502, _case_502_invalid_upstream_response),
    "502-upstream-auth-error": (502, _case_502_upstream_rejected),
    "504-upstream-timeout": (504, _case_504_upstream_timeout),
    "504-gateway-timeout": (504, _case_504_gateway_timeout),
    "500-internal": (500, _case_500_internal),
    "500-unhandled": (500, _case_500_unhandled),
}

#: Sentinels a specific path additionally injects into client-controlled positions.
_CASE_EXTRA_FORBIDDEN: dict[str, tuple[str, ...]] = {
    # Echoing the caller's own model name is not a credential leak, but a constant error
    # body is the invariant: nothing the caller sends comes back.
    "404-unknown-model": (CLIENT_MODEL,),
    "404-unknown-route": (CLIENT_MODEL,),
    "400-invalid-request": (PROMPT,),
    "400-adapter-refusal": REFUSAL_TELLS,
}


def _forbidden_for(case: str) -> tuple[str, ...]:
    return CLIENT_FORBIDDEN + _CASE_EXTRA_FORBIDDEN.get(case, ())


# --------------------------------------------------------------------------------------
# surface: HTTP response bodies
# --------------------------------------------------------------------------------------


def test_success_response_body_carries_no_credential_or_infrastructure() -> None:
    response = _case_success()

    assert response.status_code == 200
    payload = response.json()
    # Proof the path really ran: the caller's own completion is returned.
    assert payload["choices"][0]["message"]["content"] == COMPLETION
    assert payload["model"] == PUBLIC_ALIAS
    _assert_clean(
        _response_body_text(response),
        CLIENT_FORBIDDEN,
        surface="http.response.body.success",
    )


def test_success_response_headers_carry_no_credential_or_infrastructure() -> None:
    response = _case_success()

    assert response.headers["x-branchpilot-samples"] == "1"
    _assert_clean(
        _response_header_text(response),
        CLIENT_FORBIDDEN,
        surface="http.response.headers.success",
    )


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_error_response_body_carries_no_secret(case: str) -> None:
    expected_status, exercise = ERROR_CASES[case]
    response = exercise()

    assert response.status_code == expected_status
    assert response.json()["error"]["message"]
    _assert_clean(
        _response_body_text(response),
        _forbidden_for(case),
        surface=f"http.response.body.error[{case}]",
    )


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_error_response_headers_carry_no_secret(case: str) -> None:
    expected_status, exercise = ERROR_CASES[case]
    response = exercise()

    assert response.status_code == expected_status
    _assert_clean(
        _response_header_text(response),
        _forbidden_for(case),
        surface=f"http.response.headers.error[{case}]",
    )


def test_unauthenticated_error_body_does_not_confirm_the_configured_key() -> None:
    response = _case_401_bad_credential()

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["message"] == "Invalid authentication credentials."


def test_healthz_body_and_headers_carry_no_secret() -> None:
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, _):
        response = client.get("/healthz")

    assert response.status_code == 200
    _assert_clean(
        _response_body_text(response) + _response_header_text(response),
        CLIENT_FORBIDDEN,
        surface="http.response.body.healthz",
    )


def test_metrics_surface_is_still_absent() -> None:
    """Tripwire: when a metrics endpoint lands, scan it here and in FUTURE_SURFACES."""
    with _client(_gateway_config(), _mock_upstreams(_ok_handler)) as (client, app):
        paths = {route.path for route in app.routes}
        probes = {path: client.get(path) for path in ("/metrics", "/v1/metrics")}

    assert not {path for path in paths if "metric" in path}
    for path, response in probes.items():
        assert response.status_code in {401, 404}, (
            f"{path} now serves content: add it to SCANNED_SURFACES, drop "
            f"'metrics.prometheus' from FUTURE_SURFACES, and scan its exposition text "
            f"for {CLIENT_FORBIDDEN}"
        )
        _assert_clean(
            _response_body_text(response) + _response_header_text(response),
            CLIENT_FORBIDDEN,
            surface=f"metrics.prometheus[{path}]",
        )


# --------------------------------------------------------------------------------------
# surface: the Batch 1 refusal leak, both directions
# --------------------------------------------------------------------------------------


def test_adapter_refusal_hides_provider_identity_and_operator_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        response = _case_400_adapter_refusal()

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["message"] == "The request is not supported by the requested model."
    assert payload["error"]["code"] == "upstream_unsupported_request"
    _assert_clean(
        _response_body_text(response) + _response_header_text(response),
        CLIENT_FORBIDDEN + REFUSAL_TELLS,
        surface="http.response.body.error[400-adapter-refusal]",
    )


def test_adapter_refusal_keeps_the_actionable_detail_in_the_operator_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The positive half: a sanitized body is worthless if the operator loses the fix."""
    with caplog.at_level(logging.DEBUG):
        response = _case_400_adapter_refusal()

    assert response.status_code == 400
    logged = _record_text(caplog.records)
    for required in ("anthropic", "fix:", "seed", "upstream_request_unsupported"):
        assert required in logged, f"the operator log lost {required!r}"
    assert ANTHROPIC_UPSTREAM in logged, "the log must name which upstream refused"
    _assert_clean(logged, LOG_FORBIDDEN, surface="logging.record.__dict__[refusal]")


# --------------------------------------------------------------------------------------
# surface: log records
# --------------------------------------------------------------------------------------


def test_success_log_records_carry_no_credential_or_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        response = _case_success()

    assert response.status_code == 200
    logged = _record_text(caplog.records)
    assert "gateway_request_completed" in logged, "the success path must be observable"
    _assert_clean(logged, LOG_FORBIDDEN, surface="logging.record.__dict__[success]")


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_error_log_records_carry_no_credential_or_content(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    expected_status, exercise = ERROR_CASES[case]
    with caplog.at_level(logging.DEBUG):
        response = exercise()

    assert response.status_code == expected_status
    _assert_clean(
        _record_text(caplog.records),
        LOG_FORBIDDEN,
        surface=f"logging.record.__dict__[{case}]",
    )


# --------------------------------------------------------------------------------------
# surface: reprs of configuration and middleware
# --------------------------------------------------------------------------------------


def test_config_reprs_redact_every_credential() -> None:
    config = _gateway_config()
    upstream = config.upstreams[OPENAI_UPSTREAM]
    route = config.models[PUBLIC_ALIAS]
    keys = InboundAPIKeys((INBOUND_KEY,))

    surfaces = {
        "repr.GatewayConfig": [repr(config), str(config), f"{config}", f"{config!r}"],
        "repr.UpstreamConfig": [repr(upstream), str(upstream), f"{upstream}"],
        "repr.ModelRoute": [repr(route), str(route)],
        "repr.InboundAPIKeys": [repr(keys), str(keys), f"{keys}"],
        # Nesting is where redaction usually dies: a container repr recurses into
        # __repr__ of every element.
        "repr.nested": [
            repr({"config": config, "keys": keys}),
            repr([config, upstream, route, keys]),
            repr(dict(config.upstreams)),
            repr(dict(config.models)),
        ],
    }
    for surface, texts in surfaces.items():
        for text in texts:
            _assert_clean(text, CREDENTIALS, surface=surface)

    assert repr(keys) == "InboundAPIKeys(<redacted>)"
    # The repr must still be useful to an operator debugging a route.
    assert UPSTREAM_MODEL in repr(route)


def test_middleware_reprs_redact_every_credential() -> None:
    config = _gateway_config()
    middleware = _BoundaryMiddleware(
        lambda scope, receive, send: None,
        config.inbound_api_keys,
        config.body_read_timeout_s,
    )
    _assert_clean(repr(middleware), CREDENTIALS, surface="repr._BoundaryMiddleware")
    _assert_clean(repr(middleware.__dict__), CREDENTIALS, surface="repr._BoundaryMiddleware")

    with _client(config, _mock_upstreams(_ok_handler)) as (_, app):
        texts = [
            repr(app.user_middleware),
            repr(app.middleware_stack),
            "\n".join(repr(item) for item in app.user_middleware),
            repr(app.state.__dict__),
        ]
    for text in texts:
        _assert_clean(text, CREDENTIALS, surface="repr.app.user_middleware")


# --------------------------------------------------------------------------------------
# surface: raised exceptions
# --------------------------------------------------------------------------------------


def _raised_gateway_errors() -> dict[str, GatewayError]:
    """Every GatewayError the upstream client can raise, collected from the real code."""
    errors: dict[str, GatewayError] = {}

    def refusal(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a refused translation must never reach the upstream")

    def invalid(request: httpx.Request) -> httpx.Response:
        del request
        broken = _upstream_payload(usage={"prompt_tokens": 11, "total_tokens": 14})
        return httpx.Response(200, json=broken, headers=_UPSTREAM_HEADERS)

    def unauthorized(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            401, json={"error": {"message": UPSTREAM_KEY}}, headers=_UPSTREAM_HEADERS
        )

    def overloaded(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, json={"error": {"message": UPSTREAM_KEY}})

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout(f"{UPSTREAM_HOST} unreachable", request=request)

    scripted = {
        "adapter-refusal": ("anthropic", refusal, {"seed": 7}),
        "invalid-upstream-response": ("openai", invalid, {}),
        "upstream-auth-error": ("openai", unauthorized, {}),
        "upstream-overloaded": ("openai", overloaded, {}),
        "upstream-timeout": ("openai", timeout, {}),
    }

    async def scenario() -> None:
        for name, (provider, handler, extra) in scripted.items():
            upstream = OpenAIUpstream(
                _upstream_config(f"{UPSTREAM_NAME}-{provider}", provider),
                0.05,
                transport=httpx.MockTransport(handler),
            )
            body: dict[str, Any] = {
                "upstream_model": UPSTREAM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT},
                ],
                "max_completion_tokens": 64,
                **extra,
            }
            with pytest.raises(GatewayError) as caught:
                await upstream.sample(
                    body,
                    extractor=lambda text: text.strip() or None,
                    public_request_id="bp-public",
                    sample_index=1,
                )
            errors[name] = caught.value
            await upstream.close()

    asyncio.run(scenario())
    return errors


def test_gateway_error_strings_carry_no_secret(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        errors = _raised_gateway_errors()

    assert set(errors) == {
        "adapter-refusal",
        "invalid-upstream-response",
        "upstream-auth-error",
        "upstream-overloaded",
        "upstream-timeout",
    }
    for name, error in errors.items():
        _assert_clean(
            _exception_text(error),
            CLIENT_FORBIDDEN + CONTENT + REFUSAL_TELLS,
            surface=f"str/repr.GatewayError[{name}]",
        )
        assert error.message
        assert error.retry_after is None

    _assert_clean(_record_text(caplog.records), LOG_FORBIDDEN, surface="logging.record.__dict__")


def test_config_error_strings_carry_no_secret(tmp_path: Path) -> None:
    literal = tmp_path / "literal-secret.json"
    literal.write_text(
        json.dumps(
            {
                "inbound_api_key_envs": ["BP_INBOUND"],
                "upstreams": {
                    OPENAI_UPSTREAM: {
                        "base_url": f"https://{UPSTREAM_HOST}.internal/v1",
                        # An operator pasting a literal key into the config file must not
                        # see it echoed back by the loader's validation report.
                        "api_key": UPSTREAM_KEY,
                        "api_key_env": "BP_UPSTREAM",
                    }
                },
                "models": {
                    PUBLIC_ALIAS: {
                        "upstream": OPENAI_UPSTREAM,
                        "upstream_model": UPSTREAM_MODEL,
                        "plan_path": "plan.json",
                        "max_completion_tokens": 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    environment = {"BP_INBOUND": INBOUND_KEY, "BP_UPSTREAM": UPSTREAM_KEY}

    with pytest.raises(ConfigError) as caught:
        load_gateway_config(literal, environ=environment)
    _assert_clean(
        _exception_text(caught.value),
        CREDENTIALS,
        surface="str/repr.ConfigError[literal-secret]",
    )
    assert "api_key" in str(caught.value), "the operator must still learn which field"

    missing = tmp_path / "missing-env.json"
    payload = json.loads(literal.read_text(encoding="utf-8"))
    del payload["upstreams"][OPENAI_UPSTREAM]["api_key"]
    missing.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError) as absent:
        load_gateway_config(missing, environ={"BP_INBOUND": INBOUND_KEY})
    _assert_clean(
        _exception_text(absent.value),
        CREDENTIALS,
        surface="str/repr.ConfigError[missing-env]",
    )
    assert "BP_UPSTREAM" in str(absent.value), "the operator must still learn which variable"


# --------------------------------------------------------------------------------------
# surface: ingest records, reports, and mid-iteration exceptions
# --------------------------------------------------------------------------------------


def _log_line(*, prompt_tokens: Any = 11, index: int = 0) -> dict[str, Any]:
    return {
        "request": {
            "model": f"gpt-4o-mini-{index}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
        },
        "response": {
            "id": f"chatcmpl-{index}",
            "created": 1775030400,
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": COMPLETION}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        },
    }


def _write_log(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_request_record_and_report_reprs_carry_no_content(tmp_path: Path) -> None:
    path = _write_log(tmp_path / "traffic.jsonl", [_log_line(index=i) for i in range(3)])

    stream = read_requests(path, format="openai-jsonl")
    records = list(stream)
    report = stream.report()

    assert len(records) == 3 and report.parsed == 3
    texts = [repr(records), str(records), repr(report), str(report)]
    for record in records:
        texts.extend([repr(record), str(record)])
        texts.extend(
            repr(getattr(record, name))
            for name in ("id", "model", "provider", "messages_hash", "system_prefix_hash")
        )
    for text in texts:
        _assert_clean(text, CONTENT, surface="repr.RequestRecord")
    _assert_clean(repr(report) + str(report), CONTENT, surface="repr.IngestReport")


def test_ingest_exception_raised_mid_iteration_carries_no_content(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path / "half-broken.jsonl",
        [_log_line(index=0), _log_line(index=1, prompt_tokens=f"many {PROMPT}")],
    )

    stream = read_requests(path, format="openai-jsonl")
    assert next(stream) is not None
    with pytest.raises(MalformedRecordError) as caught:
        next(stream)

    _assert_clean(
        _exception_text(caught.value),
        CONTENT,
        surface="str/repr.ingest-exception-mid-iteration",
    )
    message = str(caught.value)
    assert "record 1 (line 2)" in message and "fix:" in message
    _assert_clean(repr(stream.report()), CONTENT, surface="repr.IngestReport")


# --------------------------------------------------------------------------------------
# the suite's own guard rails
# --------------------------------------------------------------------------------------


def test_surface_registry_is_coherent() -> None:
    assert len(set(SCANNED_SURFACES)) == len(SCANNED_SURFACES)
    assert len(set(FUTURE_SURFACES)) == len(FUTURE_SURFACES)
    assert not set(SCANNED_SURFACES) & set(FUTURE_SURFACES)
    assert len({*CREDENTIALS, *INFRASTRUCTURE, *CONTENT, CLIENT_MODEL}) == 9


def test_scanner_detects_a_planted_sentinel() -> None:
    """Negative control: a scanner that cannot fail proves nothing."""
    for sentinel in (*CREDENTIALS, *INFRASTRUCTURE, *CONTENT):
        with pytest.raises(AssertionError) as caught:
            _assert_clean(f"prefix {sentinel} suffix", CLIENT_FORBIDDEN + CONTENT, surface="probe")
        assert sentinel in str(caught.value)
        assert "probe" in str(caught.value)

    record = logging.LogRecord("probe", logging.INFO, __file__, 1, "planted", None, None)
    record.question = PROMPT
    with pytest.raises(AssertionError):
        _assert_clean(_record_text([record]), LOG_FORBIDDEN, surface="probe")
