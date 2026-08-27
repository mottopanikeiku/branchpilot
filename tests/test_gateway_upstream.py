from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from branchpilot.gateway.config import UpstreamConfig
from branchpilot.gateway.upstream import GatewayError, OpenAIUpstream, parse_chat_completion


def _response(
    text: object = "reasoning\n\\boxed{42}",
    *,
    finish_reason: object = "stop",
    prompt_tokens: object = 10,
    completion_tokens: object = 3,
    total_tokens: object = 13,
    choices: object = None,
    logprobs: object = None,
):
    if choices is None:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=text, refusal=None),
                finish_reason=finish_reason,
                logprobs=logprobs,
            )
        ]
    return SimpleNamespace(
        id="private-upstream-id",
        model="private/model",
        choices=choices,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
        _request_id="provider-request-id",
    )


class _Completions:
    def __init__(self, scripted: list[object]) -> None:
        self.scripted = scripted
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object):
        self.calls.append(kwargs)
        value = self.scripted.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeClient:
    def __init__(self, scripted: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_Completions(scripted))
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


def _config(max_connections: int = 2, max_response_bytes: int = 4_194_304) -> UpstreamConfig:
    return UpstreamConfig(
        name="local",
        base_url="http://inference.internal/v1",
        api_key="provider-secret",
        max_connections=max_connections,
        max_response_bytes=max_response_bytes,
        fixed_extra_body={"guided_decoding_backend": "x"},
    )


def test_parses_exact_response_usage_and_logprobs() -> None:
    response = _response(
        logprobs=SimpleNamespace(
            content=[
                SimpleNamespace(token="a", logprob=-1.0),
                SimpleNamespace(token="b", logprob=-3.0),
            ]
        )
    )
    parsed = parse_chat_completion(response, lambda text: "42")

    assert parsed.sample.answer == "42"
    assert parsed.sample.mean_logprob == -2.0
    assert parsed.usage.prompt_tokens == 10
    assert parsed.usage.total_tokens == 13
    assert parsed.usage.prompt_tokens_details == {"cached_tokens": 2}
    assert parsed.upstream_request_id == "provider-request-id"


def test_sdk_call_forces_structure_and_closes_once() -> None:
    async def scenario() -> None:
        client = FakeClient([_response()])
        upstream = OpenAIUpstream(_config(), 0.1, client=client)
        result = await upstream.sample(
            {
                "upstream_model": "private/model",
                "messages": [{"role": "user", "content": "question"}],
                "n": 9,
                "stream": True,
                "temperature": 0.7,
            },
            extractor=lambda text: "42",
            public_request_id="public-id",
            sample_index=2,
        )
        call = client.chat.completions.calls[0]
        assert result.sample.answer == "42"
        assert call["model"] == "private/model"
        assert call["n"] == 1
        assert call["stream"] is False
        assert call["extra_headers"] == {"X-Request-ID": "public-id:2"}
        assert call["extra_body"] == {"guided_decoding_backend": "x"}
        await upstream.close()
        await upstream.close()
        assert client.close_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[], usage=SimpleNamespace()),
        _response(choices=[]),
        _response(choices=[1, 2]),
        _response(text=None),
        _response(text=["multimodal"]),
        _response(finish_reason="vendor_reason"),
        _response(prompt_tokens=-1),
        _response(completion_tokens=None),
        _response(total_tokens=99),
        _response(prompt_tokens=True),
        _response(logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=float("nan"))])),
    ],
)
def test_malformed_responses_fail_with_sanitized_502(response: object) -> None:
    with pytest.raises(GatewayError) as caught:
        parse_chat_completion(response, lambda text: "42")
    assert caught.value.status_code == 502
    assert caught.value.code == "invalid_upstream_response"
    assert "private" not in caught.value.message


def test_status_and_transport_errors_are_sanitized() -> None:
    async def scenario() -> None:
        request = httpx.Request("POST", "https://secret.internal/v1/chat/completions")
        response = httpx.Response(
            429,
            headers={"retry-after": "3", "x-secret": "leak"},
            request=request,
            json={"error": "secret provider body"},
        )
        errors = [
            openai.APIStatusError("secret provider body", response=response, body=response.json()),
            openai.APIConnectionError(message="secret URL", request=request),
            openai.APITimeoutError(request=request),
        ]
        expected = [(429, "gateway_overloaded"), (502, "upstream_error"), (504, "upstream_timeout")]
        for error, pair in zip(errors, expected, strict=True):
            upstream = OpenAIUpstream(_config(), 0.1, client=FakeClient([error]))
            with pytest.raises(GatewayError) as caught:
                await upstream.sample(
                    {"upstream_model": "m", "messages": []},
                    extractor=lambda text: None,
                    public_request_id="id",
                    sample_index=1,
                )
            assert (caught.value.status_code, caught.value.code) == pair
            assert "secret" not in caught.value.message
        assert caught.value.retry_after is None

    asyncio.run(scenario())


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False
        self.iterations = 0

    async def __aiter__(self):
        self.iterations += 1
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _chat_completion_bytes() -> bytes:
    return json.dumps(
        {
            "id": "private-upstream-id",
            "object": "chat.completion",
            "created": 1,
            "model": "private/model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "reasoning\n\\boxed{42}",
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        },
        separators=(",", ":"),
    ).encode()


@pytest.mark.parametrize("oversize_kind", ["content-length", "chunked"])
def test_production_response_limit_rejects_before_parsing_and_reuses_semaphore(
    oversize_kind: str,
) -> None:
    async def scenario() -> None:
        oversized = _ChunkStream(
            [] if oversize_kind == "content-length" else [b"x" * 300, b"x" * 300]
        )
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            assert request.url == "http://inference.internal/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer provider-secret"
            if calls == 1:
                headers = {"content-length": "513"} if oversize_kind == "content-length" else {}
                return httpx.Response(200, headers=headers, stream=oversized)
            return httpx.Response(200, content=_chat_completion_bytes())

        upstream = OpenAIUpstream(
            _config(max_connections=1, max_response_bytes=512),
            0.01,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(GatewayError) as caught:
            await upstream.sample(
                {"upstream_model": "private/model", "messages": []},
                extractor=lambda text: "42",
                public_request_id="first",
                sample_index=1,
            )
        assert caught.value.status_code == 502
        assert caught.value.code == "invalid_upstream_response"
        assert oversized.closed

        result = await upstream.sample(
            {"upstream_model": "private/model", "messages": []},
            extractor=lambda text: "42",
            public_request_id="second",
            sample_index=1,
        )
        assert result.sample.answer == "42"
        assert calls == 2
        await upstream.close()

    asyncio.run(scenario())


def test_production_rejects_compression_before_decoding_and_reuses_semaphore() -> None:
    async def scenario() -> None:
        compressed_body = gzip.compress(b"x" * 4096)
        assert len(compressed_body) < 512
        compressed = _ChunkStream([compressed_body])
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    headers={
                        "content-encoding": "gzip",
                        "content-length": str(len(compressed_body)),
                    },
                    stream=compressed,
                )
            return httpx.Response(200, content=_chat_completion_bytes())

        upstream = OpenAIUpstream(
            _config(max_connections=1, max_response_bytes=512),
            0.01,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(GatewayError) as caught:
            await upstream.sample(
                {"upstream_model": "private/model", "messages": []},
                extractor=lambda text: "42",
                public_request_id="compressed",
                sample_index=1,
            )
        assert caught.value.code == "invalid_upstream_response"
        assert compressed.iterations == 0
        assert compressed.closed

        result = await upstream.sample(
            {"upstream_model": "private/model", "messages": []},
            extractor=lambda text: "42",
            public_request_id="reused",
            sample_index=1,
        )
        assert result.sample.answer == "42"
        assert calls == 2
        await upstream.close()

    asyncio.run(scenario())


def test_semaphore_admission_timeout_and_cancellation_restore_capacity() -> None:
    class BlockingCompletions:
        def __init__(self) -> None:
            self.event = asyncio.Event()
            self.calls = 0

        async def create(self, **kwargs: object):
            self.calls += 1
            await self.event.wait()
            return _response()

    async def scenario() -> None:
        completions = BlockingCompletions()
        client = FakeClient([])
        client.chat.completions = completions
        upstream = OpenAIUpstream(_config(max_connections=1), 0.01, client=client)
        first = asyncio.create_task(
            upstream.sample(
                {"upstream_model": "m", "messages": []},
                extractor=lambda text: "42",
                public_request_id="first",
                sample_index=1,
            )
        )
        await asyncio.sleep(0)
        with pytest.raises(GatewayError) as caught:
            await upstream.sample(
                {"upstream_model": "m", "messages": []},
                extractor=lambda text: "42",
                public_request_id="second",
                sample_index=1,
            )
        assert caught.value.status_code == 429
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        completions.event.set()
        result = await upstream.sample(
            {"upstream_model": "m", "messages": []},
            extractor=lambda text: "42",
            public_request_id="third",
            sample_index=1,
        )
        assert result.sample.answer == "42"

    asyncio.run(scenario())
