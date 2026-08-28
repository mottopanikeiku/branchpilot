from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from branchpilot.gateway import config as gateway_config
from branchpilot.gateway.config import ConfigError, UpstreamConfig, load_gateway_config
from branchpilot.gateway.providers import (
    PROVIDER_IDS,
    AnthropicAdapter,
    BedrockAdapter,
    CanonicalResponse,
    GeminiAdapter,
    OpenAIAdapter,
    ProviderRequestError,
    ProviderResponseError,
    UnknownProviderError,
    resolve_adapter,
)
from branchpilot.gateway.upstream import GatewayError, OpenAIUpstream
from branchpilot.strategies import FixedStrategy

# Recorded provider response payloads. Every one of them is a document shape the provider
# documents for a single-candidate text completion with prompt caching active.
_OPENAI_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-recorded",
    "object": "chat.completion",
    "created": 1730000000,
    "model": "private/model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "reasoning\n\\boxed{42}", "refusal": None},
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {"token": "4", "logprob": -0.5, "bytes": [52], "top_logprobs": []},
                    {"token": "2", "logprob": -1.5, "bytes": [50], "top_logprobs": []},
                ]
            },
        }
    ],
    "usage": {
        "prompt_tokens": 2060,
        "completion_tokens": 9,
        "total_tokens": 2069,
        "prompt_tokens_details": {"cached_tokens": 1800, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 30,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
}
_ANTHROPIC_RESPONSE: dict[str, Any] = {
    "id": "msg_recorded",
    "type": "message",
    "role": "assistant",
    "model": "private/model",
    "content": [{"type": "text", "text": "reasoning\n\\boxed{42}"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "cache_read_input_tokens": 1800,
        "cache_creation_input_tokens": 248,
        "output_tokens": 9,
    },
}
_GEMINI_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "reasoning\n\\boxed{42}"}]},
            "finishReason": "STOP",
            "avgLogprobs": -1.0,
            "logprobsResult": {
                "chosenCandidates": [
                    {"token": "4", "logProbability": -0.5},
                    {"token": "2", "logProbability": -1.5},
                ]
            },
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 2060,
        "cachedContentTokenCount": 1800,
        "candidatesTokenCount": 9,
        "thoughtsTokenCount": 30,
        "totalTokenCount": 2099,
    },
    "modelVersion": "private/model",
    "responseId": "resp-recorded",
}
_BEDROCK_RESPONSE: dict[str, Any] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "reasoning\n\\boxed{42}"}]}},
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 12,
        "outputTokens": 9,
        "totalTokens": 2069,
        "cacheReadInputTokens": 1800,
        "cacheWriteInputTokens": 248,
    },
    "metrics": {"latencyMs": 903},
}

_ADAPTERS = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
    "bedrock": BedrockAdapter(),
}
_RESPONSES = {
    "openai": _OPENAI_RESPONSE,
    "anthropic": _ANTHROPIC_RESPONSE,
    "gemini": _GEMINI_RESPONSE,
    "bedrock": _BEDROCK_RESPONSE,
}
_CREDENTIAL_HEADERS = {
    "openai": ("authorization", "Bearer provider-secret"),
    "anthropic": ("x-api-key", "provider-secret"),
    "gemini": ("x-goog-api-key", "provider-secret"),
    "bedrock": ("authorization", "Bearer provider-secret"),
}
_PATHS = {
    "openai": "/chat/completions",
    "anthropic": "/messages",
    "gemini": "/models/private%2Fmodel:generateContent",
    "bedrock": "/model/private%2Fmodel/converse",
}
_PAYLOADS: dict[str, dict[str, Any]] = {
    "openai": {
        "model": "private/model",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "question"},
        ],
        "max_completion_tokens": 64,
        "temperature": 0.7,
        "n": 1,
        "stream": False,
    },
    "anthropic": {
        "model": "private/model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "question"}]}],
        "system": [{"type": "text", "text": "be terse"}],
        "temperature": 0.7,
    },
    "gemini": {
        "contents": [{"role": "user", "parts": [{"text": "question"}]}],
        "generationConfig": {
            "candidateCount": 1,
            "maxOutputTokens": 64,
            "temperature": 0.7,
        },
        "systemInstruction": {"parts": [{"text": "be terse"}]},
    },
    "bedrock": {
        "messages": [{"role": "user", "content": [{"text": "question"}]}],
        "system": [{"text": "be terse"}],
        "inferenceConfig": {"maxTokens": 64, "temperature": 0.7},
    },
}
_EXPECTED_USAGE = {
    "openai": (2060, 1800, 9, 2069, {"cached_tokens": 1800, "audio_tokens": 0}),
    "anthropic": (2060, 1800, 9, 2069, {"cached_tokens": 1800, "cache_write_tokens": 248}),
    "gemini": (2060, 1800, 39, 2099, {"cached_tokens": 1800}),
    "bedrock": (2060, 1800, 9, 2069, {"cached_tokens": 1800, "cache_write_tokens": 248}),
}


def _canonical(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": "private/model",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "question"},
        ],
        "max_completion_tokens": 64,
        "temperature": 0.7,
        "n": 1,
        "stream": False,
        "api_key": "provider-secret",
        "request_id": "public-id:1",
    }
    value.update(changes)
    return value


def _without(payload: dict[str, Any], *path: Any) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    target: Any = value
    for name in path[:-1]:
        target = target[name]
    del target[path[-1]]
    return value


def _replace(payload: dict[str, Any], value: Any, *path: Any) -> dict[str, Any]:
    changed = copy.deepcopy(payload)
    target: Any = changed
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    return changed


@pytest.mark.parametrize("provider", sorted(_ADAPTERS))
def test_recorded_response_round_trips_to_canonical(provider: str) -> None:
    parsed = _ADAPTERS[provider].parse_response(_RESPONSES[provider])
    prompt, cached, completion, total, prompt_details = _EXPECTED_USAGE[provider]

    assert parsed.content == "reasoning\n\\boxed{42}"
    assert parsed.finish_reason == "stop"
    assert parsed.refusal is None
    assert parsed.prompt_tokens == prompt
    assert parsed.cached_prompt_tokens == cached
    assert parsed.completion_tokens == completion
    assert parsed.total_tokens == total
    assert parsed.total_tokens == parsed.prompt_tokens + parsed.completion_tokens
    assert dict(parsed.prompt_tokens_details) == prompt_details


def test_openai_round_trip_preserves_logprobs_and_reasoning_tokens() -> None:
    parsed = _ADAPTERS["openai"].parse_response(_OPENAI_RESPONSE)

    assert parsed.upstream_request_id == "chatcmpl-recorded"
    assert dict(parsed.completion_tokens_details) == {
        "reasoning_tokens": 30,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }
    assert parsed.logprobs is not None
    assert [entry["logprob"] for entry in parsed.logprobs["content"]] == [-0.5, -1.5]


def test_anthropic_round_trip_folds_cache_tokens_into_the_prompt_total() -> None:
    parsed = _ADAPTERS["anthropic"].parse_response(_ANTHROPIC_RESPONSE)

    assert parsed.upstream_request_id == "msg_recorded"
    assert parsed.prompt_tokens == 12 + 1800 + 248
    assert parsed.logprobs is None
    assert parsed.completion_tokens_details is None


def test_gemini_round_trip_bills_thoughts_as_completion_and_translates_logprobs() -> None:
    parsed = _ADAPTERS["gemini"].parse_response(_GEMINI_RESPONSE)

    assert parsed.upstream_request_id == "resp-recorded"
    assert parsed.completion_tokens == 9 + 30
    assert dict(parsed.completion_tokens_details) == {"reasoning_tokens": 30}
    assert parsed.logprobs == {
        "content": [{"token": "4", "logprob": -0.5}, {"token": "2", "logprob": -1.5}]
    }


def test_bedrock_round_trip_accepts_both_documented_total_conventions() -> None:
    inside = _replace(
        _replace(_BEDROCK_RESPONSE, 2060, "usage", "inputTokens"), 2069, "usage", "totalTokens"
    )
    alongside = _ADAPTERS["bedrock"].parse_response(_BEDROCK_RESPONSE)
    folded = _ADAPTERS["bedrock"].parse_response(inside)

    assert alongside.prompt_tokens == 2060
    assert folded.prompt_tokens == 2060
    assert alongside.cached_prompt_tokens == folded.cached_prompt_tokens == 1800
    assert alongside.upstream_request_id is None


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        ("openai", _without(_OPENAI_RESPONSE, "usage")),
        ("openai", _without(_OPENAI_RESPONSE, "usage", "completion_tokens")),
        ("openai", _replace(_OPENAI_RESPONSE, None, "usage", "completion_tokens")),
        ("openai", _replace(_OPENAI_RESPONSE, "9", "usage", "completion_tokens")),
        ("openai", _replace(_OPENAI_RESPONSE, 99, "usage", "total_tokens")),
        ("anthropic", _without(_ANTHROPIC_RESPONSE, "usage")),
        ("anthropic", _without(_ANTHROPIC_RESPONSE, "usage", "output_tokens")),
        ("anthropic", _replace(_ANTHROPIC_RESPONSE, -1, "usage", "output_tokens")),
        ("anthropic", _replace(_ANTHROPIC_RESPONSE, "1800", "usage", "cache_read_input_tokens")),
        ("gemini", _without(_GEMINI_RESPONSE, "usageMetadata")),
        ("gemini", _without(_GEMINI_RESPONSE, "usageMetadata", "candidatesTokenCount")),
        ("gemini", _replace(_GEMINI_RESPONSE, 2069, "usageMetadata", "totalTokenCount")),
        ("bedrock", _without(_BEDROCK_RESPONSE, "usage")),
        ("bedrock", _without(_BEDROCK_RESPONSE, "usage", "outputTokens")),
        ("bedrock", _replace(_BEDROCK_RESPONSE, 4096, "usage", "totalTokens")),
    ],
)
def test_absent_or_incoherent_usage_is_refused(provider: str, payload: dict[str, Any]) -> None:
    with pytest.raises(ProviderResponseError) as caught:
        _ADAPTERS[provider].parse_response(payload)
    assert "fix:" in str(caught.value)
    assert "boxed{42}" not in str(caught.value)


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        ("openai", _replace(_OPENAI_RESPONSE, "vendor_reason", "choices", 0, "finish_reason")),
        ("openai", _replace(_OPENAI_RESPONSE, [], "choices")),
        ("openai", _replace(_OPENAI_RESPONSE, None, "choices", 0, "message", "content")),
        ("anthropic", _replace(_ANTHROPIC_RESPONSE, "pause_turn", "stop_reason")),
        (
            "anthropic",
            _replace(_ANTHROPIC_RESPONSE, [{"type": "thinking", "thinking": "..."}], "content"),
        ),
        (
            "gemini",
            _replace(_GEMINI_RESPONSE, "MALFORMED_FUNCTION_CALL", "candidates", 0, "finishReason"),
        ),
        (
            "gemini",
            _replace(
                _GEMINI_RESPONSE,
                [{"text": "hidden", "thought": True}],
                "candidates",
                0,
                "content",
                "parts",
            ),
        ),
        ("bedrock", _replace(_BEDROCK_RESPONSE, "malformed_model_output", "stopReason")),
        (
            "bedrock",
            _replace(
                _BEDROCK_RESPONSE,
                [{"reasoningContent": {"reasoningText": {"text": "hidden"}}}],
                "output",
                "message",
                "content",
            ),
        ),
    ],
)
def test_untranslatable_response_shapes_are_refused(provider: str, payload: dict[str, Any]) -> None:
    with pytest.raises(ProviderResponseError) as caught:
        _ADAPTERS[provider].parse_response(payload)
    assert "fix:" in str(caught.value)
    assert "boxed{42}" not in str(caught.value)
    assert "hidden" not in str(caught.value)


def test_canonical_response_refuses_invented_usage() -> None:
    with pytest.raises(ProviderResponseError) as inconsistent:
        CanonicalResponse(
            content="answer",
            finish_reason="stop",
            refusal=None,
            logprobs=None,
            prompt_tokens=10,
            cached_prompt_tokens=None,
            completion_tokens=3,
            total_tokens=99,
            upstream_request_id=None,
        )
    with pytest.raises(ProviderResponseError) as unbilled:
        CanonicalResponse(
            content="answer",
            finish_reason="stop",
            refusal=None,
            logprobs=None,
            prompt_tokens=10,
            cached_prompt_tokens=None,
            completion_tokens=0,
            total_tokens=10,
            upstream_request_id=None,
        )
    with pytest.raises(ProviderResponseError) as impossible_cache:
        CanonicalResponse(
            content="answer",
            finish_reason="stop",
            refusal=None,
            logprobs=None,
            prompt_tokens=10,
            cached_prompt_tokens=11,
            completion_tokens=3,
            total_tokens=13,
            upstream_request_id=None,
        )
    for caught in (inconsistent, unbilled, impossible_cache):
        assert "fix:" in str(caught.value)
    assert "never synthesizes or estimates" in str(unbilled.value)


def test_canonical_response_refuses_non_finite_logprobs() -> None:
    with pytest.raises(ProviderResponseError):
        CanonicalResponse(
            content="answer",
            finish_reason="stop",
            refusal=None,
            logprobs={"content": [{"token": "a", "logprob": float("nan")}]},
            prompt_tokens=10,
            cached_prompt_tokens=None,
            completion_tokens=3,
            total_tokens=13,
            upstream_request_id=None,
        )


@pytest.mark.parametrize("provider", sorted(_ADAPTERS))
def test_build_request_translates_the_canonical_request(provider: str) -> None:
    request = _ADAPTERS[provider].build_request(_canonical())
    name, credential = _CREDENTIAL_HEADERS[provider]
    headers = {key.lower(): value for key, value in request.headers.items()}

    assert request.path == _PATHS[provider]
    assert dict(request.payload) == _PAYLOADS[provider]
    assert headers[name] == credential
    assert headers["accept-encoding"] == "identity"
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "public-id:1"
    assert "api_key" not in request.payload
    assert "request_id" not in request.payload
    assert "provider-secret" not in repr(request)


@pytest.mark.parametrize(
    ("provider", "changes"),
    [
        ("anthropic", {"seed": 7}),
        ("anthropic", {"logprobs": True}),
        ("anthropic", {"logit_bias": {"5": 10}}),
        ("anthropic", {"response_format": {"type": "json_object"}}),
        ("anthropic", {"max_completion_tokens": None}),
        ("bedrock", {"frequency_penalty": 0.5}),
        ("bedrock", {"seed": 7}),
        ("gemini", {"logit_bias": {"5": 10}}),
        ("gemini", {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}),
        (
            "anthropic",
            {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
        ),
        ("anthropic", {"messages": [{"role": "assistant", "content": "a"}]}),
        ("gemini", {"messages": [{"role": "user", "content": "a", "name": "alice"}]}),
        ("gemini", {"n": 2}),
        ("bedrock", {"stream": True}),
        ("bedrock", {"model": ".."}),
    ],
)
def test_untranslatable_requests_are_refused_with_a_fix(
    provider: str, changes: dict[str, Any]
) -> None:
    with pytest.raises(ProviderRequestError) as caught:
        _ADAPTERS[provider].build_request(_canonical(**changes))
    assert "fix:" in str(caught.value)
    assert "provider-secret" not in str(caught.value)


def test_gemini_translates_supported_sampling_and_json_schema() -> None:
    request = _ADAPTERS["gemini"].build_request(
        _canonical(
            seed=11,
            frequency_penalty=0.25,
            presence_penalty=-0.25,
            top_p=0.9,
            stop="STOP",
            logprobs=True,
            top_logprobs=3,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}},
            },
        )
    )

    assert request.payload["generationConfig"] == {
        "candidateCount": 1,
        "maxOutputTokens": 64,
        "temperature": 0.7,
        "topP": 0.9,
        "seed": 11,
        "frequencyPenalty": 0.25,
        "presencePenalty": -0.25,
        "stopSequences": ["STOP"],
        "responseLogprobs": True,
        "logprobs": 3,
        "responseMimeType": "application/json",
        "responseSchema": {"type": "object"},
    }


def test_operator_extras_reach_the_wire_but_never_shadow_translated_fields() -> None:
    request = _ADAPTERS["anthropic"].build_request(
        _canonical(thinking={"type": "enabled", "budget_tokens": 1024})
    )
    assert request.payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}

    with pytest.raises(ProviderRequestError) as caught:
        _ADAPTERS["anthropic"].build_request(_canonical(system="operator override"))
    assert "fix: remove system from upstreams.<name>.fixed_extra_body" in str(caught.value)


def _upstream_config(provider: str) -> UpstreamConfig:
    return UpstreamConfig(
        name="remote",
        base_url="https://provider.internal/v1",
        api_key="provider-secret",
        max_connections=1,
        max_response_bytes=65_536,
        provider=provider,
    )


def _sample_body() -> dict[str, Any]:
    return {
        "upstream_model": "private/model",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "question"},
        ],
        "max_completion_tokens": 64,
        "temperature": 0.7,
    }


def test_adapter_refusal_never_reaches_the_client_message(caplog) -> None:
    """A translation refusal names the provider and operator config; the client must not see it."""

    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_RESPONSES["anthropic"])

        upstream = OpenAIUpstream(
            _upstream_config("anthropic"), 0.05, transport=httpx.MockTransport(handler)
        )
        body = _sample_body()
        body["seed"] = 7
        with pytest.raises(GatewayError) as caught:
            await upstream.sample(
                body,
                extractor=lambda text: "42",
                public_request_id="public-id",
                sample_index=1,
            )
        await upstream.close()

        error = caught.value
        assert error.status_code == 400
        assert error.code == "upstream_unsupported_request"
        assert error.message == "The request is not supported by the requested model."
        for leaked in ("anthropic", "fix:", "seed", "options", "fixed_extra_body", "remote"):
            assert leaked not in error.message
        assert calls == 0, "a refused translation must never reach the upstream"

    with caplog.at_level("WARNING", logger="branchpilot.gateway.upstream"):
        asyncio.run(scenario())

    detail = " ".join(record.getMessage() + str(record.__dict__) for record in caplog.records)
    assert "anthropic" in detail, "the actionable detail must survive server-side"
    assert "fix:" in detail
    assert "provider-secret" not in detail, "logs must never carry credentials"


@pytest.mark.parametrize("provider", sorted(_ADAPTERS))
def test_gateway_sends_the_provider_wire_format_and_normalizes_usage(provider: str) -> None:
    async def scenario() -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_RESPONSES[provider])

        upstream = OpenAIUpstream(
            _upstream_config(provider), 0.05, transport=httpx.MockTransport(handler)
        )
        result = await upstream.sample(
            _sample_body(),
            extractor=lambda text: "42",
            public_request_id="public-id",
            sample_index=1,
        )
        prompt, cached, completion, total, prompt_details = _EXPECTED_USAGE[provider]
        name, credential = _CREDENTIAL_HEADERS[provider]

        assert seen[0].url.raw_path.decode() == f"/v1{_PATHS[provider]}"
        assert seen[0].headers[name] == credential
        assert seen[0].headers["accept-encoding"] == "identity"
        assert json.loads(seen[0].content) == _PAYLOADS[provider]
        assert result.sample.answer == "42"
        assert result.sample.text == "reasoning\n\\boxed{42}"
        assert result.sample.token_count == completion
        assert result.usage.prompt_tokens == prompt
        assert result.usage.completion_tokens == completion
        assert result.usage.total_tokens == total
        assert result.usage.prompt_tokens_details == prompt_details
        assert result.usage.prompt_tokens_details["cached_tokens"] == cached
        await upstream.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_translated_logprobs_reach_the_selection_signal(provider: str) -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_RESPONSES[provider])

        upstream = OpenAIUpstream(
            _upstream_config(provider), 0.05, transport=httpx.MockTransport(handler)
        )
        result = await upstream.sample(
            _sample_body(),
            extractor=lambda text: "42",
            public_request_id="public-id",
            sample_index=1,
        )
        assert result.sample.mean_logprob == -1.0
        await upstream.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", sorted(_ADAPTERS))
def test_missing_completion_usage_becomes_a_sanitized_502_and_frees_capacity(
    provider: str,
) -> None:
    broken = {
        "openai": _without(_OPENAI_RESPONSE, "usage", "completion_tokens"),
        "anthropic": _without(_ANTHROPIC_RESPONSE, "usage", "output_tokens"),
        "gemini": _without(_GEMINI_RESPONSE, "usageMetadata", "candidatesTokenCount"),
        "bedrock": _without(_BEDROCK_RESPONSE, "usage", "outputTokens"),
    }[provider]

    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=broken if calls == 1 else _RESPONSES[provider])

        upstream = OpenAIUpstream(
            _upstream_config(provider), 0.05, transport=httpx.MockTransport(handler)
        )
        with pytest.raises(GatewayError) as caught:
            await upstream.sample(
                _sample_body(),
                extractor=lambda text: "42",
                public_request_id="public-id",
                sample_index=1,
            )
        assert caught.value.status_code == 502
        assert caught.value.code == "invalid_upstream_response"
        assert "private" not in caught.value.message
        assert "provider-secret" not in caught.value.message

        recovered = await upstream.sample(
            _sample_body(),
            extractor=lambda text: "42",
            public_request_id="public-id",
            sample_index=2,
        )
        assert recovered.sample.answer == "42"
        assert calls == 2
        await upstream.close()

    asyncio.run(scenario())


def test_untranslatable_request_fails_before_the_upstream_is_contacted() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_ANTHROPIC_RESPONSE)

        upstream = OpenAIUpstream(
            _upstream_config("anthropic"), 0.05, transport=httpx.MockTransport(handler)
        )
        body = _sample_body()
        body["seed"] = 7
        with pytest.raises(GatewayError) as caught:
            await upstream.sample(
                body,
                extractor=lambda text: "42",
                public_request_id="public-id",
                sample_index=1,
            )
        assert caught.value.status_code == 400
        assert caught.value.code == "upstream_unsupported_request"
        assert caught.value.message == "The request is not supported by the requested model."
        assert "provider-secret" not in caught.value.message
        assert calls == 0

        allowed = await upstream.sample(
            _sample_body(),
            extractor=lambda text: "42",
            public_request_id="public-id",
            sample_index=2,
        )
        assert allowed.usage.completion_tokens == 9
        await upstream.close()

    asyncio.run(scenario())


def test_duplicate_json_keys_and_non_finite_numbers_are_refused() -> None:
    async def scenario() -> None:
        bodies = [
            b'{"choices": [], "choices": []}',
            json.dumps(_OPENAI_RESPONSE).replace("-0.5", "NaN").encode(),
        ]

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=bodies.pop(0))

        upstream = OpenAIUpstream(
            _upstream_config("openai"), 0.05, transport=httpx.MockTransport(handler)
        )
        for _ in range(2):
            with pytest.raises(GatewayError) as caught:
                await upstream.sample(
                    _sample_body(),
                    extractor=lambda text: "42",
                    public_request_id="public-id",
                    sample_index=1,
                )
            assert caught.value.code == "invalid_upstream_response"
        await upstream.close()

    asyncio.run(scenario())


def test_injected_sdk_client_cannot_serve_a_non_openai_provider() -> None:
    with pytest.raises(ValueError) as caught:
        OpenAIUpstream(_upstream_config("anthropic"), 0.05, client=SimpleNamespace())
    assert "fix:" in str(caught.value)


def test_registry_is_fixed_and_refuses_unknown_ids() -> None:
    assert PROVIDER_IDS == ("anthropic", "bedrock", "gemini", "openai")
    assert resolve_adapter("openai") is resolve_adapter("openai")
    with pytest.raises(UnknownProviderError) as caught:
        resolve_adapter("branchpilot.gateway.providers.evil:Adapter")
    assert "fix: set upstreams.<name>.provider to one of: " in str(caught.value)
    for name in PROVIDER_IDS:
        assert name in str(caught.value)


def test_upstream_config_refuses_an_unregistered_provider() -> None:
    assert _upstream_config("openai").provider == "openai"
    with pytest.raises(ConfigError) as caught:
        _upstream_config("vertex")
    assert "fix: set upstreams.<name>.provider to one of: " in str(caught.value)


def _gateway_payload(provider: str | None) -> dict[str, Any]:
    upstream: dict[str, Any] = {
        "base_url": "https://provider.internal/v1",
        "api_key_env": "UPSTREAM_KEY",
        "max_connections": 2,
    }
    if provider is not None:
        upstream["provider"] = provider
    return {
        "inbound_api_key_envs": ["INBOUND_KEY"],
        "upstreams": {"remote": upstream},
        "models": {
            "math": {
                "upstream": "remote",
                "upstream_model": "private/model",
                "plan_path": "plans/deployment.json",
                "extractor": "numeric-strict",
                "max_completion_tokens": 128,
            }
        },
    }


@pytest.fixture
def stub_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(path: Path):
        deployment = SimpleNamespace(
            strategy=FixedStrategy(samples=2, max_samples=3),
            cost=0.0,
            spec={"type": "fixed", "samples": 2, "max_samples": 3},
        )
        return deployment, SimpleNamespace(policy="fixed-2", family="fixed")

    monkeypatch.setattr(gateway_config, "load_deployment_plan", load)


def _load(tmp_path: Path, provider: str | None):
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(_gateway_payload(provider)), encoding="utf-8")
    return load_gateway_config(
        path, environ={"INBOUND_KEY": "client-secret", "UPSTREAM_KEY": "provider-secret"}
    )


@pytest.mark.usefixtures("stub_plan")
def test_config_defaults_to_openai_and_binds_registered_providers(tmp_path: Path) -> None:
    assert _load(tmp_path, None).upstreams["remote"].provider == "openai"
    assert _load(tmp_path, "bedrock").upstreams["remote"].provider == "bedrock"


@pytest.mark.usefixtures("stub_plan")
def test_config_load_refuses_an_unknown_provider_with_a_fix(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _load(tmp_path, "vertex")
    message = str(caught.value)
    assert "upstreams.remote.provider" in message
    assert "fix: set upstreams.<name>.provider to one of: " in message
    for name in PROVIDER_IDS:
        assert name in message


@pytest.mark.usefixtures("stub_plan")
def test_config_refuses_credential_shadowing_through_fixed_extra_body(tmp_path: Path) -> None:
    payload = _gateway_payload("openai")
    payload["upstreams"]["remote"]["fixed_extra_body"] = {"api_key": "attacker"}
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_gateway_config(
            path, environ={"INBOUND_KEY": "client-secret", "UPSTREAM_KEY": "provider-secret"}
        )
    assert "api_key" in str(caught.value)
