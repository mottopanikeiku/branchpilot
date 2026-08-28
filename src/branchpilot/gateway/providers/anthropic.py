from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from branchpilot.gateway.providers.base import (
    CanonicalResponse,
    ProviderRequest,
    ProviderResponseError,
    body_fields,
    bounded_request_id,
    canonical_api_key,
    canonical_model,
    canonical_request_id,
    mapped_finish_reason,
    max_output_tokens,
    merge_extras,
    operator_extras,
    optional_count,
    reject_unsupported,
    require_single_completion,
    required_count,
    response_object,
    split_messages,
    stop_sequences,
    text_content,
    usage_details,
)

_PROVIDER = "anthropic"
_PATH = "/messages"
_API_VERSION = "2023-06-01"
_UNSUPPORTED = (
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "presence_penalty",
    "response_format",
    "seed",
    "top_logprobs",
)
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class AnthropicAdapter:
    """Translates to and from the Anthropic Messages API.

    Anthropic reports ``input_tokens`` net of cache activity, so the canonical prompt total is
    ``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``; that keeps
    ``prompt_tokens`` comparable across providers and priced against one prompt-token rate.
    """

    __slots__ = ()

    def build_request(self, canonical: Mapping[str, Any]) -> ProviderRequest:
        body = body_fields(canonical)
        require_single_completion(body)
        reject_unsupported(body, _UNSUPPORTED, _PROVIDER)
        system, turns = split_messages(body, provider=_PROVIDER, require_alternating=True)
        payload: dict[str, Any] = {
            "model": canonical_model(body),
            "max_tokens": max_output_tokens(body, provider=_PROVIDER, required=True),
            "messages": [
                {"role": role, "content": [{"type": "text", "text": content}]}
                for role, content in turns
            ],
        }
        if system:
            payload["system"] = [{"type": "text", "text": text} for text in system]
        if body.get("temperature") is not None:
            payload["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            payload["top_p"] = body["top_p"]
        sequences = stop_sequences(body)
        if sequences is not None:
            payload["stop_sequences"] = sequences
        if body.get("user") is not None:
            payload["metadata"] = {"user_id": body["user"]}
        merge_extras(payload, operator_extras(body), provider=_PROVIDER)
        return ProviderRequest(
            path=_PATH,
            headers={
                "x-api-key": canonical_api_key(canonical),
                "anthropic-version": _API_VERSION,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
                "X-Request-ID": canonical_request_id(canonical),
            },
            payload=payload,
        )

    def parse_response(self, payload: Mapping[str, Any]) -> CanonicalResponse:
        if payload.get("type", "message") != "message":
            raise ProviderResponseError(
                "provider 'anthropic' response is not a message document; "
                "fix: point this upstream at the Anthropic /v1/messages endpoint"
            )
        if payload.get("role", "assistant") != "assistant":
            raise ProviderResponseError(
                "provider 'anthropic' response is not an assistant message; "
                "fix: point this upstream at the Anthropic /v1/messages endpoint"
            )
        usage = response_object(payload, "usage", provider=_PROVIDER)
        cache_read = optional_count(usage, "cache_read_input_tokens", provider=_PROVIDER)
        cache_write = optional_count(usage, "cache_creation_input_tokens", provider=_PROVIDER)
        prompt = (
            required_count(usage, "input_tokens", provider=_PROVIDER)
            + (cache_read or 0)
            + (cache_write or 0)
        )
        completion = required_count(usage, "output_tokens", provider=_PROVIDER)
        return CanonicalResponse(
            content=text_content(payload.get("content"), provider=_PROVIDER),
            finish_reason=mapped_finish_reason(
                payload.get("stop_reason"), _STOP_REASONS, provider=_PROVIDER
            ),
            refusal=None,
            logprobs=None,
            prompt_tokens=prompt,
            cached_prompt_tokens=cache_read,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            upstream_request_id=bounded_request_id(payload.get("id")),
            prompt_tokens_details=usage_details(
                cached_tokens=cache_read, cache_write_tokens=cache_write
            ),
            completion_tokens_details=None,
        )
