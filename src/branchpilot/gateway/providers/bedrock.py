from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from branchpilot.gateway.providers.base import (
    CanonicalResponse,
    ProviderRequest,
    ProviderResponseError,
    body_fields,
    canonical_api_key,
    canonical_model,
    canonical_request_id,
    mapped_finish_reason,
    max_output_tokens,
    merge_extras,
    operator_extras,
    optional_count,
    path_segment,
    reject_unsupported,
    require_single_completion,
    required_count,
    response_object,
    split_messages,
    stop_sequences,
    text_content,
    usage_details,
)

_PROVIDER = "bedrock"
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
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}


class BedrockAdapter:
    """Translates to and from the Amazon Bedrock Converse API.

    Authentication uses a Bedrock API key as a bearer token, so no request signing happens in
    this path. Converse reports ``totalTokens`` under two documented conventions — cache tokens
    inside ``inputTokens``, or alongside it — so the canonical prompt total is derived from
    whichever convention the reported total agrees with, and a response that matches neither is
    refused rather than guessed at. The request field ``user`` has no Converse equivalent and is
    dropped: it is abuse-tracking metadata that cannot affect the completion or its cost.
    """

    __slots__ = ()

    def build_request(self, canonical: Mapping[str, Any]) -> ProviderRequest:
        body = body_fields(canonical)
        require_single_completion(body)
        reject_unsupported(body, _UNSUPPORTED, _PROVIDER)
        system, turns = split_messages(body, provider=_PROVIDER, require_alternating=True)
        inference: dict[str, Any] = {}
        limit = max_output_tokens(body, provider=_PROVIDER, required=False)
        if limit is not None:
            inference["maxTokens"] = limit
        if body.get("temperature") is not None:
            inference["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            inference["topP"] = body["top_p"]
        sequences = stop_sequences(body)
        if sequences is not None:
            inference["stopSequences"] = sequences
        payload: dict[str, Any] = {
            "messages": [{"role": role, "content": [{"text": content}]} for role, content in turns]
        }
        if system:
            payload["system"] = [{"text": text} for text in system]
        if inference:
            payload["inferenceConfig"] = inference
        merge_extras(payload, operator_extras(body), provider=_PROVIDER)
        return ProviderRequest(
            path=f"/model/{path_segment(canonical_model(body))}/converse",
            headers={
                "Authorization": f"Bearer {canonical_api_key(canonical)}",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
                "X-Request-ID": canonical_request_id(canonical),
            },
            payload=payload,
        )

    def parse_response(self, payload: Mapping[str, Any]) -> CanonicalResponse:
        output = response_object(payload, "output", provider=_PROVIDER)
        message = response_object(output, "message", provider=_PROVIDER)
        if message.get("role", "assistant") != "assistant":
            raise ProviderResponseError(
                "provider 'bedrock' returned a non-assistant message; "
                "fix: point this upstream at the Bedrock Converse endpoint"
            )
        usage = response_object(payload, "usage", provider=_PROVIDER)
        read = optional_count(usage, "cacheReadInputTokens", provider=_PROVIDER)
        write = optional_count(usage, "cacheWriteInputTokens", provider=_PROVIDER)
        declared = required_count(usage, "inputTokens", provider=_PROVIDER)
        completion = required_count(usage, "outputTokens", provider=_PROVIDER)
        total = required_count(usage, "totalTokens", provider=_PROVIDER)
        cached = (read or 0) + (write or 0)
        if total == declared + completion:
            prompt = declared
        elif total == declared + cached + completion:
            prompt = declared + cached
        else:
            raise ProviderResponseError(
                "provider 'bedrock' totalTokens matches neither inputTokens + outputTokens nor "
                "inputTokens + cache tokens + outputTokens; fix: use a Bedrock model whose "
                "Converse usage block is self-consistent — BranchPilot never reconciles usage "
                "by estimating"
            )
        return CanonicalResponse(
            content=text_content(message.get("content"), provider=_PROVIDER, text_type=None),
            finish_reason=mapped_finish_reason(
                payload.get("stopReason"), _STOP_REASONS, provider=_PROVIDER
            ),
            refusal=None,
            logprobs=None,
            prompt_tokens=prompt,
            cached_prompt_tokens=read,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            upstream_request_id=None,
            prompt_tokens_details=usage_details(cached_tokens=read, cache_write_tokens=write),
            completion_tokens_details=None,
        )
