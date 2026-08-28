from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from branchpilot.gateway.providers.base import (
    COMPLETION_DETAIL_FIELDS,
    PROMPT_DETAIL_FIELDS,
    CanonicalResponse,
    ProviderRequest,
    ProviderResponseError,
    body_fields,
    bounded_request_id,
    canonical_api_key,
    canonical_request_id,
    checked_json,
    mapped_finish_reason,
    optional_count,
    required_count,
    response_object,
    usage_details,
)

_PROVIDER = "openai"
_PATH = "/chat/completions"
_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "function_call",
}


class OpenAIAdapter:
    """The reference adapter: the canonical shape is already the OpenAI wire shape."""

    __slots__ = ()

    def build_request(self, canonical: Mapping[str, Any]) -> ProviderRequest:
        return ProviderRequest(
            path=_PATH,
            headers={
                "Authorization": f"Bearer {canonical_api_key(canonical)}",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
                "X-Request-ID": canonical_request_id(canonical),
            },
            payload=body_fields(canonical),
        )

    def parse_response(self, payload: Mapping[str, Any]) -> CanonicalResponse:
        choices = payload.get("choices")
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or len(choices) != 1
        ):
            raise ProviderResponseError(
                "provider 'openai' response must carry exactly one choice; "
                "fix: leave n at 1 so the gateway strategy controls sampling"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderResponseError(
                "provider 'openai' choice must be an object; "
                "fix: point this upstream at an OpenAI chat completions endpoint"
            )
        message = response_object(choice, "message", provider=_PROVIDER)
        if message.get("role", "assistant") != "assistant":
            raise ProviderResponseError(
                "provider 'openai' returned a non-assistant message; "
                "fix: point this upstream at an OpenAI chat completions endpoint"
            )
        logprobs = choice.get("logprobs")
        if logprobs is not None and not isinstance(logprobs, Mapping):
            raise ProviderResponseError(
                "provider 'openai' logprobs must be an object; "
                "fix: point this upstream at an OpenAI chat completions endpoint"
            )
        usage = response_object(payload, "usage", provider=_PROVIDER)
        prompt_details = _details(usage, "prompt_tokens_details", PROMPT_DETAIL_FIELDS)
        completion_details = _details(usage, "completion_tokens_details", COMPLETION_DETAIL_FIELDS)
        return CanonicalResponse(
            content=message.get("content"),
            finish_reason=mapped_finish_reason(
                choice.get("finish_reason"), _FINISH_REASONS, provider=_PROVIDER
            ),
            refusal=message.get("refusal"),
            logprobs=checked_json(logprobs),
            prompt_tokens=required_count(usage, "prompt_tokens", provider=_PROVIDER),
            cached_prompt_tokens=(
                None if prompt_details is None else prompt_details.get("cached_tokens")
            ),
            completion_tokens=required_count(usage, "completion_tokens", provider=_PROVIDER),
            total_tokens=required_count(usage, "total_tokens", provider=_PROVIDER),
            upstream_request_id=bounded_request_id(payload.get("id")),
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
        )


def _details(usage: Mapping[str, Any], name: str, names: Sequence[str]) -> dict[str, int] | None:
    value = usage.get(name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProviderResponseError(
            f"provider 'openai' usage field {name!r} must be an object of integer counts; "
            "fix: point this upstream at an OpenAI chat completions endpoint"
        )
    return usage_details(
        **{field: optional_count(value, field, provider=_PROVIDER) for field in names}
    )
