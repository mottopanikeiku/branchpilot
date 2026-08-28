from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import quote

from branchpilot.gateway.schemas import ChatCompletionRequest

FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
PROMPT_DETAIL_FIELDS = ("audio_tokens", "cached_tokens", "cache_write_tokens")
COMPLETION_DETAIL_FIELDS = (
    "accepted_prediction_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
)

CANONICAL_API_KEY = "api_key"
CANONICAL_REQUEST_ID = "request_id"
CANONICAL_META_FIELDS = frozenset({CANONICAL_API_KEY, CANONICAL_REQUEST_ID})
CANONICAL_BODY_FIELDS = frozenset(ChatCompletionRequest.model_fields) - {"branchpilot"}
SYSTEM_ROLES = ("system", "developer")

_MAX_REQUEST_ID_CHARS = 128
_MAX_JSON_DEPTH = 64
_HEADER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")
_AUTH_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key"})
_REQUIRED_HEADERS = MappingProxyType(
    {
        "accept": "application/json",
        "accept-encoding": "identity",
        "content-type": "application/json",
    }
)


class _ProviderError(ValueError):
    """Base for provider translation failures; every message must name the fix."""

    def __init__(self, message: str) -> None:
        if "fix:" not in message:
            raise AssertionError("provider error messages must contain a 'fix:' clause")
        super().__init__(message)


class ProviderRequestError(_ProviderError):
    """A refusal to translate an outbound request; raised before any upstream call."""


class ProviderResponseError(_ProviderError):
    """An upstream response that cannot be normalized without inventing data."""


def checked_json(value: Any, *, depth: int = 0) -> Any:
    """Validate that a decoded JSON value is finite, string-keyed, and not absurdly nested."""
    if depth > _MAX_JSON_DEPTH:
        raise ProviderResponseError(
            f"upstream response nests JSON deeper than {_MAX_JSON_DEPTH} levels; "
            "fix: point the upstream at a provider endpoint that returns a normal "
            "completion document"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProviderResponseError(
                "upstream response contains a non-finite number; "
                "fix: point the upstream at a provider endpoint that returns finite JSON numbers"
            )
        return value
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProviderResponseError(
                    "upstream response contains a non-string JSON object key; "
                    "fix: point the upstream at a provider endpoint that returns JSON objects"
                )
            checked_json(item, depth=depth + 1)
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            checked_json(item, depth=depth + 1)
        return value
    raise ProviderResponseError(
        f"upstream response contains an unsupported JSON value of type {type(value).__name__}; "
        "fix: point the upstream at a provider endpoint that returns plain JSON"
    )


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """One fully translated outbound request: no credential ever enters ``repr``."""

    path: str
    headers: Mapping[str, str] = field(repr=False)
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        path = self.path
        if not isinstance(path, str) or not path.startswith("/") or len(path) > 2048:
            raise ProviderRequestError(
                "provider request path must be a relative path beginning with '/'; "
                "fix: return an adapter path such as '/chat/completions'"
            )
        segments = path.split("/")
        if any(segment in {"", ".", ".."} for segment in segments[1:]) or any(
            character in path for character in ("?", "#", "\\", "\r", "\n")
        ):
            raise ProviderRequestError(
                "provider request path must not contain empty, relative, or query segments; "
                "fix: percent-encode every dynamic path segment with path_segment()"
            )
        headers: dict[str, str] = {}
        lowered_headers: dict[str, str] = {}
        for name, value in dict(self.headers).items():
            if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
                raise ProviderRequestError(
                    "provider request header names must be ASCII tokens; "
                    "fix: use header names matching [A-Za-z][A-Za-z0-9-]*"
                )
            if not isinstance(value, str) or not value or not value.isprintable():
                raise ProviderRequestError(
                    f"provider request header {name!r} must be a non-empty printable value; "
                    "fix: build the header from a validated credential or request id"
                )
            lowered = name.lower()
            if lowered in lowered_headers:
                raise ProviderRequestError(
                    f"provider request repeats header {name!r}; "
                    "fix: emit each header exactly once from the adapter"
                )
            lowered_headers[lowered] = value
            headers[name] = value
        for required, expected in _REQUIRED_HEADERS.items():
            if lowered_headers.get(required) != expected:
                raise ProviderRequestError(
                    f"provider request must send {required}: {expected}; "
                    f"fix: add the {required} header to the adapter's build_request"
                )
        if not lowered_headers.keys() & _AUTH_HEADERS:
            raise ProviderRequestError(
                "provider request carries no credential header; "
                f"fix: send one of: {', '.join(sorted(_AUTH_HEADERS))}"
            )
        payload = dict(self.payload)
        if any(not isinstance(key, str) for key in payload):
            raise ProviderRequestError(
                "provider request payload keys must be strings; "
                "fix: build the payload from canonical body fields only"
            )
        if CANONICAL_API_KEY in payload or CANONICAL_REQUEST_ID in payload:
            raise ProviderRequestError(
                "provider request payload must not carry canonical meta fields; "
                f"fix: drop {CANONICAL_API_KEY} and {CANONICAL_REQUEST_ID} from the wire payload"
            )
        object.__setattr__(self, "headers", MappingProxyType(headers))
        object.__setattr__(self, "payload", MappingProxyType(payload))


@dataclass(frozen=True, slots=True)
class CanonicalResponse:
    """One normalized, single-choice completion with real, self-consistent token usage."""

    content: str
    finish_reason: str
    refusal: str | None
    logprobs: Mapping[str, Any] | None
    prompt_tokens: int
    cached_prompt_tokens: int | None
    completion_tokens: int
    total_tokens: int
    upstream_request_id: str | None
    prompt_tokens_details: Mapping[str, int] | None = None
    completion_tokens_details: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ProviderResponseError(
                "upstream completion content must be a single text string; "
                "fix: route multimodal or tool-calling models to a provider path that "
                "returns text content"
            )
        if self.finish_reason not in FINISH_REASONS:
            raise ProviderResponseError(
                f"upstream reported an unknown finish reason; "
                f"fix: map the provider stop reason to one of: {', '.join(sorted(FINISH_REASONS))}"
            )
        if self.refusal is not None and not isinstance(self.refusal, str):
            raise ProviderResponseError(
                "upstream refusal must be text when present; "
                "fix: pass None when the provider reports no refusal message"
            )
        if self.logprobs is not None:
            if not isinstance(self.logprobs, Mapping):
                raise ProviderResponseError(
                    "upstream logprobs must be an object; "
                    "fix: translate provider logprobs into {'content': [...]} or pass None"
                )
            checked_json(self.logprobs)
            _check_logprob_content(self.logprobs.get("content"))
        prompt = _checked_count(self.prompt_tokens, "prompt_tokens")
        completion = _checked_count(self.completion_tokens, "completion_tokens")
        total = _checked_count(self.total_tokens, "total_tokens")
        if total != prompt + completion:
            raise ProviderResponseError(
                "upstream usage is inconsistent: total_tokens must equal "
                "prompt_tokens + completion_tokens; "
                "fix: use a provider deployment that reports coherent token usage — "
                "BranchPilot never reconciles usage by estimating"
            )
        if completion == 0 and self.content != "":
            raise ProviderResponseError(
                "upstream returned completion text with zero completion tokens; "
                "fix: use a provider deployment that reports real output token usage — "
                "BranchPilot never synthesizes or estimates a token count"
            )
        if self.cached_prompt_tokens is not None:
            cached = _checked_count(self.cached_prompt_tokens, "cached_prompt_tokens")
            if cached > prompt:
                raise ProviderResponseError(
                    "upstream reported more cached prompt tokens than prompt tokens; "
                    "fix: use a provider deployment whose cached-token field is a subset "
                    "of the prompt tokens"
                )
        object.__setattr__(
            self,
            "prompt_tokens_details",
            _checked_details(self.prompt_tokens_details, self.cached_prompt_tokens),
        )
        object.__setattr__(
            self,
            "completion_tokens_details",
            _checked_details(self.completion_tokens_details, None),
        )
        if self.upstream_request_id is not None and not isinstance(self.upstream_request_id, str):
            raise ProviderResponseError(
                "upstream request id must be text when present; "
                "fix: pass None when the provider response carries no id"
            )


class ProviderAdapter(Protocol):
    """Translates the canonical OpenAI-shaped request and response for one provider."""

    def build_request(self, canonical: Mapping[str, Any]) -> ProviderRequest: ...

    def parse_response(self, payload: Mapping[str, Any]) -> CanonicalResponse: ...


def _checked_count(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(
            f"upstream usage field {name} must be a non-negative integer; "
            "fix: use a provider deployment that reports real integer token usage — "
            "BranchPilot never synthesizes or estimates a token count"
        )
    return value


def _checked_details(value: Any, cached: int | None) -> Mapping[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProviderResponseError(
            "usage detail breakdowns must be objects of integer counts; "
            "fix: pass None when the provider reports no usage breakdown"
        )
    if any(not isinstance(name, str) for name in value):
        raise ProviderResponseError(
            "usage detail names must be strings; "
            "fix: build the breakdown with OpenAI usage detail field names"
        )
    details = {name: _checked_count(item, name) for name, item in value.items()}
    if "cached_tokens" in details and details["cached_tokens"] != cached:
        raise ProviderResponseError(
            "cached prompt tokens disagree with the usage breakdown; "
            "fix: set cached_prompt_tokens from the same provider field as cached_tokens"
        )
    return MappingProxyType(details) if details else None


def _check_logprob_content(content: Any) -> None:
    if content is None:
        return
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise ProviderResponseError(
            "upstream logprobs content must be a list of token entries; "
            "fix: translate provider logprobs into {'content': [{'token', 'logprob'}]}"
        )
    for entry in content:
        number = entry.get("logprob") if isinstance(entry, Mapping) else None
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ProviderResponseError(
                "upstream logprobs entries must each carry a numeric logprob; "
                "fix: translate provider logprobs into {'content': [{'token', 'logprob'}]}"
            )
        if not math.isfinite(float(number)):
            raise ProviderResponseError(
                "upstream logprobs contain a non-finite value; "
                "fix: use a provider deployment that reports finite log probabilities"
            )


def body_fields(canonical: Mapping[str, Any]) -> dict[str, Any]:
    """Return the wire body fields of a canonical request, without the meta fields."""
    if not isinstance(canonical, Mapping):
        raise ProviderRequestError(
            "canonical request must be a mapping of OpenAI chat completion fields; "
            "fix: call build_request with the canonical request body"
        )
    return {name: value for name, value in canonical.items() if name not in CANONICAL_META_FIELDS}


def canonical_api_key(canonical: Mapping[str, Any]) -> str:
    value = canonical.get(CANONICAL_API_KEY)
    if not isinstance(value, str) or not value:
        raise ProviderRequestError(
            "canonical request carries no upstream credential; "
            "fix: build requests through OpenAIUpstream, which supplies the credential "
            "from upstreams.<name>.api_key_env"
        )
    return value


def canonical_request_id(canonical: Mapping[str, Any]) -> str:
    value = canonical.get(CANONICAL_REQUEST_ID)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_REQUEST_ID_CHARS
        or not value.isprintable()
    ):
        raise ProviderRequestError(
            "canonical request carries no usable correlation id; "
            "fix: build requests through OpenAIUpstream, which supplies request_id"
        )
    return value


def canonical_model(canonical: Mapping[str, Any]) -> str:
    value = canonical.get("model")
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ProviderRequestError(
            "canonical request has no upstream model; "
            "fix: set models.<alias>.upstream_model in the gateway config"
        )
    return value


def require_single_completion(canonical: Mapping[str, Any]) -> None:
    samples = canonical.get("n", 1)
    if isinstance(samples, bool) or samples != 1:
        raise ProviderRequestError(
            "provider adapters translate exactly one completion per request; "
            "fix: leave n at 1 and let the gateway strategy request further samples"
        )
    if canonical.get("stream", False) is not False:
        raise ProviderRequestError(
            "provider adapters do not translate streaming responses; "
            "fix: send stream=false to this route"
        )


def path_segment(value: str) -> str:
    """Percent-encode one dynamic path segment so a model name can never escape the path."""
    if value.strip() != value or value in {".", ".."}:
        raise ProviderRequestError(
            "upstream model name is not usable as a URL path segment; "
            "fix: set models.<alias>.upstream_model to the provider's model id"
        )
    return quote(value, safe="")


def split_messages(
    body: Mapping[str, Any],
    *,
    provider: str,
    require_alternating: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split canonical messages into system texts and strictly alternating chat turns."""
    messages = body.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ProviderRequestError(
            "canonical request has no messages; fix: send at least one user message"
        )
    system: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ProviderRequestError(
                "canonical messages must be objects with a role and text content; "
                "fix: send OpenAI chat messages with string content"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ProviderRequestError(
                "canonical messages must be objects with a role and text content; "
                "fix: send OpenAI chat messages with string content"
            )
        if message.get("name") is not None:
            raise ProviderRequestError(
                f"provider {provider!r} has no per-message name field; "
                "fix: remove 'name' from the request messages"
            )
        if role in SYSTEM_ROLES:
            if turns:
                raise ProviderRequestError(
                    f"provider {provider!r} accepts system instructions only before the "
                    "first chat turn; fix: move every system or developer message to the "
                    "start of the request"
                )
            system.append(content)
            continue
        if role not in {"user", "assistant"}:
            raise ProviderRequestError(
                f"provider {provider!r} accepts only user and assistant turns; "
                "fix: send chat turns with role 'user' or 'assistant'"
            )
        turns.append((role, content))
    if not turns or turns[0][0] != "user":
        raise ProviderRequestError(
            f"provider {provider!r} requires the conversation to begin with a user turn; "
            "fix: send a user message after any system messages"
        )
    if require_alternating:
        for previous, current in zip(turns, turns[1:], strict=False):
            if previous[0] == current[0]:
                raise ProviderRequestError(
                    f"provider {provider!r} requires user and assistant turns to alternate; "
                    "fix: merge consecutive same-role messages before sending"
                )
    return system, turns


def reject_unsupported(body: Mapping[str, Any], names: Iterable[str], provider: str) -> None:
    """Refuse rather than silently drop a sampling parameter the provider cannot honor."""
    present = sorted(name for name in names if body.get(name) is not None)
    if present:
        raise ProviderRequestError(
            f"provider {provider!r} does not support request field(s) "
            f"{', '.join(present)}; fix: remove {', '.join(present)} from the request and "
            f"from models.<alias>.options, or route this model to an 'openai' upstream"
        )


def max_output_tokens(body: Mapping[str, Any], *, provider: str, required: bool) -> int | None:
    value = body.get("max_completion_tokens")
    if value is None:
        value = body.get("max_tokens")
    if value is None:
        if required:
            raise ProviderRequestError(
                f"provider {provider!r} requires an output token limit; "
                "fix: set max_completion_tokens on the request or "
                "models.<alias>.options.max_completion_tokens in the gateway config"
            )
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProviderRequestError(
            "output token limit must be a positive integer; "
            "fix: set max_completion_tokens to a positive integer"
        )
    return value


def stop_sequences(body: Mapping[str, Any]) -> list[str] | None:
    value = body.get("stop")
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ProviderRequestError(
        "stop must be a string or a list of strings; "
        "fix: send stop as a string or a list of up to four strings"
    )


def operator_extras(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return operator-supplied provider-native fields from ``fixed_extra_body``."""
    return {name: value for name, value in body.items() if name not in CANONICAL_BODY_FIELDS}


def merge_extras(
    payload: dict[str, Any], extras: Mapping[str, Any], *, provider: str
) -> dict[str, Any]:
    collisions = sorted(set(extras) & set(payload))
    if collisions:
        raise ProviderRequestError(
            f"fixed_extra_body would overwrite translated field(s) {', '.join(collisions)} "
            f"for provider {provider!r}; fix: remove {', '.join(collisions)} from "
            "upstreams.<name>.fixed_extra_body"
        )
    payload.update(extras)
    return payload


def response_object(payload: Mapping[str, Any], name: str, *, provider: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ProviderResponseError(
            f"provider {provider!r} response is missing the {name!r} object; "
            f"fix: point this upstream at a {provider} endpoint that returns {name!r}"
        )
    return value


def required_count(usage: Mapping[str, Any], name: str, *, provider: str) -> int:
    value = usage.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(
            f"provider {provider!r} response does not report usage field {name!r}; "
            f"fix: use a {provider} deployment that returns real {name!r} — BranchPilot "
            "never synthesizes or estimates a token count"
        )
    return value


def optional_count(usage: Mapping[str, Any], name: str, *, provider: str) -> int | None:
    if usage.get(name) is None:
        return None
    return required_count(usage, name, provider=provider)


def text_content(
    blocks: Any,
    *,
    provider: str,
    type_key: str = "type",
    text_key: str = "text",
    text_type: str | None = "text",
) -> str:
    """Concatenate text blocks, refusing any block type this path cannot represent."""
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ProviderResponseError(
            f"provider {provider!r} response content must be a list of text blocks; "
            f"fix: point this upstream at a {provider} text completion endpoint"
        )
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ProviderResponseError(
                f"provider {provider!r} response content must be a list of text blocks; "
                f"fix: point this upstream at a {provider} text completion endpoint"
            )
        if text_type is not None and block.get(type_key) != text_type:
            raise ProviderResponseError(
                f"provider {provider!r} returned a non-text content block; "
                "fix: disable tool use, reasoning blocks, and multimodal output on this route"
            )
        text = block.get(text_key)
        if not isinstance(text, str):
            if text_type is None:
                raise ProviderResponseError(
                    f"provider {provider!r} returned a non-text content block; "
                    "fix: disable tool use, reasoning blocks, and multimodal output "
                    "on this route"
                )
            raise ProviderResponseError(
                f"provider {provider!r} returned a text block without text; "
                f"fix: use a {provider} deployment that returns string content"
            )
        parts.append(text)
    return "".join(parts)


def mapped_finish_reason(value: Any, table: Mapping[str, str], *, provider: str) -> str:
    reason = table.get(value) if isinstance(value, str) else None
    if reason is None:
        raise ProviderResponseError(
            f"provider {provider!r} reported an unsupported stop reason; "
            f"fix: retry without tools or guardrails — supported stop reasons are: "
            f"{', '.join(sorted(table))}"
        )
    return reason


def bounded_request_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > _MAX_REQUEST_ID_CHARS:
        return None
    return value


def usage_details(**counts: int | None) -> dict[str, int] | None:
    details = {name: value for name, value in counts.items() if value is not None}
    return details or None
