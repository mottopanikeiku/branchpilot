from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from branchpilot.gateway.providers.base import (
    CanonicalResponse,
    ProviderRequest,
    ProviderRequestError,
    ProviderResponseError,
    body_fields,
    bounded_request_id,
    canonical_api_key,
    canonical_model,
    canonical_request_id,
    checked_json,
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
    usage_details,
)

_PROVIDER = "gemini"
_UNSUPPORTED = ("logit_bias",)
_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
}


class GeminiAdapter:
    """Translates to and from the Gemini ``generateContent`` API.

    Gemini bills "thoughts" as output, so ``thoughtsTokenCount`` joins ``candidatesTokenCount``
    in the canonical completion total, and ``toolUsePromptTokenCount`` joins the prompt total.
    ``cachedContentTokenCount`` is already inside ``promptTokenCount``. The request field ``user``
    has no Gemini equivalent and is dropped: it is abuse-tracking metadata that cannot affect the
    completion or its cost.
    """

    __slots__ = ()

    def build_request(self, canonical: Mapping[str, Any]) -> ProviderRequest:
        body = body_fields(canonical)
        require_single_completion(body)
        reject_unsupported(body, _UNSUPPORTED, _PROVIDER)
        system, turns = split_messages(body, provider=_PROVIDER, require_alternating=False)
        generation: dict[str, Any] = {"candidateCount": 1}
        limit = max_output_tokens(body, provider=_PROVIDER, required=False)
        if limit is not None:
            generation["maxOutputTokens"] = limit
        for canonical_name, wire_name in (
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("seed", "seed"),
            ("frequency_penalty", "frequencyPenalty"),
            ("presence_penalty", "presencePenalty"),
        ):
            if body.get(canonical_name) is not None:
                generation[wire_name] = body[canonical_name]
        sequences = stop_sequences(body)
        if sequences is not None:
            generation["stopSequences"] = sequences
        if body.get("logprobs") is True:
            generation["responseLogprobs"] = True
            if body.get("top_logprobs") is not None:
                generation["logprobs"] = body["top_logprobs"]
        generation.update(_response_format(body))
        payload: dict[str, Any] = {
            "contents": [
                {"role": "user" if role == "user" else "model", "parts": [{"text": content}]}
                for role, content in turns
            ],
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": text} for text in system]}
        merge_extras(payload, operator_extras(body), provider=_PROVIDER)
        return ProviderRequest(
            path=f"/models/{path_segment(canonical_model(body))}:generateContent",
            headers={
                "x-goog-api-key": canonical_api_key(canonical),
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
                "X-Request-ID": canonical_request_id(canonical),
            },
            payload=payload,
        )

    def parse_response(self, payload: Mapping[str, Any]) -> CanonicalResponse:
        candidates = payload.get("candidates")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or len(candidates) != 1
        ):
            raise ProviderResponseError(
                "provider 'gemini' response must carry exactly one candidate; "
                "fix: leave n at 1, and expect a refusal when Gemini blocks the prompt "
                "and returns no candidate"
            )
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise ProviderResponseError(
                "provider 'gemini' candidate must be an object; "
                "fix: point this upstream at a Gemini generateContent endpoint"
            )
        content = response_object(candidate, "content", provider=_PROVIDER)
        if content.get("role", "model") != "model":
            raise ProviderResponseError(
                "provider 'gemini' returned a non-model candidate; "
                "fix: point this upstream at a Gemini generateContent endpoint"
            )
        usage = response_object(payload, "usageMetadata", provider=_PROVIDER)
        cached = optional_count(usage, "cachedContentTokenCount", provider=_PROVIDER)
        thoughts = optional_count(usage, "thoughtsTokenCount", provider=_PROVIDER)
        tool_prompt = optional_count(usage, "toolUsePromptTokenCount", provider=_PROVIDER)
        prompt = required_count(usage, "promptTokenCount", provider=_PROVIDER) + (tool_prompt or 0)
        completion = required_count(usage, "candidatesTokenCount", provider=_PROVIDER) + (
            thoughts or 0
        )
        reported = optional_count(usage, "totalTokenCount", provider=_PROVIDER)
        if reported is not None and reported != prompt + completion:
            raise ProviderResponseError(
                "provider 'gemini' totalTokenCount disagrees with its own prompt, candidate, "
                "thought, and tool token counts; fix: use a Gemini deployment whose "
                "usageMetadata is self-consistent — BranchPilot never reconciles usage "
                "by estimating"
            )
        return CanonicalResponse(
            content=_parts_text(content.get("parts")),
            finish_reason=mapped_finish_reason(
                candidate.get("finishReason"), _FINISH_REASONS, provider=_PROVIDER
            ),
            refusal=None,
            logprobs=_logprobs(candidate.get("logprobsResult")),
            prompt_tokens=prompt,
            cached_prompt_tokens=cached,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            upstream_request_id=bounded_request_id(payload.get("responseId")),
            prompt_tokens_details=usage_details(cached_tokens=cached),
            completion_tokens_details=usage_details(reasoning_tokens=thoughts),
        )


def _response_format(body: Mapping[str, Any]) -> dict[str, Any]:
    value = body.get("response_format")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProviderRequestError(
            "response_format must be an object; "
            "fix: send response_format as {'type': 'text'|'json_object'|'json_schema'}"
        )
    kind = value.get("type")
    if kind == "text":
        return {}
    if kind == "json_object":
        return {"responseMimeType": "application/json"}
    schema = value.get("json_schema")
    if kind == "json_schema" and isinstance(schema, Mapping) and "schema" in schema:
        return {
            "responseMimeType": "application/json",
            "responseSchema": schema["schema"],
        }
    raise ProviderRequestError(
        "response_format is not translatable to Gemini; "
        "fix: send response_format as {'type': 'json_schema', 'json_schema': {'schema': {...}}}"
    )


def _parts_text(parts: Any) -> str:
    """Concatenate answer text parts, refusing thought, tool, and multimodal parts."""
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise ProviderResponseError(
            "provider 'gemini' candidate content must carry a list of parts; "
            "fix: point this upstream at a Gemini generateContent endpoint"
        )
    texts: list[str] = []
    for part in parts:
        if (
            not isinstance(part, Mapping)
            or set(part) != {"text"}
            or not isinstance(part.get("text"), str)
        ):
            raise ProviderResponseError(
                "provider 'gemini' returned a non-answer part (thought, tool call, or "
                "inline data); fix: disable thinking summaries, tools, and multimodal "
                "output on this route"
            )
        texts.append(part["text"])
    return "".join(texts)


def _logprobs(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProviderResponseError(
            "provider 'gemini' logprobsResult must be an object; "
            "fix: point this upstream at a Gemini generateContent endpoint"
        )
    chosen = value.get("chosenCandidates")
    if chosen is None:
        return None
    if not isinstance(chosen, Sequence) or isinstance(chosen, (str, bytes)):
        raise ProviderResponseError(
            "provider 'gemini' chosenCandidates must be a list; "
            "fix: point this upstream at a Gemini generateContent endpoint"
        )
    entries: list[dict[str, Any]] = []
    for item in chosen:
        if not isinstance(item, Mapping):
            raise ProviderResponseError(
                "provider 'gemini' chosenCandidates entries must be objects; "
                "fix: point this upstream at a Gemini generateContent endpoint"
            )
        entries.append(
            {"token": item.get("token"), "logprob": checked_json(item.get("logProbability"))}
        )
    return {"content": entries}
