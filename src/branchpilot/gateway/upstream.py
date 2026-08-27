from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion

from branchpilot.gateway.config import UpstreamConfig
from branchpilot.schema import Sample

_MISSING = object()
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_PROMPT_DETAIL_FIELDS = ("audio_tokens", "cached_tokens", "cache_write_tokens")
_COMPLETION_DETAIL_FIELDS = (
    "accepted_prediction_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
)


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        error_type: str,
        code: str,
        message: str,
        *,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.message = message
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Mapping[str, int] | None = None
    completion_tokens_details: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class UpstreamChoice:
    content: str
    finish_reason: str
    refusal: str | None
    logprobs: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class UpstreamSample:
    sample: Sample
    choice: UpstreamChoice
    usage: Usage
    upstream_request_id: str | None


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _invalid_response()
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _dump(model_dump(mode="json", exclude_none=False))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _dump({key: item for key, item in attributes.items() if not key.startswith("_")})
    if isinstance(value, float) and not math.isfinite(value):
        raise _invalid_response()
    if isinstance(value, (str, int, float, bool)):
        return value
    raise _invalid_response()


def _invalid_response() -> GatewayError:
    return GatewayError(
        502,
        "api_error",
        "invalid_upstream_response",
        "The upstream returned an invalid response.",
    )


def _nonnegative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _invalid_response()
    return value


def _usage(response: Any) -> Usage:
    value = _field(response, "usage")
    if value is _MISSING or value is None:
        raise _invalid_response()
    prompt = _nonnegative_int(_field(value, "prompt_tokens"))
    completion = _nonnegative_int(_field(value, "completion_tokens"))
    total = _nonnegative_int(_field(value, "total_tokens"))
    if total != prompt + completion:
        raise _invalid_response()
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=_usage_details(
            _field(value, "prompt_tokens_details", None), _PROMPT_DETAIL_FIELDS
        ),
        completion_tokens_details=_usage_details(
            _field(value, "completion_tokens_details", None), _COMPLETION_DETAIL_FIELDS
        ),
    )


def _usage_details(value: Any, names: Sequence[str]) -> Mapping[str, int] | None:
    if value is None:
        return None
    details: dict[str, int] = {}
    for name in names:
        item = _field(value, name, _MISSING)
        if item is not _MISSING and item is not None:
            details[name] = _nonnegative_int(item)
    return details or None


def _mean_logprob(logprobs: Any) -> float | None:
    if logprobs is None:
        return None
    content = _field(logprobs, "content", None)
    if content is None:
        return None
    if isinstance(content, (str, bytes, Mapping)):
        raise _invalid_response()
    try:
        tokens = list(content)
    except TypeError as exc:
        raise _invalid_response() from exc
    if not tokens:
        return None
    values: list[float] = []
    for token in tokens:
        number = _field(token, "logprob")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise _invalid_response()
        value = float(number)
        if not math.isfinite(value):
            raise _invalid_response()
        values.append(value)
    mean = math.fsum(values) / len(values)
    if not math.isfinite(mean):
        raise _invalid_response()
    return mean


def parse_chat_completion(
    response: Any,
    extractor: Callable[[str], str | None],
) -> UpstreamSample:
    choices = _field(response, "choices")
    if choices is _MISSING or choices is None or isinstance(choices, (str, bytes, Mapping)):
        raise _invalid_response()
    try:
        choice_values = list(choices)
    except TypeError as exc:
        raise _invalid_response() from exc
    if len(choice_values) != 1:
        raise _invalid_response()
    choice = choice_values[0]
    message = _field(choice, "message")
    if message is _MISSING or message is None:
        raise _invalid_response()
    content = _field(message, "content")
    if not isinstance(content, str):
        raise _invalid_response()
    role = _field(message, "role", "assistant")
    if role != "assistant":
        raise _invalid_response()
    refusal = _field(message, "refusal", None)
    if refusal is not None and not isinstance(refusal, str):
        raise _invalid_response()
    finish_reason = _field(choice, "finish_reason")
    if finish_reason not in _FINISH_REASONS:
        raise _invalid_response()
    logprobs_value = _field(choice, "logprobs", None)
    mean_logprob = _mean_logprob(logprobs_value)
    logprobs = _dump(logprobs_value)
    if logprobs is not None and not isinstance(logprobs, Mapping):
        raise _invalid_response()

    usage = _usage(response)
    if finish_reason == "length":
        answer = None
        parse_status = "truncated"
    elif finish_reason != "stop":
        answer = None
        parse_status = "incomplete"
    else:
        try:
            answer = extractor(content)
        except Exception as exc:
            raise GatewayError(
                500, "api_error", "internal_error", "The gateway encountered an internal error."
            ) from exc
        if answer is not None and (not isinstance(answer, str) or not answer):
            raise GatewayError(
                500, "api_error", "internal_error", "The gateway encountered an internal error."
            )
        parse_status = "parsed" if answer is not None else "unparsed"
    sample = Sample(
        text=content,
        answer=answer,
        token_count=usage.completion_tokens,
        mean_logprob=mean_logprob,
        finish_reason=finish_reason,
        parse_status=parse_status,
    )
    request_id = _field(response, "_request_id", None)
    if request_id is not None and not isinstance(request_id, str):
        request_id = None
    return UpstreamSample(
        sample=sample,
        choice=UpstreamChoice(
            content=content,
            finish_reason=finish_reason,
            refusal=refusal,
            logprobs=logprobs,
        ),
        usage=usage,
        upstream_request_id=request_id,
    )


def _retry_after(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    if value.isdecimal():
        return value
    try:
        parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value


def _map_openai_error(exc: Exception) -> GatewayError:
    if isinstance(exc, httpx.TimeoutException):
        return GatewayError(504, "api_error", "upstream_timeout", "The upstream request timed out.")
    if isinstance(exc, openai.APITimeoutError):
        return GatewayError(504, "api_error", "upstream_timeout", "The upstream request timed out.")
    if isinstance(exc, openai.APIResponseValidationError):
        return _invalid_response()
    if isinstance(exc, openai.APIConnectionError):
        return GatewayError(502, "api_error", "upstream_error", "The upstream is unavailable.")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status in {400, 422}:
            return GatewayError(
                400,
                "invalid_request_error",
                "upstream_rejected_request",
                "The upstream rejected the request.",
            )
        if status == 429:
            return GatewayError(
                429,
                "rate_limit_error",
                "gateway_overloaded",
                "The upstream is temporarily overloaded.",
                retry_after=_retry_after(exc.response),
            )
        if status in {401, 403}:
            return GatewayError(
                502,
                "api_error",
                "upstream_auth_error",
                "The upstream is unavailable.",
            )
        return GatewayError(502, "api_error", "upstream_error", "The upstream is unavailable.")
    return GatewayError(502, "api_error", "upstream_error", "The upstream is unavailable.")


def _map_http_response(response: httpx.Response) -> GatewayError:
    status = response.status_code
    if status in {400, 422}:
        return GatewayError(
            400,
            "invalid_request_error",
            "upstream_rejected_request",
            "The upstream rejected the request.",
        )
    if status == 429:
        return GatewayError(
            429,
            "rate_limit_error",
            "gateway_overloaded",
            "The upstream is temporarily overloaded.",
            retry_after=_retry_after(response),
        )
    if status in {401, 403}:
        return GatewayError(
            502,
            "api_error",
            "upstream_auth_error",
            "The upstream is unavailable.",
        )
    return GatewayError(502, "api_error", "upstream_error", "The upstream is unavailable.")


def _declared_length_exceeds(value: str, maximum: int) -> bool:
    normalized = value.lstrip("0") or "0"
    limit = str(maximum)
    return len(normalized) > len(limit) or (len(normalized) == len(limit) and normalized > limit)


class OpenAIUpstream:
    """One long-lived, non-retrying upstream client with bounded admission and responses."""

    def __init__(
        self,
        config: UpstreamConfig,
        queue_timeout_s: float,
        *,
        client: Any | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("client and transport cannot both be provided")
        self.config = config
        self.queue_timeout_s = queue_timeout_s
        self._semaphore = asyncio.Semaphore(config.max_connections)
        self._closed = False
        self._client = client
        if client is None:
            timeout = httpx.Timeout(
                connect=config.connect_timeout_s,
                read=config.read_timeout_s,
                write=config.write_timeout_s,
                pool=config.pool_timeout_s,
            )
            self._http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=config.max_connections,
                    max_keepalive_connections=config.max_connections,
                ),
                follow_redirects=False,
                trust_env=False,
                transport=transport,
            )
        else:
            self._http_client = None

    async def _bounded_response(
        self,
        payload: Mapping[str, Any],
        *,
        public_request_id: str,
        sample_index: int,
    ) -> ChatCompletion:
        assert self._http_client is not None
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
            "X-Request-ID": f"{public_request_id}:{sample_index}",
        }
        url = f"{self.config.base_url}/chat/completions"
        try:
            async with self._http_client.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise _map_http_response(response)
                content_encodings = response.headers.get_list("content-encoding")
                if content_encodings and (
                    len(content_encodings) != 1
                    or content_encodings[0].strip().lower() != "identity"
                ):
                    raise _invalid_response()
                lengths = response.headers.get_list("content-length")
                if lengths:
                    if len(lengths) != 1 or not lengths[0].isdigit():
                        raise _invalid_response()
                    if _declared_length_exceeds(lengths[0], self.config.max_response_bytes):
                        raise _invalid_response()
                encoded = bytearray()
                if response.is_stream_consumed:
                    if len(response.content) > self.config.max_response_bytes:
                        raise _invalid_response()
                    encoded.extend(response.content)
                else:
                    async for chunk in response.aiter_raw():
                        if len(encoded) + len(chunk) > self.config.max_response_bytes:
                            raise _invalid_response()
                        encoded.extend(chunk)
        except GatewayError:
            raise
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            raise _map_openai_error(exc) from exc

        try:
            return ChatCompletion.model_validate_json(encoded)
        except (TypeError, ValueError) as exc:
            raise _invalid_response() from exc

    async def sample(
        self,
        body: Mapping[str, Any],
        *,
        extractor: Callable[[str], str | None],
        public_request_id: str,
        sample_index: int,
    ) -> UpstreamSample:
        if self._closed:
            raise GatewayError(500, "api_error", "internal_error", "The gateway is unavailable.")
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            raise GatewayError(
                429,
                "rate_limit_error",
                "gateway_overloaded",
                "The gateway is temporarily overloaded.",
            ) from exc
        try:
            payload = dict(body)
            payload["model"] = payload.pop("upstream_model")
            payload["n"] = 1
            payload["stream"] = False

            if self._client is None:
                direct_payload = dict(payload)
                direct_payload.update(self.config.fixed_extra_body)
                response = await self._bounded_response(
                    direct_payload,
                    public_request_id=public_request_id,
                    sample_index=sample_index,
                )
            else:
                kwargs = dict(payload)
                kwargs["extra_headers"] = {"X-Request-ID": f"{public_request_id}:{sample_index}"}
                if self.config.fixed_extra_body:
                    kwargs["extra_body"] = dict(self.config.fixed_extra_body)
                try:
                    response = await self._client.chat.completions.create(**kwargs)
                except (openai.APIError, httpx.HTTPError) as exc:
                    raise _map_openai_error(exc) from exc
            return parse_chat_completion(response, extractor)
        finally:
            self._semaphore.release()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._http_client is not None:
            await self._http_client.aclose()
        else:
            await self._client.close()
