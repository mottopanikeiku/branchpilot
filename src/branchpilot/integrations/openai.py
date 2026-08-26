from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from branchpilot.runtime import PilotResult
from branchpilot.schema import Sample

_RESERVED_REQUEST_OPTIONS = frozenset({"model", "messages", "n", "stream"})
_MISSING = object()


def _freeze(value: Any) -> Any:
    """Copy common request containers into an immutable snapshot."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Make a fresh set of ordinary request containers for each API call."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}
    return value


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _completion_tokens(response: Any) -> int:
    usage = _field(response, "usage")
    if usage is _MISSING or usage is None:
        raise ValueError("OpenAI response usage is missing; completion_tokens is required")
    token_count = _field(usage, "completion_tokens")
    if token_count is _MISSING or token_count is None:
        raise ValueError("OpenAI response usage.completion_tokens is missing")
    if not isinstance(token_count, int) or isinstance(token_count, bool):
        raise TypeError("OpenAI response usage.completion_tokens must be an integer")
    if token_count < 0:
        raise ValueError("OpenAI response usage.completion_tokens cannot be negative")
    return token_count


def _mean_logprob(choice: Any) -> float | None:
    logprobs = _field(choice, "logprobs", None)
    if logprobs is None:
        return None
    content = _field(logprobs, "content", None)
    if content is None:
        return None
    if isinstance(content, (str, bytes, Mapping)):
        raise TypeError("OpenAI choice logprobs.content must be a sequence of token logprobs")
    try:
        tokens = list(content)
    except TypeError as error:
        raise TypeError(
            "OpenAI choice logprobs.content must be a sequence of token logprobs"
        ) from error
    if not tokens:
        return None

    values: list[float] = []
    for index, token in enumerate(tokens):
        value = _field(token, "logprob")
        if value is _MISSING:
            raise ValueError(f"OpenAI choice logprobs.content[{index}].logprob is missing")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"OpenAI choice logprobs.content[{index}].logprob must be a real number"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"OpenAI choice logprobs.content[{index}].logprob must be finite")
        values.append(number)

    mean = math.fsum(values) / len(values)
    if not math.isfinite(mean):
        raise ValueError("OpenAI choice mean logprob must be finite")
    return mean


@dataclass(frozen=True, slots=True, init=False)
class OpenAIChatSampler:
    """Adapt one OpenAI-compatible chat completion into one BranchPilot sample."""

    client: Any
    model: str
    messages: tuple[Mapping[str, Any], ...]
    extractor: Callable[[str], str | None]
    request_options: Mapping[str, Any]

    def __init__(
        self,
        client: Any,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        extractor: Callable[[str], str | None],
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model cannot be empty")
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of mappings")
        frozen_messages: list[Mapping[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] must be a mapping")
            frozen_messages.append(_freeze(message))
        if not callable(extractor):
            raise TypeError("extractor must be callable")
        if request_options is None:
            request_options = {}
        if not isinstance(request_options, Mapping):
            raise TypeError("request_options must be a mapping")
        reserved = _RESERVED_REQUEST_OPTIONS.intersection(request_options)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"request_options cannot override reserved option(s): {names}")

        object.__setattr__(self, "client", client)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "messages", tuple(frozen_messages))
        object.__setattr__(self, "extractor", extractor)
        object.__setattr__(self, "request_options", _freeze(request_options))

    async def __call__(self, sample_index: int) -> Sample:
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise TypeError("sample_index must be an integer")
        if sample_index < 1:
            raise ValueError("sample_index must be positive")

        options = _thaw(self.request_options)
        response = await self.client.chat.completions.create(
            **options,
            model=self.model,
            messages=_thaw(self.messages),
            n=1,
            stream=False,
        )

        choices = _field(response, "choices")
        if choices is _MISSING or choices is None:
            raise ValueError("OpenAI response choices are missing; expected exactly one choice")
        if isinstance(choices, (str, bytes, Mapping)):
            raise TypeError("OpenAI response choices must be a sequence with exactly one choice")
        try:
            choice_list = list(choices)
        except TypeError as error:
            raise TypeError(
                "OpenAI response choices must be a sequence with exactly one choice"
            ) from error
        if len(choice_list) != 1:
            raise ValueError(
                f"OpenAI response must contain exactly one choice; received {len(choice_list)}"
            )
        choice = choice_list[0]

        message = _field(choice, "message")
        if message is _MISSING or message is None:
            raise ValueError("OpenAI response choices[0].message is missing")
        text = _field(message, "content")
        if text is _MISSING or text is None:
            raise ValueError("OpenAI response choices[0].message.content is missing")
        if not isinstance(text, str):
            raise TypeError("OpenAI response choices[0].message.content must be a string")

        finish_reason = _field(choice, "finish_reason", None)
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TypeError("OpenAI response choices[0].finish_reason must be a string or None")

        token_count = _completion_tokens(response)
        mean_logprob = _mean_logprob(choice)
        if finish_reason == "length":
            answer = None
            parse_status = "truncated"
        else:
            answer = self.extractor(text)
            if answer is not None and not isinstance(answer, str):
                raise TypeError("extractor must return a string or None")
            parse_status = "parsed" if answer is not None else "unparsed"

        return Sample(
            text=text,
            answer=answer,
            token_count=token_count,
            mean_logprob=mean_logprob,
            finish_reason=finish_reason,
            parse_status=parse_status,
        )


async def run_openai(
    policy: Any,
    client: Any,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    extractor: Callable[[str], str | None],
    question: str,
    cost: float,
    prompt_tokens: int = 0,
    max_samples: int | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> PilotResult:
    """Run live chat requests only until the policy's session stops."""
    sampler = OpenAIChatSampler(
        client,
        model,
        messages,
        extractor,
        request_options=request_options,
    )
    session = policy.start(
        question,
        cost,
        prompt_tokens=prompt_tokens,
        max_samples=max_samples,
    )
    return await session.run_async(sampler)
