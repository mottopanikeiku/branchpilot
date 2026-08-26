import asyncio
import copy
import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest

from branchpilot.integrations.openai import OpenAIChatSampler, run_openai
from branchpilot.policy import Decision
from branchpilot.schema import Sample

_MISSING = object()


def make_response(
    text: Any = "work\n#### 42",
    *,
    completion_tokens: Any = 4,
    finish_reason: Any = "stop",
    token_logprobs: Any = _MISSING,
    choices: Any = _MISSING,
    usage: Any = _MISSING,
) -> SimpleNamespace:
    if choices is _MISSING:
        choice_fields: dict[str, Any] = {
            "message": SimpleNamespace(content=text),
            "finish_reason": finish_reason,
        }
        if token_logprobs is not _MISSING:
            choice_fields["logprobs"] = (
                None
                if token_logprobs is None
                else SimpleNamespace(
                    content=[SimpleNamespace(logprob=value) for value in token_logprobs]
                )
            )
        choices = [SimpleNamespace(**choice_fields)]
    fields: dict[str, Any] = {"choices": choices}
    if usage is _MISSING:
        fields["usage"] = SimpleNamespace(completion_tokens=completion_tokens)
    elif usage is not None:
        fields["usage"] = usage
    return SimpleNamespace(**fields)


class FakeCompletions:
    def __init__(self, responses: Sequence[Any], *, mutate_messages: bool = False) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.received_messages: list[list[dict[str, Any]]] = []
        self.mutate_messages = mutate_messages

    async def create(self, **kwargs: Any) -> Any:
        self.received_messages.append(copy.deepcopy(kwargs["messages"]))
        self.calls.append(kwargs)
        if self.mutate_messages:
            kwargs["messages"][0]["content"][0]["text"] = "changed by client"
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: Sequence[Any], *, mutate_messages: bool = False) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions(responses, mutate_messages=mutate_messages)
        )


class FakePolicy:
    def __init__(self, *, max_samples: int = 5, stop_after: int = 2) -> None:
        self.max_samples = max_samples
        self.stop_after = stop_after
        self.decide_calls: list[tuple[str, float, int, int | None, int]] = []

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        self.decide_calls.append((question, cost, prompt_tokens, max_samples, len(samples)))
        horizon = self.max_samples if max_samples is None else max_samples
        action = "stop" if len(samples) >= min(self.stop_after, horizon) else "continue"
        return Decision(
            action=action,
            q_stop=float(len(samples)),
            q_continue=float(horizon - len(samples)),
            sample_count=len(samples),
            majority_answer=samples[-1].answer,
        )


def extract_hash_answer(text: str) -> str | None:
    marker = "#### "
    return text.rsplit(marker, maxsplit=1)[1] if marker in text else None


def test_run_openai_stops_live_requests_and_forwards_request_configuration() -> None:
    client = FakeClient(
        [
            make_response(
                "first\n#### 41",
                completion_tokens=3,
                token_logprobs=[-0.5, -1.5],
            ),
            make_response(
                "second\n#### 42",
                completion_tokens=5,
                token_logprobs=[-1.0, -2.0, -3.0],
            ),
            make_response("must never be requested"),
        ]
    )
    policy = FakePolicy(max_samples=5, stop_after=2)
    messages = [{"role": "user", "content": "Solve it"}]

    result = asyncio.run(
        run_openai(
            policy,
            client,
            "chat-model",
            messages,
            extract_hash_answer,
            "What is 6 * 7?",
            0.25,
            prompt_tokens=9,
            max_samples=4,
            request_options={
                "temperature": 0.7,
                "max_completion_tokens": 64,
                "logprobs": True,
            },
        )
    )

    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls == [
        {
            "model": "chat-model",
            "messages": messages,
            "n": 1,
            "stream": False,
            "temperature": 0.7,
            "max_completion_tokens": 64,
            "logprobs": True,
        },
        {
            "model": "chat-model",
            "messages": messages,
            "n": 1,
            "stream": False,
            "temperature": 0.7,
            "max_completion_tokens": 64,
            "logprobs": True,
        },
    ]
    assert policy.decide_calls == [
        ("What is 6 * 7?", 0.25, 9, 4, 1),
        ("What is 6 * 7?", 0.25, 9, 4, 2),
    ]
    assert result.sample_count == 2
    assert result.answer == "42"
    assert result.completion_tokens == 8
    assert result.samples == (
        Sample(
            text="first\n#### 41",
            answer="41",
            token_count=3,
            mean_logprob=-1.0,
            finish_reason="stop",
            parse_status="parsed",
        ),
        Sample(
            text="second\n#### 42",
            answer="42",
            token_count=5,
            mean_logprob=-2.0,
            finish_reason="stop",
            parse_status="parsed",
        ),
    )


def test_sampler_owns_immutable_messages_and_sends_a_fresh_copy_per_call() -> None:
    original = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "original"}],
        }
    ]
    client = FakeClient(
        [make_response(), make_response()],
        mutate_messages=True,
    )
    sampler = OpenAIChatSampler(
        client,
        "model",
        original,
        extract_hash_answer,
        request_options={"metadata": {"source": "test"}},
    )

    original[0]["role"] = "assistant"
    original[0]["content"][0]["text"] = "changed by caller"
    with pytest.raises(TypeError):
        sampler.messages[0]["role"] = "assistant"  # type: ignore[index]
    with pytest.raises(TypeError):
        sampler.messages[0]["content"][0]["text"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        sampler.model = "other"  # type: ignore[misc]

    asyncio.run(sampler(1))
    asyncio.run(sampler(2))

    assert client.chat.completions.received_messages == [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "original"}],
            }
        ],
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "original"}],
            }
        ],
    ]
    assert all(
        call["messages"][0]["content"][0]["text"] == "changed by client"
        for call in client.chat.completions.calls
    )
    assert sampler.messages[0]["content"][0]["text"] == "original"


@pytest.mark.parametrize("token_logprobs", [_MISSING, None, []])
def test_missing_logprobs_are_recorded_explicitly_as_none(token_logprobs: Any) -> None:
    response = make_response(token_logprobs=token_logprobs)
    sampler = OpenAIChatSampler(FakeClient([response]), "model", [], extract_hash_answer)

    sample = asyncio.run(sampler(1))

    assert sample.mean_logprob is None
    assert sample.answer == "42"
    assert sample.parse_status == "parsed"


def test_extractor_miss_is_an_explicit_unparsed_sample() -> None:
    sampler = OpenAIChatSampler(
        FakeClient([make_response("reasoning without a canonical answer")]),
        "model",
        [],
        extract_hash_answer,
    )

    sample = asyncio.run(sampler(1))

    assert sample.answer is None
    assert sample.parse_status == "unparsed"


def test_length_truncation_never_calls_extractor_or_contributes_an_answer() -> None:
    extracted: list[str] = []

    def extractor(text: str) -> str:
        extracted.append(text)
        return "42"

    sampler = OpenAIChatSampler(
        FakeClient(
            [
                make_response(
                    "partial text containing #### 42",
                    finish_reason="length",
                    token_logprobs=[-0.25],
                )
            ]
        ),
        "model",
        [],
        extractor,
    )

    sample = asyncio.run(sampler(1))

    assert extracted == []
    assert sample.answer is None
    assert sample.parse_status == "truncated"
    assert sample.finish_reason == "length"
    assert sample.mean_logprob == -0.25


@pytest.mark.parametrize("finish_reason", (None, "content_filter", "tool_calls", "aborted"))
def test_abnormal_finish_reasons_never_vote(finish_reason: str | None) -> None:
    extracted: list[str] = []
    sampler = OpenAIChatSampler(
        FakeClient([make_response("partial #### 42", finish_reason=finish_reason)]),
        "model",
        [],
        lambda text: extracted.append(text) or "42",
    )

    sample = asyncio.run(sampler(1))

    assert extracted == []
    assert sample.answer is None
    assert sample.parse_status == "incomplete"
    assert sample.finish_reason == finish_reason


@pytest.mark.parametrize("reserved", ["model", "messages", "n", "stream"])
def test_reserved_request_options_are_rejected(reserved: str) -> None:
    with pytest.raises(ValueError, match=reserved):
        OpenAIChatSampler(
            FakeClient([]),
            "model",
            [],
            extract_hash_answer,
            request_options={reserved: "override"},
        )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (SimpleNamespace(usage=SimpleNamespace(completion_tokens=1)), "choices.*missing"),
        (make_response(choices=[]), "exactly one choice; received 0"),
        (
            make_response(choices=[SimpleNamespace(), SimpleNamespace()]),
            "exactly one choice; received 2",
        ),
    ],
)
def test_missing_or_multiple_choices_are_rejected(response: SimpleNamespace, message: str) -> None:
    sampler = OpenAIChatSampler(FakeClient([response]), "model", [], extract_hash_answer)

    with pytest.raises(ValueError, match=message):
        asyncio.run(sampler(1))


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        (make_response(usage=None), ValueError, "usage is missing"),
        (
            make_response(usage=SimpleNamespace()),
            ValueError,
            "completion_tokens is missing",
        ),
        (make_response(completion_tokens=None), ValueError, "completion_tokens is missing"),
        (make_response(completion_tokens=True), TypeError, "must be an integer"),
        (make_response(completion_tokens=1.5), TypeError, "must be an integer"),
        (make_response(completion_tokens=-1), ValueError, "cannot be negative"),
    ],
)
def test_missing_or_invalid_completion_usage_is_rejected(
    response: SimpleNamespace, error: type[Exception], message: str
) -> None:
    sampler = OpenAIChatSampler(FakeClient([response]), "model", [], extract_hash_answer)

    with pytest.raises(error, match=message):
        asyncio.run(sampler(1))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_token_logprobs_are_rejected(value: float) -> None:
    sampler = OpenAIChatSampler(
        FakeClient([make_response(token_logprobs=[-0.5, value])]),
        "model",
        [],
        extract_hash_answer,
    )

    with pytest.raises(ValueError, match=r"logprobs\.content\[1\]\.logprob must be finite"):
        asyncio.run(sampler(1))


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        (
            make_response(choices=[SimpleNamespace(finish_reason="stop")]),
            ValueError,
            r"choices\[0\]\.message is missing",
        ),
        (
            make_response(
                choices=[SimpleNamespace(message=SimpleNamespace(), finish_reason="stop")]
            ),
            ValueError,
            r"message\.content is missing",
        ),
        (
            make_response(text=None),
            ValueError,
            r"message\.content is missing",
        ),
        (
            make_response(text=[{"type": "text", "text": "not flattened"}]),
            TypeError,
            r"message\.content must be a string",
        ),
    ],
)
def test_malformed_choice_content_is_rejected(
    response: SimpleNamespace, error: type[Exception], message: str
) -> None:
    sampler = OpenAIChatSampler(FakeClient([response]), "model", [], extract_hash_answer)

    with pytest.raises(error, match=message):
        asyncio.run(sampler(1))
