from __future__ import annotations

import pytest
from pydantic import ValidationError

from branchpilot.gateway.schemas import MAX_COMPLETION_TOKENS, ChatCompletionRequest


def _request(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "What is 6 * 7?"}],
    }
    value.update(changes)
    return value


def test_accepts_text_common_options_and_namespaced_overrides() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=-0.2,
            presence_penalty=0.1,
            max_completion_tokens=100,
            stop=["one", "two"],
            seed=4,
            logit_bias={"42": 10},
            logprobs=True,
            top_logprobs=3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            user="tenant_1",
            n=1,
            stream=False,
            branchpilot={"strategy": "fixed-2", "max_samples": 2},
        )
    )

    assert request.question() == "What is 6 * 7?"
    body = request.upstream_body()
    assert body["n"] == 1
    assert body["stream"] is False
    assert "branchpilot" not in body
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 2),
        ("stream", True),
        ("tools", []),
        ("functions", []),
        ("audio", {}),
        ("modalities", ["audio"]),
        ("unknown_vendor_field", 1),
    ],
)
def test_rejects_unsupported_and_unknown_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_request(**{field: value}))


def test_rejects_multimodal_message_content_and_unknown_message_fields() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            _request(messages=[{"role": "user", "content": [{"type": "text", "text": "x"}]}])
        )
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            _request(messages=[{"role": "user", "content": "x", "tool_call_id": "call"}])
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"messages": [{"role": "assistant", "content": "answer"}]},
        {"messages": [{"role": "user", "content": "  "}]},
        {"temperature": float("nan")},
        {"top_p": 1.1},
        {"frequency_penalty": -2.1},
        {"stop": ["1", "2", "3", "4", "5"]},
        {"max_tokens": 1, "max_completion_tokens": 1},
        {"max_tokens": MAX_COMPLETION_TOKENS + 1},
        {"max_completion_tokens": MAX_COMPLETION_TOKENS + 1},
        {"top_logprobs": 1},
        {"branchpilot": {"cost": float("inf")}},
        {"branchpilot": {"upstream_url": "https://attacker.invalid"}},
    ],
)
def test_rejects_invalid_combinations_and_ranges(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_request(**changes))


def test_uses_final_nonempty_user_message_as_question() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "final"},
            ]
        )
    )
    assert request.question() == "final"
