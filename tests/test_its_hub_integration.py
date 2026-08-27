from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from branchpilot.policy import Decision
from branchpilot.schema import Sample

try:
    from its_hub.api import ChatMessage
except ImportError:
    ChatMessage = None  # type: ignore[assignment,misc]
    _HAS_ITS_HUB = False
else:
    from branchpilot.integrations.its_hub import (
        BranchPilotAlgorithm,
        BranchPilotScalingResult,
    )

    _HAS_ITS_HUB = True

requires_its_hub = pytest.mark.skipif(
    not _HAS_ITS_HUB,
    reason="its-hub 1.2.0 is an optional integration dependency",
)


class FakeLanguageModel:
    def __init__(self) -> None:
        self.closed_loops: list[Any] = []

    async def close_session(self, loop: Any) -> None:
        self.closed_loops.append(loop)


class RecordingOrchestrator:
    def __init__(
        self,
        responses: Sequence[dict],
        usage_increments: Sequence[tuple[int, int]],
    ) -> None:
        self.responses = list(responses)
        self.usage_increments = list(usage_increments)
        self.calls: list[tuple[Any, list[list[Any]], dict[str, Any]]] = []

    async def agenerate(
        self,
        lm: Any,
        messages_lst: list[list[Any]],
        **kwargs: Any,
    ) -> list[dict]:
        self.calls.append((lm, messages_lst, kwargs.copy()))
        prompt_tokens, completion_tokens = self.usage_increments.pop(0)
        kwargs["usage_accumulator"].add(prompt_tokens, completion_tokens)
        return [self.responses.pop(0)]


class RecordingStrategy:
    def __init__(self, *, max_samples: int = 5, stop_after: int = 2) -> None:
        self.max_samples = max_samples
        self.stop_after = stop_after
        self.calls: list[tuple[str, tuple[Sample, ...], float, int, int | None]] = []

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        observed = tuple(samples)
        self.calls.append((question, observed, cost, prompt_tokens, max_samples))
        horizon = self.max_samples if max_samples is None else max_samples
        counts = Counter(sample.answer for sample in observed if sample.answer is not None)
        majority = None
        if counts:
            highest = max(counts.values())
            majority = next(
                sample.answer
                for sample in observed
                if sample.answer is not None and counts[sample.answer] == highest
            )
        action = "stop" if len(observed) >= min(self.stop_after, horizon) else "continue"
        return Decision(
            action=action,
            q_stop=float(len(observed)),
            q_continue=float(horizon - len(observed)),
            sample_count=len(observed),
            majority_answer=majority,
        )


def sample_converter(
    deltas: list[int] | None = None,
) -> Callable[[dict, int], Sample]:
    def convert(response: dict, completion_token_delta: int) -> Sample:
        if deltas is not None:
            deltas.append(completion_token_delta)
        answer = response.get("answer")
        return Sample(
            text=response.get("content") or "",
            answer=answer,
            token_count=completion_token_delta,
            parse_status="parsed" if answer is not None else "unparsed",
        )

    return convert


@requires_its_hub
def test_full_result_stops_early_aggregates_usage_and_selects_first_majority() -> None:
    responses = [
        {"role": "assistant", "content": "no parse"},
        {"role": "assistant", "content": "first A", "answer": "A"},
        {"role": "assistant", "content": "second A", "answer": "A"},
        {"role": "assistant", "content": "must not run", "answer": "B"},
    ]
    orchestrator = RecordingOrchestrator(
        responses,
        [(11, 3), (11, 5), (11, 7), (11, 100)],
    )
    strategy = RecordingStrategy(max_samples=5, stop_after=3)
    seen_deltas: list[int] = []
    algorithm = BranchPilotAlgorithm(
        strategy,
        cost=0.25,
        response_to_sample=sample_converter(seen_deltas),
        orchestrator=orchestrator,
    )
    lm = FakeLanguageModel()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    tool_choice = {"type": "function", "function": {"name": "lookup"}}

    result = algorithm.infer(
        lm,
        "Solve this",
        budget=4,
        return_response_only=False,
        tools=tools,
        tool_choice=tool_choice,
    )

    assert isinstance(result, BranchPilotScalingResult)
    assert len(orchestrator.calls) == 3
    assert seen_deltas == [3, 5, 7]
    assert result.responses == responses[:3]
    assert all(
        result.responses[index] is responses[index] for index in range(len(result.responses))
    )
    assert result.response_counts == Counter({"A": 2, "<unparsed:0>": 1})
    assert result.selected_index == 1
    assert result.the_one is responses[1]
    assert result.usage.prompt_tokens == 33
    assert result.usage.completion_tokens == 15
    assert result.usage.num_calls == 3
    assert len(result.decisions) == 3
    assert [call[3] for call in strategy.calls] == [11, 11, 11]
    assert [call[4] for call in strategy.calls] == [4, 4, 4]
    assert all(call[0] is lm for call in orchestrator.calls)
    assert all(call[1][0][0].role == "user" for call in orchestrator.calls)
    assert all(call[1][0][0].content == "Solve this" for call in orchestrator.calls)
    assert all(call[2]["tools"] is tools for call in orchestrator.calls)
    assert all(call[2]["tool_choice"] is tool_choice for call in orchestrator.calls)
    assert lm.closed_loops


@pytest.mark.parametrize(
    ("budget", "policy_max_samples", "expected_calls"),
    [(1, 5, 1), (9, 2, 2)],
)
@requires_its_hub
def test_budget_and_policy_max_samples_bound_the_sequential_horizon(
    budget: int,
    policy_max_samples: int,
    expected_calls: int,
) -> None:
    responses = [
        {"role": "assistant", "content": str(index), "answer": str(index)} for index in range(10)
    ]
    orchestrator = RecordingOrchestrator(responses, [(2, 1)] * 10)
    strategy = RecordingStrategy(max_samples=policy_max_samples, stop_after=99)
    algorithm = BranchPilotAlgorithm(
        strategy,
        cost=0.0,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )

    result = algorithm.infer(
        FakeLanguageModel(),
        "bounded",
        budget=budget,
        return_response_only=False,
    )

    assert len(orchestrator.calls) == expected_calls
    assert len(result.responses) == expected_calls
    assert all(call[4] == expected_calls for call in strategy.calls)


@requires_its_hub
def test_structured_messages_are_normalized_and_response_only_is_original_dict() -> None:
    response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}],
        "answer": "used-tool",
    }
    orchestrator = RecordingOrchestrator([response], [(7, 2)])
    strategy = RecordingStrategy(max_samples=3, stop_after=1)
    algorithm = BranchPilotAlgorithm(
        strategy,
        cost=1.0,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )
    messages = [
        ChatMessage(role="system", content="Follow rules"),
        ChatMessage(role="user", content="Find it"),
    ]

    selected = algorithm.infer(FakeLanguageModel(), messages, budget=3)

    assert selected is response
    assert strategy.calls[0][0] == "system: Follow rules\nuser: Find it"
    sent = orchestrator.calls[0][1]
    assert sent == [messages]
    assert sent is not messages
    assert orchestrator.calls[0][2]["tools"] is None
    assert orchestrator.calls[0][2]["tool_choice"] is None


@requires_its_hub
def test_unparsed_only_result_falls_back_to_response_zero() -> None:
    response = {"role": "assistant", "content": "uncertain"}
    orchestrator = RecordingOrchestrator([response], [(4, 6)])
    algorithm = BranchPilotAlgorithm(
        RecordingStrategy(max_samples=3, stop_after=1),
        cost=0.0,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )

    result = algorithm.infer(
        FakeLanguageModel(),
        "question",
        budget=3,
        return_response_only=False,
    )

    assert result.response_counts == Counter({"<unparsed:0>": 1})
    assert result.selected_index == 0
    assert result.the_one is response


@requires_its_hub
def test_max_samples_and_decide_observed_delegate_without_translation() -> None:
    class SentinelStrategy:
        max_samples = 7

        def __init__(self) -> None:
            self.received: tuple[Any, ...] | None = None
            self.decision = Decision("stop", 1.0, 0.0, 1, "x")

        def decide_observed(self, *args: Any, **kwargs: Any) -> Decision:
            self.received = (*args, kwargs)
            return self.decision

    strategy = SentinelStrategy()
    orchestrator = RecordingOrchestrator([], [])
    algorithm = BranchPilotAlgorithm(
        strategy,
        cost=0.5,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )
    samples = [Sample("x", "x", 1, parse_status="parsed")]

    decision = algorithm.decide_observed(
        "question",
        samples,
        0.75,
        prompt_tokens=13,
        max_samples=4,
    )

    assert algorithm.max_samples == 7
    assert decision is strategy.decision
    assert strategy.received is not None
    assert strategy.received[0] == "question"
    assert strategy.received[1] is samples
    assert strategy.received[2] == 0.75
    assert strategy.received[3] == {"prompt_tokens": 13, "max_samples": 4}


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5])
@requires_its_hub
def test_invalid_budget_is_rejected_before_generation(budget: Any) -> None:
    orchestrator = RecordingOrchestrator([], [])
    algorithm = BranchPilotAlgorithm(
        RecordingStrategy(),
        cost=0.0,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )

    with pytest.raises((TypeError, ValueError)):
        algorithm.infer(FakeLanguageModel(), "question", budget=budget)

    assert orchestrator.calls == []


@requires_its_hub
def test_empty_prompt_and_invalid_policy_horizon_are_rejected_before_generation() -> None:
    orchestrator = RecordingOrchestrator([], [])
    empty_prompt_algorithm = BranchPilotAlgorithm(
        RecordingStrategy(),
        cost=0.0,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )
    invalid_policy_algorithm = BranchPilotAlgorithm(
        RecordingStrategy(max_samples=0),
        cost=0.0,
        response_to_sample=sample_converter(),
        orchestrator=orchestrator,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        empty_prompt_algorithm.infer(FakeLanguageModel(), " ", budget=1)
    with pytest.raises(ValueError, match="max_samples"):
        invalid_policy_algorithm.infer(FakeLanguageModel(), "question", budget=1)

    assert orchestrator.calls == []


@requires_its_hub
def test_converter_must_return_sample_and_failure_closes_lm_without_extra_call() -> None:
    responses = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "must not run"},
    ]
    orchestrator = RecordingOrchestrator(responses, [(3, 2), (3, 2)])
    lm = FakeLanguageModel()

    def fail_converter(response: dict, completion_token_delta: int) -> Sample:
        raise LookupError("conversion failed")

    failing = BranchPilotAlgorithm(
        RecordingStrategy(stop_after=2),
        cost=0.0,
        response_to_sample=fail_converter,
        orchestrator=orchestrator,
    )
    with pytest.raises(LookupError, match="conversion failed"):
        failing.infer(lm, "question", budget=2)

    assert len(orchestrator.calls) == 1
    assert len(lm.closed_loops) == 1

    wrong_type_orchestrator = RecordingOrchestrator([responses[0]], [(3, 2)])
    wrong_type = BranchPilotAlgorithm(
        RecordingStrategy(stop_after=1),
        cost=0.0,
        response_to_sample=lambda response, delta: response,
        orchestrator=wrong_type_orchestrator,
    )
    with pytest.raises(TypeError, match="must return.*Sample"):
        wrong_type.infer(FakeLanguageModel(), "question", budget=1)
    assert len(wrong_type_orchestrator.calls) == 1


@pytest.mark.parametrize("cost", [float("inf"), float("-inf"), float("nan"), -0.1])
@requires_its_hub
def test_cost_must_be_finite_and_non_negative(cost: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        BranchPilotAlgorithm(
            RecordingStrategy(),
            cost=cost,
            response_to_sample=sample_converter(),
            orchestrator=RecordingOrchestrator([], []),
        )


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_importing_branchpilot_core_does_not_import_optional_its_hub() -> None:
    completed = _run_isolated(
        "import branchpilot, sys; "
        "assert not any(name == 'its_hub' or name.startswith('its_hub.') "
        "for name in sys.modules)"
    )

    assert completed.returncode == 0, completed.stderr


def test_adapter_import_error_names_the_missing_optional_dependency() -> None:
    completed = _run_isolated(
        "import builtins; "
        "original = builtins.__import__; "
        "builtins.__import__ = lambda name, *args, **kwargs: "
        "(_ for _ in ()).throw(ImportError('blocked')) "
        "if name == 'its_hub' or name.startswith('its_hub.') "
        "else original(name, *args, **kwargs); "
        "\ntry:\n import branchpilot.integrations.its_hub\n"
        "except ImportError as error:\n"
        " assert 'optional' in str(error) and 'its-hub>=1.2.0' in str(error)\n"
        "else:\n raise AssertionError('adapter import unexpectedly succeeded')"
    )

    assert completed.returncode == 0, completed.stderr
