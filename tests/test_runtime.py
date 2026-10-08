import asyncio
from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import pytest

from branchpilot.policy import Decision
from branchpilot.runtime import PilotSession
from branchpilot.schema import Sample


class FakePolicy:
    def __init__(self, *, max_samples: int = 5, stop_after: int = 3) -> None:
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
        snapshot = tuple(samples)
        self.calls.append((question, snapshot, cost, prompt_tokens, max_samples))
        horizon = self.max_samples if max_samples is None else max_samples
        action = "stop" if len(snapshot) >= min(self.stop_after, horizon) else "continue"
        return Decision(
            action=action,
            q_stop=float(len(snapshot)),
            q_continue=float(horizon - len(snapshot)),
            sample_count=len(snapshot),
            majority_answer=snapshot[-1].answer,
        )


def make_sample(index: int) -> Sample:
    return Sample(text=f"completion {index}", answer=f"answer-{index}", token_count=index)


def test_session_stops_at_configured_horizon() -> None:
    policy = FakePolicy(max_samples=5, stop_after=99)
    sampled: list[int] = []

    def sampler(index: int) -> Sample:
        sampled.append(index)
        return make_sample(index)

    session = PilotSession(
        policy,
        "What is the answer?",
        0.25,
        prompt_tokens=11,
        max_samples=3,
    )
    result = session.run(sampler)

    assert sampled == [1, 2, 3]
    assert session.stopped
    assert not session.should_continue
    assert result.answer == "answer-3"
    assert result.sample_count == 3
    assert result.completion_tokens == 6
    assert result.cost == 0.25
    assert result.samples == tuple(make_sample(index) for index in range(1, 4))
    assert len(result.decisions) == 3
    assert [len(call[1]) for call in policy.calls] == [1, 2, 3]
    assert all(call[0] == "What is the answer?" for call in policy.calls)
    assert all(call[2:] == (0.25, 11, 3) for call in policy.calls)


def test_horizon_defaults_to_policy_maximum() -> None:
    policy = FakePolicy(max_samples=2, stop_after=99)
    sampled: list[int] = []

    result = PilotSession(policy, "question", 0.0).run(
        lambda index: sampled.append(index) or make_sample(index)
    )

    assert sampled == [1, 2]
    assert result.sample_count == 2
    assert all(call[4] == 2 for call in policy.calls)


def test_early_stop_prevents_more_generation_or_observation() -> None:
    policy = FakePolicy(stop_after=2)
    sampled: list[int] = []

    def sampler(index: int) -> Sample:
        sampled.append(index)
        return make_sample(index)

    session = PilotSession(policy, "question", 0.1)
    first_result = session.run(sampler)
    call_count = len(policy.calls)

    assert sampled == [1, 2]
    assert session.run(sampler) == first_result
    assert sampled == [1, 2]
    assert len(policy.calls) == call_count
    with pytest.raises(RuntimeError, match="after.*stopped"):
        session.observe(make_sample(3))
    assert len(session.samples) == 2
    assert len(policy.calls) == call_count


def test_sync_and_async_runs_have_identical_behavior() -> None:
    sync_policy = FakePolicy(stop_after=3)
    async_policy = FakePolicy(stop_after=3)
    sync_indices: list[int] = []
    async_indices: list[int] = []

    def sampler(index: int) -> Sample:
        sync_indices.append(index)
        return make_sample(index)

    async def async_sampler(index: int) -> Sample:
        async_indices.append(index)
        return make_sample(index)

    sync_result = PilotSession(sync_policy, "question", 0.2, prompt_tokens=4).run(sampler)
    async_result = asyncio.run(
        PilotSession(async_policy, "question", 0.2, prompt_tokens=4).run_async(async_sampler)
    )

    assert sync_indices == async_indices == [1, 2, 3]
    assert async_result == sync_result
    assert async_policy.calls == sync_policy.calls


def test_result_is_rejected_before_stop() -> None:
    session = PilotSession(FakePolicy(stop_after=2), "question", 0.0)

    with pytest.raises(RuntimeError, match="before.*stopped"):
        session.result()
    decision = session.observe(make_sample(1))
    assert decision.action == "continue"
    with pytest.raises(RuntimeError, match="before.*stopped"):
        session.result()


def test_sample_and_decision_views_are_immutable_snapshots() -> None:
    session = PilotSession(FakePolicy(stop_after=2), "question", 0.0)
    first_decision = session.observe(make_sample(1))
    sample_snapshot = session.samples
    decision_snapshot = session.decisions

    assert isinstance(sample_snapshot, tuple)
    assert isinstance(decision_snapshot, tuple)
    session.observe(make_sample(2))
    result = session.result()

    assert sample_snapshot == (make_sample(1),)
    assert decision_snapshot == (first_decision,)
    assert len(session.samples) == len(session.decisions) == 2
    with pytest.raises(TypeError):
        result.samples[0] = make_sample(9)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.cost = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question: str) -> None:
    with pytest.raises(ValueError, match="question"):
        PilotSession(FakePolicy(), question, 0.0)


@pytest.mark.parametrize("cost", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_invalid_cost_is_rejected(cost: float) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        PilotSession(FakePolicy(), "question", cost)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"prompt_tokens": -1}, ValueError, "prompt_tokens"),
        ({"prompt_tokens": 1.5}, TypeError, "prompt_tokens"),
        ({"max_samples": 0}, ValueError, "max_samples"),
        ({"max_samples": 6}, ValueError, "max_samples"),
        ({"max_samples": 1.5}, TypeError, "max_samples"),
    ],
)
def test_invalid_token_and_horizon_bounds_are_rejected(
    kwargs: dict[str, float], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        PilotSession(FakePolicy(max_samples=5), "question", 0.0, **kwargs)  # type: ignore[arg-type]


class NeverStopPolicy(FakePolicy):
    def decide_observed(self, question, samples, cost, *, prompt_tokens=0, max_samples=None):
        decision = super().decide_observed(
            question, samples, cost, prompt_tokens=prompt_tokens, max_samples=max_samples
        )
        return Decision(
            action="continue",
            q_stop=decision.q_stop,
            q_continue=decision.q_continue,
            sample_count=decision.sample_count,
            majority_answer=decision.majority_answer,
        )


def test_policy_that_ignores_the_horizon_never_triggers_an_extra_sample() -> None:
    sampled: list[int] = []
    session = PilotSession(NeverStopPolicy(max_samples=5), "question", 0.0, max_samples=3)

    with pytest.raises(RuntimeError, match="failed to stop at the session horizon"):
        session.run(lambda index: sampled.append(index) or make_sample(index))

    assert sampled == [1, 2, 3]
    assert not session.stopped


def test_async_run_also_refuses_to_sample_past_the_horizon() -> None:
    sampled: list[int] = []

    async def sampler(index: int) -> Sample:
        sampled.append(index)
        return make_sample(index)

    session = PilotSession(NeverStopPolicy(max_samples=2), "question", 0.0)
    with pytest.raises(RuntimeError, match="failed to stop at the session horizon"):
        asyncio.run(session.run_async(sampler))

    assert sampled == [1, 2]
