import math
from dataclasses import FrozenInstanceError

import pytest

from branchpilot.evaluate import agreement_rule, confidence_rule, fixed_rule
from branchpilot.policy import MAX_SAMPLES
from branchpilot.runtime import PilotSession
from branchpilot.schema import Rollout, Sample
from branchpilot.strategies import (
    ConsecutiveAgreementStrategy,
    FixedStrategy,
    StoppingStrategy,
    VoteConfidenceStrategy,
    strategy_from_spec,
)


def _sample(answer: str | None, index: int = 1, mean_logprob: float | None = None) -> Sample:
    return Sample(
        text=f"completion {index}",
        answer=answer,
        token_count=index,
        mean_logprob=mean_logprob,
    )


def _rollout(answers: list[str | None]) -> Rollout:
    return Rollout(
        uid="rollout",
        question="What is the answer?",
        gold="unused-gold",
        samples=tuple(_sample(answer, index) for index, answer in enumerate(answers, start=1)),
        prompt_tokens=17,
    )


def _live_stop_count(strategy: StoppingStrategy, rollout: Rollout) -> int:
    horizon = len(rollout.samples)
    for count in range(1, horizon + 1):
        decision = strategy.decide_observed(
            rollout.question,
            rollout.samples[:count],
            0.25,
            prompt_tokens=rollout.prompt_tokens,
            max_samples=horizon,
        )
        if decision.action == "stop":
            return count
    raise AssertionError("strategy did not stop at the horizon")


@pytest.mark.parametrize(
    ("strategy", "rule", "answers"),
    [
        (FixedStrategy(samples=3, max_samples=7), fixed_rule(3), ["A"] * 7),
        (
            VoteConfidenceStrategy(threshold=0.6, minimum=2, max_samples=7),
            confidence_rule(0.6, minimum=2),
            ["A", "B", "A", "C", "A", "A", None],
        ),
        (
            ConsecutiveAgreementStrategy(streak=2, max_samples=7),
            agreement_rule(2),
            ["A", None, None, "B", "B", "C", "C"],
        ),
    ],
)
@pytest.mark.parametrize("prefix_length", range(1, 8))
def test_live_strategy_matches_evaluator_rule_at_every_prefix(
    strategy: StoppingStrategy,
    rule,
    answers: list[str | None],
    prefix_length: int,
) -> None:
    rollout = _rollout(answers[:prefix_length])

    assert _live_stop_count(strategy, rollout) == rule(rollout)


@pytest.mark.parametrize(
    "strategy",
    [
        FixedStrategy(samples=5, max_samples=5),
        VoteConfidenceStrategy(threshold=1.0, minimum=5, max_samples=5),
        ConsecutiveAgreementStrategy(streak=5, max_samples=5),
    ],
)
def test_session_horizon_forces_stop_before_strategy_limit(strategy: StoppingStrategy) -> None:
    samples = tuple(_sample(answer, index) for index, answer in enumerate("ABC", start=1))
    actions = [
        strategy.decide_observed("question", samples[:count], 0.0, max_samples=3).action
        for count in range(1, 4)
    ]

    assert actions == ["continue", "continue", "stop"]


@pytest.mark.parametrize(
    ("strategy", "answers"),
    [
        (FixedStrategy(samples=2, max_samples=5), ["A", "B", "C"]),
        (
            VoteConfidenceStrategy(threshold=1.0, minimum=2, max_samples=5),
            ["A", "A", "B"],
        ),
        (ConsecutiveAgreementStrategy(streak=2, max_samples=5), ["A", "A", "B"]),
    ],
)
def test_rules_stop_early_when_their_condition_is_met(
    strategy: StoppingStrategy, answers: list[str]
) -> None:
    rollout = _rollout(answers)

    assert _live_stop_count(strategy, rollout) == 2


@pytest.mark.parametrize(
    ("strategy", "expected_calls"),
    [
        (FixedStrategy(samples=2, max_samples=5), 2),
        (VoteConfidenceStrategy(threshold=1.0, minimum=2, max_samples=5), 2),
        (ConsecutiveAgreementStrategy(streak=3, max_samples=5), 3),
    ],
)
def test_pilot_session_calls_sampler_exactly_until_stop(
    strategy: StoppingStrategy, expected_calls: int
) -> None:
    calls: list[int] = []

    def sampler(index: int) -> Sample:
        calls.append(index)
        return _sample("A", index)

    result = PilotSession(strategy, "question", 0.1).run(sampler)

    assert calls == list(range(1, expected_calls + 1))
    assert result.sample_count == expected_calls
    assert len(result.decisions) == expected_calls
    assert result.decisions[-1].action == "stop"


def test_vote_confidence_uses_top_vote_share_for_unparsed_tie() -> None:
    strategy = VoteConfidenceStrategy(threshold=0.5, minimum=2, max_samples=4)
    samples = (_sample(None, 1), _sample("A", 2))

    decision = strategy.decide_observed("question", samples, 0.0)

    assert decision.action == "stop"
    assert decision.majority_answer is None


def test_vote_tie_breaking_matches_prefix_state_semantics() -> None:
    strategy = VoteConfidenceStrategy(threshold=0.5, minimum=2, max_samples=4)
    samples = (_sample("A", 1, -2.0), _sample("B", 2, -1.0))

    decision = strategy.decide_observed("question", samples, 0.0)

    assert decision.action == "stop"
    assert decision.majority_answer == "B"


def test_unparsed_answers_never_extend_an_agreement_streak() -> None:
    strategy = ConsecutiveAgreementStrategy(streak=2, max_samples=5)
    rollout = _rollout(["A", None, None, "A", "A"])

    assert _live_stop_count(strategy, rollout) == 5
    assert _live_stop_count(strategy, rollout) == agreement_rule(2)(rollout)


def test_single_unparsed_answer_does_not_satisfy_agreement() -> None:
    strategy = ConsecutiveAgreementStrategy(streak=1, max_samples=3)

    first = strategy.decide_observed("question", (_sample(None),), 0.0)
    second = strategy.decide_observed("question", (_sample(None), _sample("A", 2)), 0.0)

    assert first.action == "continue"
    assert second.action == "stop"


@pytest.mark.parametrize(
    "strategy",
    [
        FixedStrategy(samples=3, max_samples=5),
        VoteConfidenceStrategy(threshold=0.75, minimum=2, max_samples=5),
        ConsecutiveAgreementStrategy(streak=2, max_samples=5),
    ],
)
def test_decision_depends_only_on_the_observed_prefix(strategy: StoppingStrategy) -> None:
    shared_prefix = (_sample("A", 1), _sample("B", 2))
    first = Rollout(
        uid="first",
        question="question",
        gold="A",
        samples=shared_prefix + (_sample("A", 3),),
    )
    second = Rollout(
        uid="second",
        question="question",
        gold="different-gold",
        samples=shared_prefix + (_sample("C", 3), _sample(None, 4)),
    )

    first_decision = strategy.decide_observed(first.question, first.samples[:2], 0.1, max_samples=5)
    second_decision = strategy.decide_observed(
        second.question, second.samples[:2], 0.1, max_samples=5
    )

    assert first_decision == second_decision
    assert math.isfinite(first_decision.q_stop)
    assert math.isfinite(first_decision.q_continue)
    assert (first_decision.q_stop > first_decision.q_continue) == (first_decision.action == "stop")


@pytest.mark.parametrize(
    "strategy",
    [
        FixedStrategy(samples=3, max_samples=8),
        VoteConfidenceStrategy(threshold=0.75, minimum=3, max_samples=8),
        ConsecutiveAgreementStrategy(streak=3, max_samples=8),
    ],
)
def test_strategy_specs_round_trip_exactly(strategy: StoppingStrategy) -> None:
    spec = strategy.to_spec()

    restored = strategy_from_spec(spec)

    assert restored == strategy
    assert restored.to_spec() == spec
    assert isinstance(restored, StoppingStrategy)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        [],
        {"type": 1, "samples": 1, "max_samples": 2},
        {"type": "unknown", "max_samples": 2},
        {"type": "fixed", "samples": 1},
        {"type": "fixed", "samples": 1, "max_samples": 2, "extra": False},
        {"type": "fixed", "samples": True, "max_samples": 2},
        {"type": "fixed", "samples": 1, "max_samples": 2.0},
        {
            "type": "vote_confidence",
            "threshold": 1,
            "minimum": 2,
            "max_samples": 2,
        },
        {
            "type": "vote_confidence",
            "threshold": 0.5,
            "minimum": 2.0,
            "max_samples": 2,
        },
        {"type": "vote_confidence", "threshold": 0.5, "max_samples": 2},
        {
            "type": "vote_confidence",
            "threshold": 0.5,
            "minimum": 2,
            "max_samples": 2,
            "extra": None,
        },
        {"type": "consecutive_agreement", "streak": False, "max_samples": 2},
        {"type": "consecutive_agreement", "streak": 2, "max_samples": 2, "x": 1},
    ],
)
def test_strategy_from_spec_rejects_wrong_shape_keys_and_types(spec: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        strategy_from_spec(spec)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: FixedStrategy(samples=0, max_samples=2),
        lambda: FixedStrategy(samples=3, max_samples=2),
        lambda: FixedStrategy(samples=1, max_samples=0),
        lambda: FixedStrategy(samples=1, max_samples=MAX_SAMPLES + 1),
        lambda: VoteConfidenceStrategy(threshold=0.0, minimum=1, max_samples=2),
        lambda: VoteConfidenceStrategy(threshold=1.01, minimum=1, max_samples=2),
        lambda: VoteConfidenceStrategy(threshold=float("nan"), minimum=1, max_samples=2),
        lambda: VoteConfidenceStrategy(threshold=float("inf"), minimum=1, max_samples=2),
        lambda: VoteConfidenceStrategy(threshold=0.5, minimum=0, max_samples=2),
        lambda: VoteConfidenceStrategy(threshold=0.5, minimum=3, max_samples=2),
        lambda: ConsecutiveAgreementStrategy(streak=0, max_samples=2),
        lambda: ConsecutiveAgreementStrategy(streak=3, max_samples=2),
    ],
)
def test_constructor_rejects_invalid_resource_bounds(constructor) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_boundary_resource_limits_are_accepted() -> None:
    strategies = (
        FixedStrategy(samples=MAX_SAMPLES, max_samples=MAX_SAMPLES),
        VoteConfidenceStrategy(threshold=1.0, minimum=MAX_SAMPLES, max_samples=MAX_SAMPLES),
        ConsecutiveAgreementStrategy(streak=MAX_SAMPLES, max_samples=MAX_SAMPLES),
    )

    assert [strategy.max_samples for strategy in strategies] == [MAX_SAMPLES] * 3
    assert all(strategy_from_spec(strategy.to_spec()) == strategy for strategy in strategies)


@pytest.mark.parametrize(
    ("strategy", "attribute", "value"),
    [
        (FixedStrategy(samples=2, max_samples=4), "samples", 3),
        (VoteConfidenceStrategy(threshold=0.5, minimum=2, max_samples=4), "threshold", 0.8),
        (ConsecutiveAgreementStrategy(streak=2, max_samples=4), "max_samples", 8),
    ],
)
def test_strategies_are_immutable(
    strategy: StoppingStrategy, attribute: str, value: object
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(strategy, attribute, value)
