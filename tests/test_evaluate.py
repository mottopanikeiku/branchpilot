from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from branchpilot.evaluate import (
    BenchmarkResult,
    Interval,
    _paired_interval,
    benchmark,
    decision_trace,
)
from branchpilot.policy import Decision
from branchpilot.schema import Rollout, Sample


def _rollout(
    uid: str,
    answers: list[str | None],
    *,
    gold: str = "A",
    tokens: int = 10,
) -> Rollout:
    return Rollout(
        uid=uid,
        question=f"question {uid}",
        gold=gold,
        samples=tuple(
            Sample(text=f"reasoning {index}", answer=answer, token_count=tokens)
            for index, answer in enumerate(answers)
        ),
    )


@dataclass
class _StoppingPolicy:
    max_samples: int = 8
    stop_at: int = 2

    def run(self, rollout: Rollout, cost: float) -> Decision:
        count = min(self.stop_at, self.max_samples, len(rollout.samples))
        answer = rollout.samples[count - 1].answer
        return Decision("stop", 1.0, 0.0, count, answer)

    def decide(self, rollout: Rollout, count: int, cost: float) -> Decision:
        stop = count >= min(self.stop_at, self.max_samples, len(rollout.samples))
        return Decision(
            "stop" if stop else "continue",
            float(stop),
            float(not stop),
            count,
            rollout.samples[count - 1].answer,
        )


def _row(result: BenchmarkResult, name: str):
    return next(row for row in result.rows if row.policy == name)


def test_benchmark_covers_every_fixed_count_through_horizon() -> None:
    result = benchmark(
        [_rollout("long", ["A"] * 8)],
        _StoppingPolicy(max_samples=8),
        costs=(0.1,),
        bootstrap_samples=20,
        bootstrap_seed=4,
    )

    fixed_names = {row.policy for row in result if row.family == "fixed"}
    assert fixed_names == {f"fixed-{count}" for count in range(1, 9)}
    assert {"fixed-3", "fixed-5", "fixed-6", "fixed-7"} <= fixed_names
    assert result.max_samples == 8


def test_heterogeneous_rollouts_do_not_globally_truncate_long_trajectories() -> None:
    rollouts = [
        _rollout("short", ["A", "A"]),
        _rollout("long", ["B", "B", "B", "A", "A", "A", "A", "A"]),
    ]
    result = benchmark(
        rollouts,
        _StoppingPolicy(max_samples=8),
        costs=(0.1,),
        bootstrap_samples=20,
        bootstrap_seed=5,
    )

    fixed_two = _row(result, "fixed-2")
    fixed_eight = _row(result, "fixed-8")
    assert fixed_two.accuracy == 0.5
    assert fixed_eight.accuracy == 1.0
    assert fixed_eight.average_samples == 5.0
    assert fixed_eight.stop_histogram == (0, 1, 0, 0, 0, 0, 0, 1)


def test_bootstrap_output_is_deterministic() -> None:
    rollouts = [
        _rollout("one", ["A", "A", "B", "A"]),
        _rollout("two", ["B", "A", "A", "A"]),
        _rollout("three", ["B", "B", "A", "B"], gold="B"),
    ]
    arguments = {
        "costs": (0.1,),
        "bootstrap_samples": 75,
        "bootstrap_seed": 1234,
    }

    first = benchmark(rollouts, _StoppingPolicy(max_samples=4), **arguments)
    second = benchmark(rollouts, _StoppingPolicy(max_samples=4), **arguments)

    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["bootstrap"] == {
        "resamples": 75,
        "seed": 1234,
        "confidence": 0.95,
    }


def test_utility_only_charges_for_samples_after_the_first() -> None:
    result = benchmark(
        [_rollout("one", ["A", "A", "A"])],
        _StoppingPolicy(max_samples=3),
        costs=(0.25,),
        bootstrap_samples=10,
    )

    fixed_one = _row(result, "fixed-1")
    fixed_three = _row(result, "fixed-3")
    assert fixed_one.utility == 1.0
    assert fixed_three.utility == 0.5
    assert fixed_three.utility == fixed_three.accuracy - 0.25 * (fixed_three.average_samples - 1)


def test_paired_interval_is_exactly_zero_for_identical_outcomes() -> None:
    outcomes = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    indices = np.random.default_rng(9).integers(0, len(outcomes), size=(50, len(outcomes)))

    assert _paired_interval(outcomes, outcomes.copy(), indices) == Interval(0.0, 0.0)


def test_comparator_selection_is_explicit_and_frozen_names_are_exact() -> None:
    rollouts = [
        _rollout("one", ["A", "A", "A"]),
        _rollout("two", ["B", "A", "A"]),
    ]
    exploratory = benchmark(
        rollouts,
        _StoppingPolicy(max_samples=3, stop_at=1),
        costs=(0.1,),
        bootstrap_samples=20,
    )
    frozen = benchmark(
        rollouts,
        _StoppingPolicy(max_samples=3, stop_at=1),
        costs=(0.1,),
        bootstrap_samples=20,
        frozen_baselines={0.1: "fixed-1"},
    )

    assert exploratory.comparisons[0].selection == "observed-best (exploratory)"
    assert frozen.comparisons[0].selection == "validation-frozen"
    assert frozen.comparisons[0].baseline_policy == "fixed-1"
    assert len(frozen.outcomes) == len(rollouts)
    assert {outcome.uid for outcome in frozen.outcomes} == {"one", "two"}
    assert all(outcome.selection == "validation-frozen" for outcome in frozen.outcomes)
    assert len(frozen.to_dict()["outcomes"]) == len(rollouts)
    with pytest.raises(ValueError, match="not an exact baseline name"):
        benchmark(
            rollouts,
            _StoppingPolicy(max_samples=3),
            costs=(0.1,),
            bootstrap_samples=5,
            frozen_baselines={0.1: "Fixed-1"},
        )


def test_every_stop_histogram_accounts_for_every_prompt() -> None:
    result = benchmark(
        [
            _rollout("short", ["A", "A"]),
            _rollout("medium", ["A", "B", "A", "A"]),
            _rollout("long", ["B", "A", "A", "A", "A", "A"]),
        ],
        _StoppingPolicy(max_samples=6, stop_at=3),
        costs=(0.1,),
        bootstrap_samples=20,
    )

    assert all(len(row.stop_histogram) == 6 for row in result.rows)
    assert all(sum(row.stop_histogram) == result.records for row in result.rows)
    assert all(count >= 0 for row in result.rows for count in row.stop_histogram)


def test_decision_trace_still_ends_at_the_first_stop() -> None:
    rollout = _rollout("trace", ["A"] * 8)

    trace = decision_trace(
        rollout,
        _StoppingPolicy(max_samples=8, stop_at=3),
        cost=0.1,
    )

    assert [decision.sample_count for decision in trace] == [1, 2, 3]
    assert [decision.action for decision in trace] == ["continue", "continue", "stop"]
