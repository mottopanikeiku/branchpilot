from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from branchpilot.features import prefix_state
from branchpilot.policy import BranchPilotPolicy, Decision
from branchpilot.schema import Rollout

StopRule = Callable[[Rollout], int]
BOOTSTRAP_CONFIDENCE = 0.95
CONFIDENCE_THRESHOLDS = (
    0.5,
    0.55,
    0.6,
    0.65,
    0.67,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
)
AGREEMENT_STREAKS = (2, 3)


@dataclass(frozen=True, slots=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("interval bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower bound cannot exceed its upper bound")

    def to_dict(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class Metrics:
    policy: str
    family: str
    scoring_cost: float
    accuracy: float
    accuracy_interval: Interval
    average_samples: float
    average_samples_interval: Interval
    average_tokens: float
    average_tokens_interval: Interval
    p50_samples: float
    p90_samples: float
    utility: float
    utility_interval: Interval
    stop_histogram: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "family": self.family,
            "scoring_cost": self.scoring_cost,
            "accuracy": self.accuracy,
            "accuracy_interval": self.accuracy_interval.to_dict(),
            "average_samples": self.average_samples,
            "average_samples_interval": self.average_samples_interval.to_dict(),
            "average_tokens": self.average_tokens,
            "average_tokens_interval": self.average_tokens_interval.to_dict(),
            "p50_samples": self.p50_samples,
            "p90_samples": self.p90_samples,
            "utility": self.utility,
            "utility_interval": self.utility_interval.to_dict(),
            "stop_histogram": list(self.stop_histogram),
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    scoring_cost: float
    learned_policy: str
    baseline_policy: str
    selection: str
    accuracy_delta: float
    accuracy_delta_interval: Interval
    average_samples_delta: float
    average_samples_delta_interval: Interval
    average_tokens_delta: float
    average_tokens_delta_interval: Interval
    utility_delta: float
    utility_delta_interval: Interval

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring_cost": self.scoring_cost,
            "learned_policy": self.learned_policy,
            "baseline_policy": self.baseline_policy,
            "selection": self.selection,
            "accuracy_delta": self.accuracy_delta,
            "accuracy_delta_interval": self.accuracy_delta_interval.to_dict(),
            "average_samples_delta": self.average_samples_delta,
            "average_samples_delta_interval": self.average_samples_delta_interval.to_dict(),
            "average_tokens_delta": self.average_tokens_delta,
            "average_tokens_delta_interval": self.average_tokens_delta_interval.to_dict(),
            "utility_delta": self.utility_delta,
            "utility_delta_interval": self.utility_delta_interval.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    scoring_cost: float
    uid: str
    learned_policy: str
    learned_correct: bool
    learned_samples: int
    learned_tokens: int
    learned_utility: float
    baseline_policy: str
    baseline_correct: bool
    baseline_samples: int
    baseline_tokens: int
    baseline_utility: float
    selection: str

    def to_dict(self) -> dict[str, str | float | int | bool]:
        return {
            "scoring_cost": self.scoring_cost,
            "uid": self.uid,
            "learned_policy": self.learned_policy,
            "learned_correct": self.learned_correct,
            "learned_samples": self.learned_samples,
            "learned_tokens": self.learned_tokens,
            "learned_utility": self.learned_utility,
            "baseline_policy": self.baseline_policy,
            "baseline_correct": self.baseline_correct,
            "baseline_samples": self.baseline_samples,
            "baseline_tokens": self.baseline_tokens,
            "baseline_utility": self.baseline_utility,
            "selection": self.selection,
        }


@dataclass(frozen=True, slots=True)
class PolicyOutcomes:
    policy: str
    family: str
    uids: tuple[str, ...]
    correct: tuple[bool, ...]
    samples: tuple[int, ...]
    tokens: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "family": self.family,
            "uids": list(self.uids),
            "correct": list(self.correct),
            "samples": list(self.samples),
            "tokens": list(self.tokens),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult(Sequence[Metrics]):
    records: int
    max_samples: int
    costs: tuple[float, ...]
    rows: tuple[Metrics, ...]
    comparisons: tuple[Comparison, ...]
    outcomes: tuple[PairedOutcome, ...]
    bootstrap_samples: int
    policy_outcomes: tuple[PolicyOutcomes, ...]
    bootstrap_seed: int
    confidence: float = BOOTSTRAP_CONFIDENCE

    def __iter__(self) -> Iterator[Metrics]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    @overload
    def __getitem__(self, index: int) -> Metrics: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Metrics, ...]: ...

    def __getitem__(self, index: int | slice) -> Metrics | tuple[Metrics, ...]:
        return self.rows[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "max_samples": self.max_samples,
            "costs": list(self.costs),
            "rows": [row.to_dict() for row in self.rows],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "policy_outcomes": [outcome.to_dict() for outcome in self.policy_outcomes],
            "bootstrap": {
                "resamples": self.bootstrap_samples,
                "seed": self.bootstrap_seed,
                "confidence": self.confidence,
            },
        }


@dataclass(frozen=True, slots=True)
class _Measurement:
    metrics: Metrics
    correct: np.ndarray
    samples: np.ndarray
    tokens: np.ndarray
    utilities: np.ndarray


def _bootstrap_interval(values: np.ndarray, bootstrap_indices: np.ndarray) -> Interval:
    estimates = np.mean(values[bootstrap_indices], axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    lower, upper = np.quantile(estimates, (tail, 1.0 - tail))
    return Interval(float(lower), float(upper))


def _paired_interval(
    learned: np.ndarray, baseline: np.ndarray, bootstrap_indices: np.ndarray
) -> Interval:
    if learned.shape != baseline.shape:
        raise ValueError("paired outcomes must have the same shape")
    return _bootstrap_interval(learned - baseline, bootstrap_indices)


def _measure(
    rollouts: list[Rollout],
    rule: StopRule,
    name: str,
    family: str,
    scoring_cost: float,
    bootstrap_indices: np.ndarray,
    max_samples: int | None = None,
) -> _Measurement:
    correct: list[float] = []
    counts: list[int] = []
    tokens: list[int] = []
    for rollout in rollouts:
        horizon = min(len(rollout.samples), max_samples or len(rollout.samples))
        count = max(1, min(horizon, int(rule(rollout))))
        answer = prefix_state(rollout, count, horizon).majority_answer
        correct.append(float(answer == rollout.gold))
        counts.append(count)
        tokens.append(sum(sample.token_count for sample in rollout.samples[:count]))

    correct_values = np.asarray(correct, dtype=np.float64)
    count_values = np.asarray(counts, dtype=np.float64)
    token_values = np.asarray(tokens, dtype=np.float64)
    utilities = correct_values - scoring_cost * (count_values - 1.0)
    histogram_horizon = max(
        min(len(rollout.samples), max_samples or len(rollout.samples)) for rollout in rollouts
    )
    histogram_counts = np.bincount(count_values.astype(np.int64), minlength=histogram_horizon + 1)
    histogram = tuple(int(value) for value in histogram_counts[1:])
    metrics = Metrics(
        policy=name,
        family=family,
        scoring_cost=scoring_cost,
        accuracy=float(np.mean(correct_values)),
        accuracy_interval=_bootstrap_interval(correct_values, bootstrap_indices),
        average_samples=float(np.mean(count_values)),
        average_samples_interval=_bootstrap_interval(count_values, bootstrap_indices),
        average_tokens=float(np.mean(token_values)),
        average_tokens_interval=_bootstrap_interval(token_values, bootstrap_indices),
        p50_samples=float(np.quantile(count_values, 0.5)),
        p90_samples=float(np.quantile(count_values, 0.9)),
        utility=float(np.mean(utilities)),
        utility_interval=_bootstrap_interval(utilities, bootstrap_indices),
        stop_histogram=histogram,
    )
    return _Measurement(metrics, correct_values, count_values, token_values, utilities)


def fixed_rule(samples: int) -> StopRule:
    if samples < 1:
        raise ValueError("fixed sample count must be positive")
    return lambda rollout: min(samples, len(rollout.samples))


def confidence_rule(threshold: float, minimum: int = 2) -> StopRule:
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError("confidence threshold must be within (0, 1]")
    if minimum < 1:
        raise ValueError("confidence minimum must be positive")

    def stop(rollout: Rollout) -> int:
        for count in range(minimum, len(rollout.samples) + 1):
            state = prefix_state(rollout, count, len(rollout.samples))
            if state.top_votes / count >= threshold:
                return count
        return len(rollout.samples)

    return stop


def agreement_rule(streak: int) -> StopRule:
    if streak < 1:
        raise ValueError("agreement streak must be positive")

    def stop(rollout: Rollout) -> int:
        run = 0
        previous: str | None = None
        for index, sample in enumerate(rollout.samples, start=1):
            if sample.answer is not None and sample.answer == previous:
                run += 1
            else:
                run = 1
                previous = sample.answer
            if run >= streak:
                return index
        return len(rollout.samples)

    return stop


def learned_rule(policy: BranchPilotPolicy, cost: float) -> StopRule:
    return lambda rollout: policy.run(rollout, cost).sample_count


def _comparison(
    learned: _Measurement,
    baseline: _Measurement,
    selection: str,
    bootstrap_indices: np.ndarray,
) -> Comparison:
    return Comparison(
        scoring_cost=learned.metrics.scoring_cost,
        learned_policy=learned.metrics.policy,
        baseline_policy=baseline.metrics.policy,
        selection=selection,
        accuracy_delta=learned.metrics.accuracy - baseline.metrics.accuracy,
        accuracy_delta_interval=_paired_interval(
            learned.correct, baseline.correct, bootstrap_indices
        ),
        average_samples_delta=(learned.metrics.average_samples - baseline.metrics.average_samples),
        average_samples_delta_interval=_paired_interval(
            learned.samples, baseline.samples, bootstrap_indices
        ),
        average_tokens_delta=(learned.metrics.average_tokens - baseline.metrics.average_tokens),
        average_tokens_delta_interval=_paired_interval(
            learned.tokens, baseline.tokens, bootstrap_indices
        ),
        utility_delta=learned.metrics.utility - baseline.metrics.utility,
        utility_delta_interval=_paired_interval(
            learned.utilities, baseline.utilities, bootstrap_indices
        ),
    )


def benchmark(
    rollouts: list[Rollout],
    policy: BranchPilotPolicy,
    costs: Iterable[float] = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15),
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
    frozen_baselines: Mapping[float, str] | None = None,
) -> BenchmarkResult:
    if not rollouts:
        raise ValueError("benchmark requires at least one rollout")
    costs = tuple(float(cost) for cost in costs)
    if not costs or any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("benchmark costs must be a non-empty finite non-negative sequence")
    if len(set(costs)) != len(costs):
        raise ValueError("benchmark costs must be unique")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if frozen_baselines is not None and set(frozen_baselines) != set(costs):
        raise ValueError("frozen_baselines must contain exactly one name for every cost")

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        len(rollouts),
        size=(bootstrap_samples, len(rollouts)),
    )
    max_samples = min(policy.max_samples, max(len(record.samples) for record in rollouts))
    fixed_counts = range(1, max_samples + 1)
    rows: list[Metrics] = []
    comparisons: list[Comparison] = []
    outcomes: list[PairedOutcome] = []
    policy_outcomes: dict[str, PolicyOutcomes] = {}

    for cost in costs:
        learned = _measure(
            rollouts,
            learned_rule(policy, cost),
            f"BranchPilot λ={cost!r}",
            "offline-rl",
            cost,
            bootstrap_indices,
            policy.max_samples,
        )
        measured = [learned]
        for count in fixed_counts:
            measured.append(
                _measure(
                    rollouts,
                    fixed_rule(count),
                    f"fixed-{count}",
                    "fixed",
                    cost,
                    bootstrap_indices,
                    policy.max_samples,
                )
            )
        for threshold in CONFIDENCE_THRESHOLDS:
            measured.append(
                _measure(
                    rollouts,
                    confidence_rule(threshold),
                    f"confidence-{threshold:g}",
                    "heuristic",
                    cost,
                    bootstrap_indices,
                    policy.max_samples,
                )
            )
        for streak in AGREEMENT_STREAKS:
            measured.append(
                _measure(
                    rollouts,
                    agreement_rule(streak),
                    f"agreement-{streak}",
                    "heuristic",
                    cost,
                    bootstrap_indices,
                    policy.max_samples,
                )
            )

        baselines = measured[1:]
        if frozen_baselines is None:
            baseline = max(baselines, key=lambda item: item.metrics.utility)
            selection = "observed-best (exploratory)"
        else:
            baseline_name = frozen_baselines[cost]
            matching = [item for item in baselines if item.metrics.policy == baseline_name]
            if len(matching) != 1:
                available = ", ".join(item.metrics.policy for item in baselines)
                raise ValueError(
                    f"frozen baseline {baseline_name!r} is not an exact baseline name "
                    f"at cost {cost:g}; available: {available}"
                )
            baseline = matching[0]
            selection = "validation-frozen"

        rows.extend(item.metrics for item in measured)
        for measurement in measured:
            compact = PolicyOutcomes(
                policy=measurement.metrics.policy,
                family=measurement.metrics.family,
                uids=tuple(rollout.uid for rollout in rollouts),
                correct=tuple(bool(value) for value in measurement.correct),
                samples=tuple(int(value) for value in measurement.samples),
                tokens=tuple(int(value) for value in measurement.tokens),
            )
            previous = policy_outcomes.setdefault(compact.policy, compact)
            if previous != compact:
                raise RuntimeError(
                    f"policy {compact.policy!r} produced cost-dependent observable outcomes"
                )
        comparisons.append(_comparison(learned, baseline, selection, bootstrap_indices))
        for index, rollout in enumerate(rollouts):
            outcomes.append(
                PairedOutcome(
                    scoring_cost=cost,
                    uid=rollout.uid,
                    learned_policy=learned.metrics.policy,
                    learned_correct=bool(learned.correct[index]),
                    learned_samples=int(learned.samples[index]),
                    learned_tokens=int(learned.tokens[index]),
                    learned_utility=float(learned.utilities[index]),
                    baseline_policy=baseline.metrics.policy,
                    baseline_correct=bool(baseline.correct[index]),
                    baseline_samples=int(baseline.samples[index]),
                    baseline_tokens=int(baseline.tokens[index]),
                    baseline_utility=float(baseline.utilities[index]),
                    selection=selection,
                )
            )

    return BenchmarkResult(
        records=len(rollouts),
        max_samples=max_samples,
        costs=costs,
        rows=tuple(rows),
        comparisons=tuple(comparisons),
        outcomes=tuple(outcomes),
        policy_outcomes=tuple(policy_outcomes.values()),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def decision_trace(rollout: Rollout, policy: BranchPilotPolicy, cost: float) -> list[Decision]:
    trace: list[Decision] = []
    horizon = min(policy.max_samples, len(rollout.samples))
    for count in range(1, horizon + 1):
        decision = policy.decide(rollout, count, cost)
        trace.append(decision)
        if decision.action == "stop":
            break
    return trace


def pareto_frontier(rows: Iterable[Metrics]) -> list[Metrics]:
    unique: dict[tuple[str, float, float], Metrics] = {}
    for row in rows:
        key = (row.policy, row.accuracy, row.average_samples)
        unique[key] = row
    candidates = list(unique.values())
    frontier = [
        row
        for row in candidates
        if not any(
            other.accuracy >= row.accuracy
            and other.average_samples <= row.average_samples
            and (other.accuracy > row.accuracy or other.average_samples < row.average_samples)
            for other in candidates
        )
    ]
    return sorted(frontier, key=lambda row: (row.average_samples, row.accuracy))
