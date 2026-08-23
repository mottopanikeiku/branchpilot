from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from statistics import mean

import numpy as np

from branchpilot.features import prefix_state
from branchpilot.policy import BranchPilotPolicy, Decision
from branchpilot.schema import Rollout

StopRule = Callable[[Rollout], int]


@dataclass(frozen=True, slots=True)
class Metrics:
    policy: str
    family: str
    scoring_cost: float
    accuracy: float
    average_samples: float
    average_tokens: float
    p50_samples: float
    p90_samples: float
    utility: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def _measure(
    rollouts: list[Rollout],
    rule: StopRule,
    name: str,
    family: str,
    scoring_cost: float,
    max_samples: int | None = None,
) -> Metrics:
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
    accuracy = mean(correct)
    average_samples = mean(counts)
    return Metrics(
        policy=name,
        family=family,
        scoring_cost=scoring_cost,
        accuracy=accuracy,
        average_samples=average_samples,
        average_tokens=mean(tokens),
        p50_samples=float(np.quantile(counts, 0.5)),
        p90_samples=float(np.quantile(counts, 0.9)),
        utility=accuracy - scoring_cost * average_samples,
    )


def fixed_rule(samples: int) -> StopRule:
    return lambda rollout: min(samples, len(rollout.samples))


def confidence_rule(threshold: float, minimum: int = 2) -> StopRule:
    def stop(rollout: Rollout) -> int:
        for count in range(minimum, len(rollout.samples) + 1):
            state = prefix_state(rollout, count, len(rollout.samples))
            if state.top_votes / count >= threshold:
                return count
        return len(rollout.samples)

    return stop


def agreement_rule(streak: int) -> StopRule:
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


def benchmark(
    rollouts: list[Rollout],
    policy: BranchPilotPolicy,
    costs: Iterable[float] = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15),
) -> list[Metrics]:
    if not rollouts:
        raise ValueError("benchmark requires at least one rollout")
    costs = tuple(costs)
    if not costs or any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("benchmark costs must be a non-empty finite non-negative sequence")
    rows: list[Metrics] = []
    horizon = min(policy.max_samples, min(len(record.samples) for record in rollouts))
    fixed_counts = sorted(count for count in {1, 2, 4, horizon} if count <= horizon)
    for cost in costs:
        rows.append(
            _measure(
                rollouts,
                learned_rule(policy, cost),
                f"BranchPilot λ={cost:g}",
                "offline-rl",
                cost,
                horizon,
            )
        )
        for count in fixed_counts:
            rows.append(
                _measure(
                    rollouts,
                    fixed_rule(count),
                    f"fixed-{count}",
                    "fixed",
                    cost,
                    horizon,
                )
            )
        for threshold in (0.67, 0.8, 1.0):
            rows.append(
                _measure(
                    rollouts,
                    confidence_rule(threshold),
                    f"confidence-{threshold:g}",
                    "heuristic",
                    cost,
                    horizon,
                )
            )
        for streak in (2, 3):
            rows.append(
                _measure(
                    rollouts,
                    agreement_rule(streak),
                    f"agreement-{streak}",
                    "heuristic",
                    cost,
                    horizon,
                )
            )
    return rows


def decision_trace(
    rollout: Rollout, policy: BranchPilotPolicy, cost: float
) -> list[Decision]:
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
            and (
                other.accuracy > row.accuracy
                or other.average_samples < row.average_samples
            )
            for other in candidates
        )
    ]
    return sorted(frontier, key=lambda row: (row.average_samples, row.accuracy))
