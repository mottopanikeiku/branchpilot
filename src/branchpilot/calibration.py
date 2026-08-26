from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any

from branchpilot.evaluate import Interval

_SCHEMA_VERSION = 2
_LEARNED_FAMILY = "offline-rl"


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A validation-measured operating point selected for a sample budget."""

    policy: str
    cost: float
    expected_accuracy: float
    accuracy_interval: Interval
    expected_samples: float
    samples_interval: Interval
    expected_tokens: float
    tokens_interval: Interval
    requested_sample_budget: float
    conservative: bool
    budget_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "cost": self.cost,
            "expected_accuracy": self.expected_accuracy,
            "accuracy_interval": self.accuracy_interval.to_dict(),
            "expected_samples": self.expected_samples,
            "samples_interval": self.samples_interval.to_dict(),
            "expected_tokens": self.expected_tokens,
            "tokens_interval": self.tokens_interval.to_dict(),
            "requested_sample_budget": self.requested_sample_budget,
            "conservative": self.conservative,
            "budget_satisfied": self.budget_satisfied,
        }


def _finite_number(
    row: Mapping[str, Any],
    field: str,
    row_index: int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if field not in row:
        raise ValueError(f"benchmark row {row_index} is missing required field {field!r}")
    raw = row[field]
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ValueError(f"benchmark row {row_index} field {field!r} must be a finite number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"benchmark row {row_index} field {field!r} must be a finite number")
    if minimum is not None:
        below_minimum = value < minimum if minimum_inclusive else value <= minimum
        if below_minimum:
            comparison = "at least" if minimum_inclusive else "greater than"
            raise ValueError(
                f"benchmark row {row_index} field {field!r} must be {comparison} {minimum:g}"
            )
    if maximum is not None and value > maximum:
        raise ValueError(f"benchmark row {row_index} field {field!r} must be at most {maximum:g}")
    return value


def _interval(
    row: Mapping[str, Any],
    field: str,
    row_index: int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Interval:
    if field not in row:
        raise ValueError(f"benchmark row {row_index} is missing required field {field!r}")
    raw = row[field]
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"benchmark row {row_index} field {field!r} must contain "
            "numeric 'lower' and 'upper' bounds"
        )
    bounds: dict[str, float] = {}
    for bound in ("lower", "upper"):
        if bound not in raw:
            raise ValueError(
                f"benchmark row {row_index} field {field!r} is missing {bound!r} bound"
            )
        value = raw[bound]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"benchmark row {row_index} field {field!r} {bound!r} bound must be a finite number"
            )
        bounds[bound] = float(value)
        if not math.isfinite(bounds[bound]):
            raise ValueError(
                f"benchmark row {row_index} field {field!r} {bound!r} bound must be a finite number"
            )
    lower = bounds["lower"]
    upper = bounds["upper"]
    if lower > upper:
        raise ValueError(
            f"benchmark row {row_index} field {field!r} has lower bound greater than upper bound"
        )
    if minimum is not None and lower < minimum:
        raise ValueError(
            f"benchmark row {row_index} field {field!r} lower bound must be at least {minimum:g}"
        )
    if maximum is not None and upper > maximum:
        raise ValueError(
            f"benchmark row {row_index} field {field!r} upper bound must be at most {maximum:g}"
        )
    return Interval(lower=lower, upper=upper)


def _required_text(row: Mapping[str, Any], field: str, row_index: int) -> str:
    if field not in row:
        raise ValueError(f"benchmark row {row_index} is missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"benchmark row {row_index} field {field!r} must be a non-empty string")
    return value


def _benchmark_rows(payload_or_rows: Any) -> list[Mapping[str, Any]]:
    raw_rows = payload_or_rows
    if isinstance(payload_or_rows, Mapping):
        if "rows" not in payload_or_rows:
            raise ValueError("benchmark payload is missing required field 'rows'")
        if "schema_version" in payload_or_rows:
            version = payload_or_rows["schema_version"]
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version != _SCHEMA_VERSION
            ):
                raise ValueError(
                    f"benchmark payload must use schema_version {_SCHEMA_VERSION}; got {version!r}"
                )
        raw_rows = payload_or_rows["rows"]

    if isinstance(raw_rows, (str, bytes, Mapping)) or not isinstance(raw_rows, Iterable):
        raise ValueError("benchmark rows must be an iterable of row objects")

    rows = list(raw_rows)
    if not rows:
        raise ValueError("benchmark rows must not be empty")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"benchmark row {index} must be an object")
    return rows


def _learned_points(
    rows: list[Mapping[str, Any]],
    sample_budget: float,
    conservative: bool,
) -> list[OperatingPoint]:
    points: list[OperatingPoint] = []
    for index, row in enumerate(rows):
        family = _required_text(row, "family", index)
        if family != _LEARNED_FAMILY:
            continue
        policy = _required_text(row, "policy", index)
        cost = _finite_number(row, "scoring_cost", index, minimum=0.0)
        accuracy = _finite_number(row, "accuracy", index, minimum=0.0, maximum=1.0)
        accuracy_interval = _interval(row, "accuracy_interval", index, minimum=0.0, maximum=1.0)
        samples = _finite_number(
            row, "average_samples", index, minimum=0.0, minimum_inclusive=False
        )
        samples_interval = _interval(row, "average_samples_interval", index, minimum=0.0)
        tokens = _finite_number(row, "average_tokens", index, minimum=0.0)
        tokens_interval = _interval(row, "average_tokens_interval", index, minimum=0.0)
        _finite_number(row, "utility", index)

        points.append(
            OperatingPoint(
                policy=policy,
                cost=cost,
                expected_accuracy=accuracy,
                accuracy_interval=accuracy_interval,
                expected_samples=samples,
                samples_interval=samples_interval,
                expected_tokens=tokens,
                tokens_interval=tokens_interval,
                requested_sample_budget=sample_budget,
                conservative=conservative,
                budget_satisfied=False,
            )
        )

    if not points:
        raise ValueError("benchmark rows contain no learned policies with family 'offline-rl'")
    return points


def _deduplicate(points: list[OperatingPoint]) -> list[OperatingPoint]:
    unique: dict[tuple[float, ...], OperatingPoint] = {}
    for point in points:
        outcome = (
            point.expected_accuracy,
            point.accuracy_interval.lower,
            point.accuracy_interval.upper,
            point.expected_samples,
            point.samples_interval.lower,
            point.samples_interval.upper,
            point.expected_tokens,
            point.tokens_interval.lower,
            point.tokens_interval.upper,
        )
        previous = unique.get(outcome)
        if previous is None or (point.cost, point.policy) < (previous.cost, previous.policy):
            unique[outcome] = point
    return list(unique.values())


def select_operating_point(
    payload_or_rows: Any,
    sample_budget: float,
    *,
    conservative: bool = True,
) -> OperatingPoint:
    """Select the best validation-measured learned policy for a target average budget.

    In conservative mode, feasibility uses the upper confidence bound for average
    samples. Otherwise it uses the measured point estimate. If validation contains
    no feasible learned policy, the minimum-sample point is returned and explicitly
    marked as not satisfying the requested budget.
    """

    if isinstance(sample_budget, bool) or not isinstance(sample_budget, Real):
        raise ValueError("sample_budget must be a finite positive number")
    budget = float(sample_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("sample_budget must be a finite positive number")
    if not isinstance(conservative, bool):
        raise ValueError("conservative must be a boolean")

    rows = _benchmark_rows(payload_or_rows)
    points = _deduplicate(_learned_points(rows, budget, conservative))
    feasible = [
        point
        for point in points
        if (point.samples_interval.upper if conservative else point.expected_samples) <= budget
    ]

    if feasible:
        selected = min(
            feasible,
            key=lambda point: (
                -point.expected_accuracy,
                point.expected_samples,
                point.cost,
                point.policy,
            ),
        )
        return replace(selected, budget_satisfied=True)

    selected = min(
        points,
        key=lambda point: (
            point.expected_samples,
            -point.expected_accuracy,
            point.cost,
            point.policy,
        ),
    )
    return selected


def load_operating_point(
    benchmark_path: str | Path,
    sample_budget: float,
    *,
    conservative: bool = True,
) -> OperatingPoint:
    payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    return select_operating_point(payload, sample_budget, conservative=conservative)
