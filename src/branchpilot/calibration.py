from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from branchpilot.evaluate import Interval
from branchpilot.strategies import (
    ConsecutiveAgreementStrategy,
    FixedStrategy,
    VoteConfidenceStrategy,
)

_BENCHMARK_SCHEMA_VERSION = 2
_DEPLOYMENT_PLAN_SCHEMA_VERSION = 1
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


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """A deployable strategy selected from validation-measured candidates."""

    schema_version: int
    selection_source: Mapping[str, object]

    family: str
    policy: str
    strategy_spec: Mapping[str, Any]
    expected_accuracy: float
    accuracy_interval: Interval
    expected_samples: float
    samples_interval: Interval
    expected_tokens: float
    tokens_interval: Interval
    requested_sample_budget: float
    conservative: bool
    budget_satisfied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_source", MappingProxyType(dict(self.selection_source)))
        object.__setattr__(self, "strategy_spec", MappingProxyType(dict(self.strategy_spec)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_source": dict(self.selection_source),
            "family": self.family,
            "policy": self.policy,
            "strategy_spec": dict(self.strategy_spec),
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


def _benchmark_selection_source(payload: Any) -> Mapping[str, object]:
    """Identify the complete selection input without claiming its authenticity."""
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark payload must be canonicalizable as finite JSON") from exc
    return {
        "benchmark_schema_version": _BENCHMARK_SCHEMA_VERSION,
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
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
                or version != _BENCHMARK_SCHEMA_VERSION
            ):
                raise ValueError(
                    "benchmark payload must use schema_version "
                    f"{_BENCHMARK_SCHEMA_VERSION}; got {version!r}"
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


_DEPLOYABLE_FAMILIES = frozenset({_LEARNED_FAMILY, "fixed", "heuristic"})


def _deployment_budget(sample_budget: object, conservative: object) -> tuple[float, bool]:
    if isinstance(sample_budget, bool) or not isinstance(sample_budget, Real):
        raise ValueError("sample_budget must be a finite positive number")
    budget = float(sample_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("sample_budget must be a finite positive number")
    if not isinstance(conservative, bool):
        raise ValueError("conservative must be a boolean")
    return budget, conservative


def _deployment_families(families: Iterable[str] | None) -> frozenset[str]:
    if families is None:
        return _DEPLOYABLE_FAMILIES
    if isinstance(families, (str, bytes)) or not isinstance(families, Iterable):
        raise ValueError("families must be an iterable of deployable family names")
    try:
        selected = frozenset(families)
    except TypeError as error:
        raise ValueError("families must contain deployable family names") from error
    if not selected:
        raise ValueError("family filter contains no candidates")
    if any(not isinstance(family, str) or not family for family in selected):
        raise ValueError("families must contain non-empty strings")
    unsupported = sorted(selected - _DEPLOYABLE_FAMILIES)
    if unsupported:
        raise ValueError(f"nondeployable family in filter: {unsupported[0]!r}")
    return selected


def _deployment_payload(payload: Any) -> tuple[list[Mapping[str, Any]], int]:
    if not isinstance(payload, Mapping):
        raise ValueError("deployment selection requires a benchmark payload object")
    if payload.get("schema_version") != _BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"benchmark payload must use schema_version {_BENCHMARK_SCHEMA_VERSION}")
    data = payload.get("data")
    if not isinstance(data, Mapping) or data.get("split") != "validation":
        raise ValueError("deployment selection requires benchmark data split 'validation'")
    if "max_samples" not in payload:
        raise ValueError("benchmark payload is missing required field 'max_samples'")
    maximum = payload["max_samples"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("benchmark payload field 'max_samples' must be a positive integer")
    return _benchmark_rows(payload), maximum


def _deployment_interval(
    row: Mapping[str, Any],
    field: str,
    row_index: int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Interval:
    raw = row.get(field)
    if isinstance(raw, Mapping) and frozenset(raw) != frozenset({"lower", "upper"}):
        missing = sorted({"lower", "upper"} - set(raw))
        extra = sorted(repr(key) for key in set(raw) - {"lower", "upper"})
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"extra fields: {', '.join(extra)}")
        raise ValueError(
            f"benchmark row {row_index} field {field!r} is malformed ({'; '.join(details)})"
        )
    return _interval(row, field, row_index, minimum=minimum, maximum=maximum)


def _deployment_metrics(
    row: Mapping[str, Any],
    row_index: int,
    max_samples: int,
) -> tuple[float, Interval, float, Interval, float, Interval]:
    accuracy = _finite_number(row, "accuracy", row_index, minimum=0.0, maximum=1.0)
    accuracy_interval = _deployment_interval(
        row, "accuracy_interval", row_index, minimum=0.0, maximum=1.0
    )
    samples = _finite_number(
        row,
        "average_samples",
        row_index,
        minimum=1.0,
        maximum=float(max_samples),
    )
    samples_interval = _deployment_interval(
        row,
        "average_samples_interval",
        row_index,
        minimum=1.0,
        maximum=float(max_samples),
    )
    tokens = _finite_number(row, "average_tokens", row_index, minimum=0.0)
    tokens_interval = _deployment_interval(row, "average_tokens_interval", row_index, minimum=0.0)
    _finite_number(row, "utility", row_index)
    return accuracy, accuracy_interval, samples, samples_interval, tokens, tokens_interval


def _policy_binding(payload: Mapping[str, Any]) -> tuple[str, str]:
    binding = payload.get("policy")
    if not isinstance(binding, Mapping):
        raise ValueError("benchmark payload is missing deployable policy binding")
    artifact = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(artifact, str) or not artifact.strip():
        raise ValueError("benchmark payload policy binding requires a non-empty 'path'")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        raise ValueError(
            "benchmark payload policy binding requires a 64-character hexadecimal 'sha256'"
        )
    return artifact, digest


def _deployment_spec(
    payload: Mapping[str, Any],
    family: str,
    policy: str,
    cost: float,
    max_samples: int,
    row_index: int,
) -> Mapping[str, Any]:
    if family == _LEARNED_FAMILY:
        expected = f"BranchPilot λ={cost!r}"
        if policy != expected:
            raise ValueError(
                f"benchmark row {row_index} has unknown offline-rl policy name {policy!r}; "
                f"expected {expected!r}"
            )
        artifact, digest = _policy_binding(payload)
        return {
            "type": "learned",
            "cost": cost,
            "policy_artifact": artifact,
            "policy_sha256": digest,
        }

    if family == "fixed":
        prefix = "fixed-"
        count_text = policy.removeprefix(prefix)
        if (
            not policy.startswith(prefix)
            or not count_text.isascii()
            or not count_text.isdecimal()
            or count_text != str(int(count_text))
        ):
            raise ValueError(f"benchmark row {row_index} has unknown fixed policy name {policy!r}")
        try:
            return FixedStrategy(samples=int(count_text), max_samples=max_samples).to_spec()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"benchmark row {row_index} has invalid fixed policy name {policy!r}: {error}"
            ) from error

    if family == "heuristic" and policy.startswith("confidence-"):
        threshold_text = policy.removeprefix("confidence-")
        try:
            threshold = float(threshold_text)
        except ValueError as error:
            raise ValueError(
                f"benchmark row {row_index} has unknown heuristic policy name {policy!r}"
            ) from error
        if threshold_text != f"{threshold:g}":
            raise ValueError(
                f"benchmark row {row_index} has unknown heuristic policy name {policy!r}"
            )
        try:
            return VoteConfidenceStrategy(
                threshold=threshold, minimum=2, max_samples=max_samples
            ).to_spec()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"benchmark row {row_index} has invalid heuristic policy name {policy!r}: {error}"
            ) from error

    if family == "heuristic" and policy.startswith("agreement-"):
        streak_text = policy.removeprefix("agreement-")
        if (
            not streak_text.isascii()
            or not streak_text.isdecimal()
            or streak_text != str(int(streak_text))
        ):
            raise ValueError(
                f"benchmark row {row_index} has unknown heuristic policy name {policy!r}"
            )
        try:
            return ConsecutiveAgreementStrategy(
                streak=int(streak_text), max_samples=max_samples
            ).to_spec()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"benchmark row {row_index} has invalid heuristic policy name {policy!r}: {error}"
            ) from error

    raise ValueError(f"benchmark row {row_index} has unknown {family} policy name {policy!r}")


def _deployment_plans(
    payload: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    max_samples: int,
    budget: float,
    conservative: bool,
    families: frozenset[str],
    selection_source: Mapping[str, object],
) -> list[DeploymentPlan]:
    plans: list[DeploymentPlan] = []
    for index, row in enumerate(rows):
        family = _required_text(row, "family", index)
        if family not in _DEPLOYABLE_FAMILIES:
            raise ValueError(f"benchmark row {index} uses nondeployable family {family!r}")
        if family not in families:
            continue
        policy = _required_text(row, "policy", index)
        cost = _finite_number(row, "scoring_cost", index, minimum=0.0)
        strategy_spec = _deployment_spec(payload, family, policy, cost, max_samples, index)
        (
            accuracy,
            accuracy_interval,
            samples,
            samples_interval,
            tokens,
            tokens_interval,
        ) = _deployment_metrics(row, index, max_samples)
        plans.append(
            DeploymentPlan(
                schema_version=_DEPLOYMENT_PLAN_SCHEMA_VERSION,
                selection_source=selection_source,
                family=family,
                policy=policy,
                strategy_spec=strategy_spec,
                expected_accuracy=accuracy,
                accuracy_interval=accuracy_interval,
                expected_samples=samples,
                samples_interval=samples_interval,
                expected_tokens=tokens,
                tokens_interval=tokens_interval,
                requested_sample_budget=budget,
                conservative=conservative,
                budget_satisfied=False,
            )
        )
    if not plans:
        names = ", ".join(sorted(families))
        raise ValueError(f"family filter contains no candidates for: {names}")
    return plans


def _deduplicate_deployment_plans(plans: list[DeploymentPlan]) -> list[DeploymentPlan]:
    unique: dict[tuple[Any, ...], DeploymentPlan] = {}
    for plan in plans:
        observable = (
            tuple(sorted(plan.strategy_spec.items())),
            plan.expected_accuracy,
            plan.accuracy_interval.lower,
            plan.accuracy_interval.upper,
            plan.expected_samples,
            plan.samples_interval.lower,
            plan.samples_interval.upper,
            plan.expected_tokens,
            plan.tokens_interval.lower,
            plan.tokens_interval.upper,
        )
        previous = unique.get(observable)
        if previous is None or (plan.family, plan.policy) < (
            previous.family,
            previous.policy,
        ):
            unique[observable] = plan
    return list(unique.values())


def select_deployment_plan(
    payload: Any,
    sample_budget: float,
    *,
    conservative: bool = True,
    families: Iterable[str] | None = None,
) -> DeploymentPlan:
    """Select the strongest deployable point within an average sample budget."""

    budget, use_upper_bound = _deployment_budget(sample_budget, conservative)
    selected_families = _deployment_families(families)
    rows, max_samples = _deployment_payload(payload)
    selection_source = _benchmark_selection_source(payload)
    plans = _deduplicate_deployment_plans(
        _deployment_plans(
            payload,
            rows,
            max_samples,
            budget,
            use_upper_bound,
            selected_families,
            selection_source,
        )
    )
    feasible = [
        plan
        for plan in plans
        if (plan.samples_interval.upper if use_upper_bound else plan.expected_samples) <= budget
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda plan: (
                -plan.expected_accuracy,
                plan.expected_samples,
                plan.family,
                plan.policy,
            ),
        )
        return replace(selected, budget_satisfied=True)
    return min(
        plans,
        key=lambda plan: (
            plan.expected_samples,
            -plan.expected_accuracy,
            plan.family,
            plan.policy,
        ),
    )


def load_deployment_plan(
    benchmark_path: str | Path,
    sample_budget: float,
    *,
    conservative: bool = True,
    families: Iterable[str] | None = None,
) -> DeploymentPlan:
    payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    return select_deployment_plan(
        payload,
        sample_budget,
        conservative=conservative,
        families=families,
    )
