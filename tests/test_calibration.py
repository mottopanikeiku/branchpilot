from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from branchpilot.calibration import load_operating_point, select_operating_point
from branchpilot.evaluate import Interval


def _row(
    policy: str,
    *,
    cost: float,
    accuracy: float,
    samples: float,
    samples_lower: float | None = None,
    samples_upper: float | None = None,
    tokens: float = 120.0,
    family: str = "offline-rl",
) -> dict[str, object]:
    return {
        "family": family,
        "policy": policy,
        "scoring_cost": cost,
        "accuracy": accuracy,
        "accuracy_interval": {
            "lower": max(0.0, accuracy - 0.05),
            "upper": min(1.0, accuracy + 0.05),
        },
        "average_samples": samples,
        "average_samples_interval": {
            "lower": samples if samples_lower is None else samples_lower,
            "upper": samples if samples_upper is None else samples_upper,
        },
        "average_tokens": tokens,
        "average_tokens_interval": {"lower": tokens - 10.0, "upper": tokens + 10.0},
        "utility": accuracy - cost * (samples - 1.0),
    }


def test_conservative_selection_uses_upper_bound_instead_of_point_estimate() -> None:
    rows = [
        _row(
            "accurate-but-uncertain",
            cost=0.05,
            accuracy=0.9,
            samples=2.0,
            samples_lower=1.7,
            samples_upper=3.0,
        ),
        _row(
            "conservative-fit",
            cost=0.1,
            accuracy=0.8,
            samples=1.5,
            samples_lower=1.2,
            samples_upper=2.0,
        ),
        _row("fixed-baseline", cost=0.0, accuracy=1.0, samples=1.0, family="fixed"),
    ]

    conservative = select_operating_point(rows, 2.5)
    point_estimate = select_operating_point(rows, 2.5, conservative=False)

    assert conservative.policy == "conservative-fit"
    assert conservative.conservative is True
    assert point_estimate.policy == "accurate-but-uncertain"
    assert point_estimate.conservative is False
    assert conservative.budget_satisfied is True
    assert point_estimate.budget_satisfied is True


def test_accuracy_ties_prefer_fewer_samples_then_lower_cost() -> None:
    rows = [
        _row("more-samples", cost=0.01, accuracy=0.85, samples=2.0),
        _row("higher-cost", cost=0.1, accuracy=0.85, samples=1.5),
        _row("winner", cost=0.05, accuracy=0.85, samples=1.5),
    ]

    selected = select_operating_point(rows, 3.0)

    assert selected.policy == "winner"
    assert selected.cost == 0.05


def test_identical_operating_points_across_scoring_copies_are_deduplicated() -> None:
    rows = [
        _row("BranchPilot lambda=0.1", cost=0.1, accuracy=0.8, samples=2.0),
        _row("BranchPilot lambda=0.02", cost=0.02, accuracy=0.8, samples=2.0),
        _row("less-accurate", cost=0.01, accuracy=0.7, samples=1.5),
    ]

    selected = select_operating_point(rows, 2.0)

    assert selected.policy == "BranchPilot lambda=0.02"
    assert selected.cost == 0.02


def test_infeasible_budget_returns_minimum_compute_with_disclosure() -> None:
    rows = [
        _row(
            "accurate",
            cost=0.01,
            accuracy=0.95,
            samples=3.0,
            samples_lower=2.5,
            samples_upper=3.5,
        ),
        _row(
            "minimum-compute",
            cost=0.1,
            accuracy=0.7,
            samples=1.2,
            samples_lower=1.0,
            samples_upper=1.4,
        ),
    ]

    selected = select_operating_point(rows, 0.9)

    assert selected.policy == "minimum-compute"
    assert selected.requested_sample_budget == 0.9
    assert selected.budget_satisfied is False


def test_exact_confidence_boundary_is_accepted() -> None:
    rows = [
        _row(
            "on-boundary",
            cost=0.05,
            accuracy=0.9,
            samples=1.8,
            samples_lower=1.5,
            samples_upper=2.0,
        ),
        _row("lower-accuracy", cost=0.1, accuracy=0.8, samples=1.0),
    ]

    selected = select_operating_point(rows, 2.0)

    assert selected.policy == "on-boundary"
    assert selected.budget_satisfied is True


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan"), True, "2"])
def test_invalid_sample_budgets_are_rejected(budget: object) -> None:
    with pytest.raises(ValueError, match="sample_budget must be a finite positive number"):
        select_operating_point(
            [_row("learned", cost=0.1, accuracy=0.8, samples=1.0)],
            budget,  # type: ignore[arg-type]
        )


def test_missing_and_invalid_schema_fields_have_actionable_errors() -> None:
    missing = _row("learned", cost=0.1, accuracy=0.8, samples=1.0)
    del missing["average_tokens_interval"]
    with pytest.raises(ValueError, match="row 0.*average_tokens_interval"):
        select_operating_point([missing], 2.0)

    invalid_interval = _row("learned", cost=0.1, accuracy=0.8, samples=1.0)
    invalid_interval["average_samples_interval"] = {"lower": 2.0, "upper": 1.0}
    with pytest.raises(ValueError, match="average_samples_interval.*lower bound greater"):
        select_operating_point([invalid_interval], 2.0)

    with pytest.raises(ValueError, match="schema_version 2"):
        select_operating_point(
            {
                "schema_version": 1,
                "rows": [_row("learned", cost=0.1, accuracy=0.8, samples=1.0)],
            },
            2.0,
        )


def test_missing_empty_and_no_learned_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required field 'rows'"):
        select_operating_point({}, 2.0)
    with pytest.raises(ValueError, match="must not be empty"):
        select_operating_point([], 2.0)
    with pytest.raises(ValueError, match="no learned policies.*offline-rl"):
        select_operating_point(
            [_row("fixed-1", cost=0.1, accuracy=0.8, samples=1.0, family="fixed")],
            2.0,
        )


def test_operating_point_is_immutable() -> None:
    selected = select_operating_point(
        [_row("learned", cost=0.1, accuracy=0.8, samples=1.0)],
        2.0,
    )

    with pytest.raises(FrozenInstanceError):
        selected.policy = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        selected.samples_interval.upper = 9.0  # type: ignore[misc]


def test_to_dict_is_json_serializable_and_loader_matches_selector(tmp_path) -> None:
    row = _row(
        "learned",
        cost=0.05,
        accuracy=0.75,
        samples=1.5,
        samples_lower=1.25,
        samples_upper=1.75,
        tokens=100.0,
    )
    payload = {"schema_version": 2, "rows": [row]}
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    selected = load_operating_point(benchmark_path, 1.75)
    serialized = selected.to_dict()

    assert serialized == {
        "policy": "learned",
        "cost": 0.05,
        "expected_accuracy": 0.75,
        "accuracy_interval": {"lower": 0.7, "upper": 0.8},
        "expected_samples": 1.5,
        "samples_interval": {"lower": 1.25, "upper": 1.75},
        "expected_tokens": 100.0,
        "tokens_interval": {"lower": 90.0, "upper": 110.0},
        "requested_sample_budget": 1.75,
        "conservative": True,
        "budget_satisfied": True,
    }
    assert json.loads(json.dumps(serialized)) == serialized
    assert selected.accuracy_interval == Interval(0.7, 0.8)
