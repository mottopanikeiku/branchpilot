from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from branchpilot.calibration import (
    load_deployment_plan,
    load_operating_point,
    select_deployment_plan,
    select_operating_point,
)
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


def _deployment_payload(*rows: dict[str, object], max_samples: int = 4) -> dict[str, object]:
    return {
        "schema_version": 2,
        "data": {"split": "validation"},
        "max_samples": max_samples,
        "policy": {
            "path": "policy.safetensors",
            "sha256": "a" * 64,
            "bytes": 123,
        },
        "rows": list(rows),
    }


def _payload_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def test_mixed_deployment_selection_can_choose_exact_heuristic_spec() -> None:
    payload = _deployment_payload(
        _row("BranchPilot λ=0.1", cost=0.1, accuracy=0.82, samples=1.8),
        _row("fixed-2", family="fixed", cost=0.1, accuracy=0.8, samples=2.0),
        _row(
            "confidence-0.75",
            family="heuristic",
            cost=0.1,
            accuracy=0.9,
            samples=1.7,
        ),
        _row(
            "agreement-2",
            family="heuristic",
            cost=0.1,
            accuracy=0.85,
            samples=1.6,
        ),
    )

    selected = select_deployment_plan(payload, 2.0)

    assert selected.family == "heuristic"
    assert selected.policy == "confidence-0.75"
    assert selected.strategy_spec == {
        "type": "vote_confidence",
        "threshold": 0.75,
        "minimum": 2,
        "max_samples": 4,
    }
    assert selected.budget_satisfied is True


def test_deployment_selection_source_is_stable_and_covers_complete_payload() -> None:
    payload = _deployment_payload(
        _row(
            "confidence-0.75",
            family="heuristic",
            cost=0.1,
            accuracy=0.9,
            samples=1.7,
        )
    )
    reordered = json.loads(
        json.dumps(payload),
        object_pairs_hook=lambda pairs: dict(reversed(pairs)),
    )
    changed = json.loads(json.dumps(payload))
    changed["policy"]["bytes"] = 124

    selected = select_deployment_plan(payload, 2.0)
    selected_reordered = select_deployment_plan(reordered, 2.0)
    selected_changed = select_deployment_plan(changed, 2.0)

    assert selected.selection_source == {
        "benchmark_schema_version": 2,
        "payload_sha256": _payload_sha256(payload),
    }
    assert selected_reordered.selection_source == selected.selection_source
    assert (
        selected_changed.selection_source["payload_sha256"]
        != selected.selection_source["payload_sha256"]
    )
    assert (
        selected.family,
        selected.policy,
        selected.strategy_spec,
    ) == (
        selected_changed.family,
        selected_changed.policy,
        selected_changed.strategy_spec,
    )
    with pytest.raises(TypeError):
        selected.selection_source["payload_sha256"] = "0" * 64  # type: ignore[index]


def test_mixed_deployment_selection_can_choose_bound_learned_policy() -> None:
    payload = _deployment_payload(
        _row("BranchPilot λ=0.05", cost=0.05, accuracy=0.94, samples=1.9),
        _row("fixed-1", family="fixed", cost=0.05, accuracy=0.7, samples=1.0),
        _row(
            "agreement-2",
            family="heuristic",
            cost=0.05,
            accuracy=0.85,
            samples=1.5,
        ),
    )

    selected = select_deployment_plan(payload, 2.0)

    assert selected.family == "offline-rl"
    assert selected.strategy_spec == {
        "type": "learned",
        "cost": 0.05,
        "policy_artifact": "policy.safetensors",
        "policy_sha256": "a" * 64,
    }


def test_deployment_conservative_and_point_selection_differ() -> None:
    uncertain = _row(
        "BranchPilot λ=0.1",
        cost=0.1,
        accuracy=0.95,
        samples=1.8,
        samples_lower=1.5,
        samples_upper=2.6,
    )
    safe = _row(
        "fixed-1",
        family="fixed",
        cost=0.1,
        accuracy=0.75,
        samples=1.0,
        samples_upper=1.2,
    )
    payload = _deployment_payload(uncertain, safe)

    conservative = select_deployment_plan(payload, 2.0)
    point = select_deployment_plan(payload, 2.0, conservative=False)

    assert conservative.policy == "fixed-1"
    assert point.policy == "BranchPilot λ=0.1"
    assert conservative.conservative is True
    assert point.conservative is False


def test_deployment_family_filters_and_no_candidate_disclosure() -> None:
    payload = _deployment_payload(
        _row("BranchPilot λ=0.1", cost=0.1, accuracy=0.95, samples=2.0),
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.7, samples=1.0),
    )

    fixed = select_deployment_plan(payload, 3.0, families={"fixed"})

    assert fixed.family == "fixed"
    assert fixed.strategy_spec == {"type": "fixed", "samples": 1, "max_samples": 4}
    with pytest.raises(ValueError, match="family filter contains no candidates"):
        select_deployment_plan(payload, 3.0, families={"heuristic"})
    with pytest.raises(ValueError, match="nondeployable family"):
        select_deployment_plan(payload, 3.0, families={"experimental"})


def test_cost_repeated_deterministic_rows_deduplicate_by_spec_and_observation() -> None:
    first = _row("fixed-2", family="fixed", cost=0.01, accuracy=0.8, samples=2.0)
    repeated = _row("fixed-2", family="fixed", cost=0.2, accuracy=0.8, samples=2.0)
    payload = _deployment_payload(first, repeated)

    selected = select_deployment_plan(payload, 2.0)

    assert selected.policy == "fixed-2"
    assert selected.strategy_spec == {"type": "fixed", "samples": 2, "max_samples": 4}


def test_deployment_exact_specs_round_trip_json_and_loader(tmp_path) -> None:
    row = _row(
        "agreement-3",
        family="heuristic",
        cost=0.2,
        accuracy=0.86,
        samples=2.2,
        samples_lower=2.0,
        samples_upper=2.5,
        tokens=150.0,
    )
    payload = _deployment_payload(row)
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    selected = load_deployment_plan(path, 2.5)
    serialized = selected.to_dict()

    assert serialized == {
        "schema_version": 1,
        "selection_source": {
            "benchmark_schema_version": 2,
            "payload_sha256": _payload_sha256(payload),
        },
        "family": "heuristic",
        "policy": "agreement-3",
        "strategy_spec": {
            "type": "consecutive_agreement",
            "streak": 3,
            "max_samples": 4,
        },
        "expected_accuracy": 0.86,
        "accuracy_interval": {"lower": 0.8099999999999999, "upper": 0.91},
        "expected_samples": 2.2,
        "samples_interval": {"lower": 2.0, "upper": 2.5},
        "expected_tokens": 150.0,
        "tokens_interval": {"lower": 140.0, "upper": 160.0},
        "requested_sample_budget": 2.5,
        "conservative": True,
        "budget_satisfied": True,
    }
    assert json.loads(json.dumps(serialized)) == serialized
    with pytest.raises(TypeError):
        selected.strategy_spec["streak"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        selected.selection_source["payload_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        selected.policy = "changed"  # type: ignore[misc]


def test_deployment_infeasible_returns_minimum_compute_with_disclosure() -> None:
    payload = _deployment_payload(
        _row("BranchPilot λ=0.1", cost=0.1, accuracy=0.95, samples=2.5),
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.7, samples=1.0),
    )

    selected = select_deployment_plan(payload, 0.5)

    assert selected.policy == "fixed-1"
    assert selected.requested_sample_budget == 0.5
    assert selected.budget_satisfied is False


def test_deployment_ties_use_family_then_policy_order() -> None:
    payload = _deployment_payload(
        _row("confidence-0.75", family="heuristic", cost=0.1, accuracy=0.8, samples=1.5),
        _row("fixed-2", family="fixed", cost=0.1, accuracy=0.8, samples=1.5),
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.8, samples=1.5),
    )

    selected = select_deployment_plan(payload, 2.0)

    assert selected.family == "fixed"
    assert selected.policy == "fixed-1"


@pytest.mark.parametrize(
    ("family", "policy"),
    [
        ("offline-rl", "BranchPilot lambda=0.1"),
        ("offline-rl", "BranchPilot λ=0.10"),
        ("fixed", "fixed-01"),
        ("fixed", "fixed-two"),
        ("heuristic", "confidence-0.750"),
        ("heuristic", "agreement-02"),
        ("heuristic", "other-2"),
    ],
)
def test_deployment_rejects_unknown_policy_naming(family: str, policy: str) -> None:
    payload = _deployment_payload(_row(policy, family=family, cost=0.1, accuracy=0.8, samples=1.5))

    with pytest.raises(ValueError, match="policy name"):
        select_deployment_plan(payload, 2.0)


def test_deployment_requires_explicit_validation_split() -> None:
    payload = _deployment_payload(
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.8, samples=1.0)
    )
    payload["data"] = {"split": "test"}
    with pytest.raises(ValueError, match="data split 'validation'"):
        select_deployment_plan(payload, 2.0)

    del payload["data"]
    with pytest.raises(ValueError, match="data split 'validation'"):
        select_deployment_plan(payload, 2.0)


def test_deployment_rejects_nondeployable_rows_and_malformed_payload() -> None:
    nondeployable = _deployment_payload(
        _row("oracle", family="oracle", cost=0.1, accuracy=1.0, samples=1.0)
    )
    with pytest.raises(ValueError, match="nondeployable family"):
        select_deployment_plan(nondeployable, 2.0)

    missing_max = _deployment_payload(
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.8, samples=1.0)
    )
    del missing_max["max_samples"]
    with pytest.raises(ValueError, match="max_samples"):
        select_deployment_plan(missing_max, 2.0)

    extra_interval = _deployment_payload(
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.8, samples=1.0)
    )
    row = extra_interval["rows"][0]  # type: ignore[index]
    row["accuracy_interval"]["extra"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="accuracy_interval.*extra fields"):
        select_deployment_plan(extra_interval, 2.0)


@pytest.mark.parametrize(
    "binding",
    [
        None,
        {},
        {"path": "", "sha256": "a" * 64},
        {"path": "policy.safetensors", "sha256": "bad"},
    ],
)
def test_learned_deployment_requires_valid_policy_binding(binding: object) -> None:
    payload = _deployment_payload(_row("BranchPilot λ=0.1", cost=0.1, accuracy=0.8, samples=1.5))
    payload["policy"] = binding

    with pytest.raises(ValueError, match="policy binding"):
        select_deployment_plan(payload, 2.0)


def test_old_learned_only_api_still_ignores_stronger_baselines() -> None:
    rows = [
        _row("legacy-learned", cost=0.1, accuracy=0.75, samples=1.5),
        _row("fixed-1", family="fixed", cost=0.1, accuracy=0.99, samples=1.0),
    ]

    selected = select_operating_point(rows, 2.0)

    assert selected.policy == "legacy-learned"
