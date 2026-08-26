from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import branchpilot.deployment as deployment_module
from branchpilot.deployment import LoadedDeployment, load_deployment_plan, load_strategy_spec
from branchpilot.features import FEATURE_NAMES
from branchpilot.policy import BranchPilotPolicy
from branchpilot.schema import Sample
from branchpilot.strategies import (
    ConsecutiveAgreementStrategy,
    FixedStrategy,
    VoteConfidenceStrategy,
)


def _policy(*, costs: tuple[float, ...] = (0.1, 0.2), stop_bias: float = 1.0) -> BranchPilotPolicy:
    hidden_size = 2
    input_size = len(FEATURE_NAMES) + 3
    weights = {
        "input.weight": np.zeros((hidden_size, input_size), dtype=np.float32),
        "input.bias": np.zeros(hidden_size, dtype=np.float32),
        "norm.weight": np.ones(hidden_size, dtype=np.float32),
        "norm.bias": np.zeros(hidden_size, dtype=np.float32),
        "hidden.weight": np.zeros((hidden_size, hidden_size), dtype=np.float32),
        "hidden.bias": np.zeros(hidden_size, dtype=np.float32),
        "output.weight": np.zeros((2, hidden_size), dtype=np.float32),
        "output.bias": np.asarray((stop_bias, 0.0), dtype=np.float32),
    }
    return BranchPilotPolicy(
        weights,
        np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        np.ones(len(FEATURE_NAMES), dtype=np.float32),
        max_samples=4,
        hidden_size=hidden_size,
        costs=costs,
    )


def _learned_spec(path: Path, *, cost: float = 0.1, artifact: str | None = None) -> dict[str, object]:
    return {
        "type": "learned",
        "cost": cost,
        "policy_artifact": path.name if artifact is None else artifact,
        "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _plan_payload(spec: dict[str, object], family: str) -> dict[str, object]:
    return {
        "family": family,
        "policy": "selected-policy",
        "strategy_spec": spec,
        "expected_accuracy": 0.75,
        "accuracy_interval": {"lower": 0.7, "upper": 0.8},
        "expected_samples": 2.0,
        "samples_interval": {"lower": 1.5, "upper": 2.5},
        "expected_tokens": 40.0,
        "tokens_interval": {"lower": 30.0, "upper": 50.0},
        "requested_sample_budget": 3.0,
        "conservative": True,
        "budget_satisfied": True,
    }


def _write_plan(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("spec", "expected_type"),
    [
        ({"type": "fixed", "samples": 2, "max_samples": 4}, FixedStrategy),
        (
            {
                "type": "vote_confidence",
                "threshold": 0.75,
                "minimum": 2,
                "max_samples": 4,
            },
            VoteConfidenceStrategy,
        ),
        (
            {"type": "consecutive_agreement", "streak": 2, "max_samples": 4},
            ConsecutiveAgreementStrategy,
        ),
    ],
)
def test_loads_every_builtin_with_zero_cost(
    spec: dict[str, object], expected_type: type[object]
) -> None:
    loaded = load_strategy_spec(spec)

    assert isinstance(loaded.strategy, expected_type)
    assert loaded.cost == 0.0
    assert dict(loaded.spec) == spec


def test_learned_artifact_round_trip_uses_bound_cost(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    _policy().save(artifact)
    spec = _learned_spec(artifact, cost=0.15)

    loaded = load_strategy_spec(spec, base_dir=tmp_path)
    decision = loaded.strategy.decide_observed(
        "question", [Sample("answer", "A", 3)], loaded.cost
    )

    assert isinstance(loaded.strategy, BranchPilotPolicy)
    assert loaded.cost == 0.15
    assert decision.action == "stop"
    assert dict(loaded.spec) == spec


def test_learned_loader_reads_a_private_mode_600_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "policy.safetensors"
    _policy().save(artifact)
    spec = _learned_spec(artifact)
    original_load = BranchPilotPolicy.load.__func__
    observed: dict[str, object] = {}

    def inspect_copy(cls: type[BranchPilotPolicy], path: str | Path) -> BranchPilotPolicy:
        private_path = Path(path)
        observed["path"] = private_path
        observed["mode"] = private_path.stat().st_mode & 0o777
        observed["bytes"] = private_path.read_bytes()
        return original_load(cls, private_path)

    monkeypatch.setattr(BranchPilotPolicy, "load", classmethod(inspect_copy))
    load_strategy_spec(spec, base_dir=tmp_path)

    assert observed["path"] != artifact
    assert observed["mode"] == 0o600
    assert observed["bytes"] == artifact.read_bytes()
    assert not Path(observed["path"]).exists()


def test_descriptor_capture_survives_a_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "policy.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _policy(stop_bias=1.0).save(artifact)
    _policy(stop_bias=-1.0).save(replacement)
    spec = _learned_spec(artifact)
    real_read = os.read
    swapped = False

    def swap_before_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(replacement, artifact)
        return real_read(descriptor, count)

    monkeypatch.setattr(deployment_module.os, "read", swap_before_read)
    loaded = load_strategy_spec(spec, base_dir=tmp_path)
    decision = loaded.strategy.decide_observed(
        "question", [Sample("answer", "A", 3)], loaded.cost
    )

    assert swapped
    assert decision.action == "stop"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() != spec["policy_sha256"]


def test_rejects_artifact_replaced_before_descriptor_capture(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _policy(stop_bias=1.0).save(artifact)
    spec = _learned_spec(artifact)
    _policy(stop_bias=-1.0).save(replacement)
    os.replace(replacement, artifact)

    with pytest.raises(ValueError, match="SHA-256"):
        load_strategy_spec(spec, base_dir=tmp_path)


def test_rejects_hash_mismatch_and_symbolic_link(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    _policy().save(artifact)
    wrong_hash = _learned_spec(artifact)
    wrong_hash["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_strategy_spec(wrong_hash, base_dir=tmp_path)

    link = tmp_path / "linked.safetensors"
    link.symlink_to(artifact)
    linked_spec = _learned_spec(artifact, artifact=link.name)
    with pytest.raises(ValueError, match="symbolic link"):
        load_strategy_spec(linked_spec, base_dir=tmp_path)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"type": "unknown"},
        {"type": "fixed", "samples": 1.0, "max_samples": 4},
        {"type": "fixed", "samples": True, "max_samples": 4},
        {"type": "fixed", "samples": 1, "max_samples": 4, "extra": 1},
        {
            "type": "vote_confidence",
            "threshold": 1,
            "minimum": 2,
            "max_samples": 4,
        },
        {
            "type": "learned",
            "cost": 0.1,
            "policy_artifact": "policy.safetensors",
        },
        {
            "type": "learned",
            "cost": 0.1,
            "policy_artifact": "policy.safetensors",
            "policy_sha256": "0" * 64,
            "extra": False,
        },
        {
            "type": "learned",
            "cost": 1,
            "policy_artifact": "policy.safetensors",
            "policy_sha256": "0" * 64,
        },
        {
            "type": "learned",
            "cost": float("nan"),
            "policy_artifact": "policy.safetensors",
            "policy_sha256": "0" * 64,
        },
        {
            "type": "learned",
            "cost": 0.1,
            "policy_artifact": Path("policy.safetensors"),
            "policy_sha256": "0" * 64,
        },
        {
            "type": "learned",
            "cost": 0.1,
            "policy_artifact": "",
            "policy_sha256": "0" * 64,
        },
        {
            "type": "learned",
            "cost": 0.1,
            "policy_artifact": "policy.safetensors",
            "policy_sha256": "G" * 64,
        },
    ],
)
def test_rejects_malformed_extra_and_coerced_specs(spec: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        load_strategy_spec(spec)


def test_rejects_out_of_range_cost_during_load_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "policy.safetensors"
    _policy(costs=(0.1, 0.2)).save(artifact)
    spec = _learned_spec(artifact, cost=0.05)

    def must_not_sample(*args: object, **kwargs: object) -> None:
        raise AssertionError("sampling must not occur while loading")

    monkeypatch.setattr(BranchPilotPolicy, "decide_observed", must_not_sample)
    with pytest.raises(ValueError, match="trained range"):
        load_strategy_spec(spec, base_dir=tmp_path)


def test_loaded_deployment_has_defensive_immutable_state() -> None:
    source = {"type": "fixed", "samples": 1, "max_samples": 4}
    loaded = load_strategy_spec(source)
    source["samples"] = 4

    assert dict(loaded.spec) == {"type": "fixed", "samples": 1, "max_samples": 4}
    with pytest.raises(TypeError):
        loaded.spec["samples"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        loaded.cost = 1.0  # type: ignore[misc]


def test_plan_loads_learned_artifact_relative_to_plan_and_returns_metadata(
    tmp_path: Path,
) -> None:
    plan_dir = tmp_path / "deployment"
    artifact_dir = plan_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "policy.safetensors"
    _policy().save(artifact)
    spec = _learned_spec(artifact, cost=0.15, artifact="artifacts/policy.safetensors")
    payload = _plan_payload(spec, "offline-rl")
    plan_path = plan_dir / "plan.json"
    _write_plan(plan_path, payload)

    loaded, plan = load_deployment_plan(plan_path)

    assert isinstance(loaded, LoadedDeployment)
    assert isinstance(loaded.strategy, BranchPilotPolicy)
    assert loaded.cost == 0.15
    assert plan.to_dict() == payload
    with pytest.raises(TypeError):
        plan.strategy_spec["cost"] = 0.2  # type: ignore[index]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("policy"),
        lambda payload: payload.update(extra="untrusted"),
        lambda payload: payload.update(expected_accuracy=1),
        lambda payload: payload.update(expected_accuracy=True),
        lambda payload: payload.update(expected_accuracy=1.1),
        lambda payload: payload.update(expected_samples=0.0),
        lambda payload: payload.update(expected_tokens=-1.0),
        lambda payload: payload.update(requested_sample_budget=-1.0),
        lambda payload: payload.update(conservative=1),
        lambda payload: payload.update(budget_satisfied="yes"),
        lambda payload: payload.update(accuracy_interval=[0.7, 0.8]),
        lambda payload: payload["accuracy_interval"].update(extra=0.0),
        lambda payload: payload["accuracy_interval"].update(lower=0),
        lambda payload: payload["accuracy_interval"].update(lower=0.9),
        lambda payload: payload.update(expected_accuracy=0.9),
        lambda payload: payload["samples_interval"].update(upper=5.0),
        lambda payload: payload.update(family="heuristic"),
    ],
)
def test_plan_rejects_malformed_and_coerced_metric_fields(
    tmp_path: Path, mutate: object
) -> None:
    payload = _plan_payload(
        {"type": "fixed", "samples": 2, "max_samples": 4}, "fixed"
    )
    mutate(payload)  # type: ignore[operator]
    path = tmp_path / "plan.json"
    _write_plan(path, payload)

    with pytest.raises((TypeError, ValueError)):
        load_deployment_plan(path)


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "{not-json}",
        '{"family":"fixed","family":"fixed"}',
        "{\"family\": NaN}",
    ],
)
def test_plan_rejects_non_object_malformed_duplicate_and_nonfinite_json(
    tmp_path: Path, raw: str
) -> None:
    path = tmp_path / "plan.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_deployment_plan(path)


def test_plan_rejects_symbolic_link(tmp_path: Path) -> None:
    payload = _plan_payload(
        {"type": "fixed", "samples": 2, "max_samples": 4}, "fixed"
    )
    target = tmp_path / "target.json"
    link = tmp_path / "plan.json"
    _write_plan(target, payload)
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        load_deployment_plan(link)


def test_builtins_do_not_import_torch() -> None:
    project_root = Path(__file__).parents[1]
    script = """
import importlib.abc
import sys

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('torch import attempted')
        return None

sys.meta_path.insert(0, RejectTorch())
from branchpilot.deployment import load_strategy_spec
loaded = load_strategy_spec({'type': 'fixed', 'samples': 1, 'max_samples': 2})
assert loaded.cost == 0.0
"""
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
