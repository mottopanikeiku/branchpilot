from dataclasses import replace
from pathlib import Path

import pytest
import torch

from branchpilot.evaluate import benchmark
from branchpilot.policy import BranchPilotPolicy, TrainConfig, train_policy
from branchpilot.synthetic import make_synthetic_rollouts


@pytest.fixture(scope="module")
def trained() -> tuple[BranchPilotPolicy, list]:
    train = make_synthetic_rollouts(96, seed=101)
    test = make_synthetic_rollouts(48, seed=102)
    policy, _ = train_policy(
        train,
        TrainConfig(epochs=24, batch_size=512, costs=(0.01, 0.05, 0.15, 0.3), seed=3),
    )
    return policy, test


def test_policy_always_stops_within_horizon(trained: tuple[BranchPilotPolicy, list]) -> None:
    policy, test = trained
    decisions = [policy.run(rollout, 0.05) for rollout in test]
    assert all(decision.action == "stop" for decision in decisions)
    assert all(1 <= decision.sample_count <= 8 for decision in decisions)


def test_higher_inference_cost_does_not_use_more_samples(
    trained: tuple[BranchPilotPolicy, list],
) -> None:
    policy, test = trained
    cheap = sum(policy.run(rollout, 0.01).sample_count for rollout in test)
    expensive = sum(policy.run(rollout, 0.3).sample_count for rollout in test)
    assert expensive <= cheap


def test_policy_artifact_round_trip(
    trained: tuple[BranchPilotPolicy, list], tmp_path: Path
) -> None:
    policy, test = trained
    path = tmp_path / "policy.pt"
    policy.save(path, {"test": True})
    restored = BranchPilotPolicy.load(path)
    before = policy.run(test[0], 0.05)
    after = restored.run(test[0], 0.05)
    assert after == before


def test_benchmark_reports_observable_accuracy_and_compute(
    trained: tuple[BranchPilotPolicy, list],
) -> None:
    policy, test = trained
    rows = benchmark(test, policy, costs=(0.05,))
    assert any(row.family == "offline-rl" for row in rows)
    assert any(row.policy == "fixed-8" for row in rows)
    assert all(0.0 <= row.accuracy <= 1.0 for row in rows)
    assert all(1.0 <= row.average_samples <= 8.0 for row in rows)


def test_truncated_trajectory_uses_its_actual_horizon(
    trained: tuple[BranchPilotPolicy, list],
) -> None:
    policy, test = trained
    short = replace(test[0], samples=test[0].samples[:3])
    decision = policy.run(short, 0.05)
    assert 1 <= decision.sample_count <= 3


def test_non_finite_cost_is_rejected(trained: tuple[BranchPilotPolicy, list]) -> None:
    policy, test = trained
    with pytest.raises(ValueError, match="finite"):
        policy.run(test[0], float("nan"))


def test_zero_epoch_training_is_rejected() -> None:
    records = make_synthetic_rollouts(2)
    with pytest.raises(ValueError, match="epochs"):
        train_policy(records, TrainConfig(epochs=0))


def test_corrupt_normalization_artifact_is_rejected(
    trained: tuple[BranchPilotPolicy, list], tmp_path: Path
) -> None:
    policy, _ = trained
    path = tmp_path / "corrupt.pt"
    policy.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["feature_std"][0] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="standard deviations"):
        BranchPilotPolicy.load(path)


def test_benchmark_caps_every_policy_at_shared_horizon(
    trained: tuple[BranchPilotPolicy, list], tmp_path: Path
) -> None:
    policy, test = trained
    path = tmp_path / "short-horizon.pt"
    policy.save(path)
    short_policy = BranchPilotPolicy.load(path)
    short_policy.max_samples = 3
    rows = benchmark(test, short_policy, costs=(0.05,))
    assert all(row.average_samples <= 3 for row in rows)
    assert not any(row.policy in {"fixed-4", "fixed-8"} for row in rows)
