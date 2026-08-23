from pathlib import Path

import pytest

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
