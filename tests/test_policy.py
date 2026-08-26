from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from branchpilot.evaluate import benchmark
from branchpilot.policy import MAX_COSTS, MAX_HIDDEN_SIZE, MAX_SAMPLES, BranchPilotPolicy
from branchpilot.schema import Rollout
from branchpilot.synthetic import make_synthetic_rollouts
from branchpilot.training import TrainConfig, _exact_q_targets, train_policy


@pytest.fixture(scope="module")
def trained() -> tuple[BranchPilotPolicy, list[Rollout]]:
    train = make_synthetic_rollouts(96, seed=101)
    test = make_synthetic_rollouts(48, seed=102)
    policy, _ = train_policy(
        train,
        TrainConfig(epochs=24, batch_size=512, costs=(0.01, 0.05, 0.15, 0.3), seed=3),
    )
    return policy, test


def _artifact_parts(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    with safe_open(path, framework="np") as artifact:
        names = artifact.keys()  # noqa: SIM118 - safe_open is not iterable
        tensors = {name: artifact.get_tensor(name).copy() for name in names}
        metadata = dict(artifact.metadata() or {})
    return tensors, metadata


def test_policy_always_stops_within_horizon(trained: tuple[BranchPilotPolicy, list]) -> None:
    policy, test = trained
    decisions = [policy.run(rollout, 0.05) for rollout in test]
    assert all(decision.action == "stop" for decision in decisions)
    assert all(1 <= decision.sample_count <= 8 for decision in decisions)


def test_cost_control_is_monotone_for_every_trajectory(
    trained: tuple[BranchPilotPolicy, list],
) -> None:
    policy, test = trained
    for rollout in test:
        counts = [policy.run(rollout, cost).sample_count for cost in policy.costs]
        assert counts == sorted(counts, reverse=True)


def test_policy_artifact_round_trip(
    trained: tuple[BranchPilotPolicy, list], tmp_path: Path
) -> None:
    policy, test = trained
    path = tmp_path / "policy.safetensors"
    policy.save(path, {"test": True})
    restored = BranchPilotPolicy.load(path)
    assert restored.run(test[0], 0.05) == policy.run(test[0], 0.05)
    assert restored.training == {"test": True}
    copied = restored.training
    copied["changed"] = True
    assert restored.training == {"test": True}


def test_incremental_session_matches_offline_replay(
    trained: tuple[BranchPilotPolicy, list],
) -> None:
    policy, test = trained
    rollout = test[0]
    expected = policy.run(rollout, 0.05)
    session = policy.start(
        rollout.question,
        0.05,
        prompt_tokens=rollout.prompt_tokens,
        max_samples=len(rollout.samples),
    )
    for sample in rollout.samples:
        observed = session.observe(sample)
        if observed.action == "stop":
            break
    result = session.result()
    assert result.sample_count == expected.sample_count
    assert result.answer == expected.majority_answer
    assert result.decisions[-1] == expected


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


def test_non_finite_and_out_of_range_costs_are_rejected(
    trained: tuple[BranchPilotPolicy, list],
) -> None:
    policy, test = trained
    with pytest.raises(ValueError, match="finite"):
        policy.run(test[0], float("nan"))
    with pytest.raises(ValueError, match="trained range"):
        policy.run(test[0], 0.5)
    with pytest.raises(ValueError, match="trained range"):
        policy.start(test[0].question, 0.5)


def test_zero_epoch_training_is_rejected() -> None:
    records = make_synthetic_rollouts(2)
    with pytest.raises(ValueError, match="epochs"):
        train_policy(records, TrainConfig(epochs=0))


@pytest.mark.parametrize(
    "config",
    (
        TrainConfig(hidden_size=MAX_HIDDEN_SIZE + 1),
        TrainConfig(max_samples=MAX_SAMPLES + 1),
        TrainConfig(costs=tuple(float(index) for index in range(MAX_COSTS + 1))),
    ),
)
def test_training_rejects_resource_bounds_before_allocation(config: TrainConfig) -> None:
    with pytest.raises(ValueError, match="must be in|cannot exceed"):
        train_policy(make_synthetic_rollouts(2), config)


def test_exact_targets_match_hand_calculated_backward_induction() -> None:
    rewards = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    targets = _exact_q_targets(rewards, 0.1)
    np.testing.assert_allclose(
        targets,
        np.asarray([[0.0, 0.9], [1.0, -0.1], [0.0, 0.0]], dtype=np.float32),
    )


def test_horizon_one_training_has_finite_loss() -> None:
    records = make_synthetic_rollouts(4, max_samples=1)
    policy, training = train_policy(
        records,
        TrainConfig(max_samples=1, epochs=2, batch_size=1, costs=(0.0, 0.1)),
    )
    assert np.isfinite(training["final_loss"])
    assert policy.run(records[0], 0.1).sample_count == 1


def test_corrupt_normalization_artifact_is_rejected(
    trained: tuple[BranchPilotPolicy, list], tmp_path: Path
) -> None:
    policy, _ = trained
    source = tmp_path / "policy.safetensors"
    corrupt = tmp_path / "corrupt.safetensors"
    policy.save(source)
    tensors, metadata = _artifact_parts(source)
    tensors["feature_std"][0] = 0
    save_file(tensors, corrupt, metadata=metadata)
    with pytest.raises(ValueError, match="standard deviations"):
        BranchPilotPolicy.load(corrupt)


def test_excessive_artifact_dimensions_are_rejected_before_allocation(
    trained: tuple[BranchPilotPolicy, list], tmp_path: Path
) -> None:
    policy, _ = trained
    source = tmp_path / "policy.safetensors"
    oversized = tmp_path / "oversized.safetensors"
    policy.save(source)
    tensors, metadata = _artifact_parts(source)
    metadata["hidden_size"] = str(MAX_HIDDEN_SIZE + 1)
    save_file(tensors, oversized, metadata=metadata)
    with pytest.raises(ValueError, match="maximum"):
        BranchPilotPolicy.load(oversized)


def test_legacy_pickle_policy_is_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    legacy.write_bytes(b"not a safetensors artifact")
    with pytest.raises(ValueError, match="invalid policy artifact"):
        BranchPilotPolicy.load(legacy)
