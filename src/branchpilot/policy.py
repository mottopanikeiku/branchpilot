from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from branchpilot.features import FEATURE_NAMES, prefix_correct, prefix_state
from branchpilot.schema import Rollout

ARTIFACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainConfig:
    max_samples: int = 8
    hidden_size: int = 64
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 3e-4
    target_update_interval: int = 4
    seed: int = 7
    costs: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25)


@dataclass(frozen=True, slots=True)
class Decision:
    action: str
    q_stop: float
    q_continue: float
    sample_count: int
    majority_answer: str | None


class _QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class BranchPilotPolicy:
    """A universal value function approximator conditioned on inference cost."""

    def __init__(
        self,
        network: _QNetwork,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        max_samples: int,
        hidden_size: int,
    ) -> None:
        self.network = network.eval()
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.max_samples = max_samples
        self.hidden_size = hidden_size

    def _input(self, features: np.ndarray, cost: float) -> torch.Tensor:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        standardized = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_std
        progress = float(features[0])
        remaining = max(0.0, (1.0 - progress) * self.max_samples)
        model_input = np.concatenate(
            [standardized, np.asarray([cost, cost * remaining], dtype=np.float32)]
        )
        return torch.from_numpy(model_input).unsqueeze(0)

    @torch.inference_mode()
    def q_values(self, features: np.ndarray, cost: float) -> tuple[float, float]:
        values = self.network(self._input(features, cost)).squeeze(0)
        return float(values[0]), float(values[1])

    def decide(self, rollout: Rollout, count: int, cost: float) -> Decision:
        state = prefix_state(rollout, count, self.max_samples)
        q_stop, q_continue = self.q_values(state.features, cost)
        exhausted = count >= min(self.max_samples, len(rollout.samples))
        action = "stop" if exhausted or q_stop >= q_continue else "continue"
        return Decision(action, q_stop, q_continue, count, state.majority_answer)

    def run(self, rollout: Rollout, cost: float) -> Decision:
        horizon = min(self.max_samples, len(rollout.samples))
        for count in range(1, horizon + 1):
            decision = self.decide(rollout, count, cost)
            if decision.action == "stop":
                return decision
        raise RuntimeError("policy failed to stop at its horizon")

    def save(self, path: str | Path, training: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "artifact_version": ARTIFACT_VERSION,
                "feature_names": FEATURE_NAMES,
                "feature_mean": torch.from_numpy(self.feature_mean),
                "feature_std": torch.from_numpy(self.feature_std),
                "hidden_size": self.hidden_size,
                "max_samples": self.max_samples,
                "state_dict": self.network.state_dict(),
                "training": training or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> BranchPilotPolicy:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["artifact_version"]) != ARTIFACT_VERSION:
            raise ValueError(f"unsupported policy artifact {payload['artifact_version']}")
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("policy features do not match this BranchPilot version")
        hidden_size = int(payload["hidden_size"])
        network = _QNetwork(len(FEATURE_NAMES) + 2, hidden_size)
        network.load_state_dict(payload["state_dict"])
        return cls(
            network,
            payload["feature_mean"].numpy(),
            payload["feature_std"].numpy(),
            int(payload["max_samples"]),
            hidden_size,
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _prepare_training_tensors(
    rollouts: list[Rollout], config: TrainConfig
) -> tuple[TensorDataset, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    stop_rewards: list[float] = []
    can_continue: list[bool] = []
    next_can_continue: list[bool] = []
    remaining: list[float] = []

    for rollout in rollouts:
        horizon = min(config.max_samples, len(rollout.samples))
        for count in range(1, horizon + 1):
            states.append(prefix_state(rollout, count, config.max_samples).features)
            next_count = min(count + 1, horizon)
            next_states.append(prefix_state(rollout, next_count, config.max_samples).features)
            stop_rewards.append(float(prefix_correct(rollout, count)))
            can_continue.append(count < horizon)
            next_can_continue.append(next_count < horizon)
            remaining.append(float(horizon - count))

    state_matrix = np.stack(states).astype(np.float32)
    next_matrix = np.stack(next_states).astype(np.float32)
    feature_mean = state_matrix.mean(axis=0)
    feature_std = state_matrix.std(axis=0)
    feature_std[feature_std < 1e-5] = 1.0
    state_matrix = (state_matrix - feature_mean) / feature_std
    next_matrix = (next_matrix - feature_mean) / feature_std

    tiled_states: list[np.ndarray] = []
    tiled_next_states: list[np.ndarray] = []
    tiled_stop: list[float] = []
    tiled_continue: list[bool] = []
    tiled_next_continue: list[bool] = []
    tiled_cost: list[float] = []
    tiled_remaining: list[float] = []
    for cost in config.costs:
        tiled_states.append(state_matrix)
        tiled_next_states.append(next_matrix)
        tiled_stop.extend(stop_rewards)
        tiled_continue.extend(can_continue)
        tiled_next_continue.extend(next_can_continue)
        tiled_cost.extend([cost] * len(states))
        tiled_remaining.extend(remaining)

    current = np.concatenate(tiled_states, axis=0)
    following = np.concatenate(tiled_next_states, axis=0)
    costs = np.asarray(tiled_cost, dtype=np.float32)
    remaining_array = np.asarray(tiled_remaining, dtype=np.float32)
    current_input = np.concatenate(
        [current, costs[:, None], (costs * remaining_array)[:, None]], axis=1
    )
    next_remaining = np.maximum(remaining_array - 1.0, 0.0)
    next_input = np.concatenate(
        [following, costs[:, None], (costs * next_remaining)[:, None]], axis=1
    )

    dataset = TensorDataset(
        torch.from_numpy(current_input),
        torch.from_numpy(next_input),
        torch.tensor(tiled_stop, dtype=torch.float32),
        torch.tensor(tiled_continue, dtype=torch.bool),
        torch.tensor(tiled_next_continue, dtype=torch.bool),
        torch.from_numpy(costs),
    )
    return dataset, feature_mean, feature_std


def train_policy(
    rollouts: list[Rollout], config: TrainConfig | None = None
) -> tuple[BranchPilotPolicy, dict[str, Any]]:
    config = config or TrainConfig()
    if not rollouts:
        raise ValueError("training requires at least one rollout")
    if not config.costs or any(cost < 0 for cost in config.costs):
        raise ValueError("training costs must be a non-empty non-negative sequence")
    _seed_everything(config.seed)
    dataset, feature_mean, feature_std = _prepare_training_tensors(rollouts, config)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    network = _QNetwork(len(FEATURE_NAMES) + 2, config.hidden_size)
    target = _QNetwork(len(FEATURE_NAMES) + 2, config.hidden_size)
    target.load_state_dict(network.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    epoch_loss = 0.0

    for epoch in range(config.epochs):
        network.train()
        total_loss = 0.0
        batches = 0
        for current, following, stop_reward, can_continue, next_can_continue, cost in loader:
            with torch.no_grad():
                target_next = target(following)
                next_value = target_next.max(dim=1).values
                next_value = torch.where(next_can_continue, next_value, target_next[:, 0])
                continue_target = -cost + next_value

            predicted = network(current)
            stop_loss = F.smooth_l1_loss(predicted[:, 0], stop_reward)
            if can_continue.any():
                continue_loss = F.smooth_l1_loss(
                    predicted[can_continue, 1], continue_target[can_continue]
                )
            else:
                continue_loss = predicted[:, 1].sum() * 0.0
            loss = stop_loss + continue_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1

        epoch_loss = total_loss / max(1, batches)
        if (epoch + 1) % config.target_update_interval == 0:
            target.load_state_dict(network.state_dict())

    policy = BranchPilotPolicy(
        network.eval(), feature_mean, feature_std, config.max_samples, config.hidden_size
    )
    training = {
        "config": asdict(config),
        "rollouts": len(rollouts),
        "states": len(dataset),
        "final_loss": epoch_loss,
    }
    return policy, training
