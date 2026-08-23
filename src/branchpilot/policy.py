from __future__ import annotations

import math
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

ARTIFACT_VERSION = 2


@dataclass(frozen=True, slots=True)
class TrainConfig:
    max_samples: int = 8
    hidden_size: int = 64
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 3e-4
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
        feature_mean = np.asarray(feature_mean, dtype=np.float32)
        feature_std = np.asarray(feature_std, dtype=np.float32)
        expected_shape = (len(FEATURE_NAMES),)
        if max_samples < 1 or hidden_size < 1:
            raise ValueError("policy dimensions must be positive")
        if feature_mean.shape != expected_shape or feature_std.shape != expected_shape:
            raise ValueError(f"normalization tensors must have shape {expected_shape}")
        if not np.isfinite(feature_mean).all():
            raise ValueError("feature means must be finite")
        if not np.isfinite(feature_std).all() or np.any(feature_std <= 0):
            raise ValueError("feature standard deviations must be finite and positive")
        self.network = network.eval()
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.max_samples = max_samples
        self.hidden_size = hidden_size

    def _input(
        self,
        features: np.ndarray,
        cost: float,
        remaining_samples: float,
    ) -> torch.Tensor:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("features must be a finite BranchPilot state vector")
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and non-negative")
        if not math.isfinite(remaining_samples) or remaining_samples < 0:
            raise ValueError("remaining_samples must be finite and non-negative")
        standardized = (features - self.feature_mean) / self.feature_std
        model_input = np.concatenate(
            [
                standardized,
                np.asarray(
                    [
                        cost,
                        remaining_samples / self.max_samples,
                        cost * remaining_samples,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return torch.from_numpy(model_input).unsqueeze(0)

    @torch.inference_mode()
    def q_values(
        self,
        features: np.ndarray,
        cost: float,
        remaining_samples: float | None = None,
    ) -> tuple[float, float]:
        if remaining_samples is None:
            progress = float(features[0])
            remaining_samples = max(0.0, (1.0 - progress) * self.max_samples)
        values = self.network(self._input(features, cost, remaining_samples)).squeeze(0)
        return float(values[0]), float(values[1])

    def decide(self, rollout: Rollout, count: int, cost: float) -> Decision:
        horizon = min(self.max_samples, len(rollout.samples))
        if count < 1 or count > horizon:
            raise ValueError(f"count must be in [1, {horizon}], got {count}")
        state = prefix_state(rollout, count, self.max_samples)
        q_stop, q_continue = self.q_values(state.features, cost, horizon - count)
        action = "stop" if count >= horizon or q_stop >= q_continue else "continue"
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
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if int(payload["artifact_version"]) != ARTIFACT_VERSION:
                raise ValueError(f"unsupported policy artifact {payload['artifact_version']}")
            if tuple(payload["feature_names"]) != FEATURE_NAMES:
                raise ValueError("policy features do not match this BranchPilot version")
            hidden_size = int(payload["hidden_size"])
            max_samples = int(payload["max_samples"])
            feature_mean = payload["feature_mean"].detach().cpu().numpy()
            feature_std = payload["feature_std"].detach().cpu().numpy()
            network = _QNetwork(len(FEATURE_NAMES) + 3, hidden_size)
            network.load_state_dict(payload["state_dict"])
            return cls(
                network,
                feature_mean,
                feature_std,
                max_samples,
                hidden_size,
            )
        except (KeyError, RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid policy artifact: {exc}") from exc


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
        [
            current,
            costs[:, None],
            (remaining_array / config.max_samples)[:, None],
            (costs * remaining_array)[:, None],
        ],
        axis=1,
    )
    next_remaining = np.maximum(remaining_array - 1.0, 0.0)
    next_input = np.concatenate(
        [
            following,
            costs[:, None],
            (next_remaining / config.max_samples)[:, None],
            (costs * next_remaining)[:, None],
        ],
        axis=1,
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
    if config.max_samples < 1 or config.hidden_size < 1:
        raise ValueError("max_samples and hidden_size must be positive")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not config.costs or any(not math.isfinite(cost) or cost < 0 for cost in config.costs):
        raise ValueError("training costs must be a non-empty finite non-negative sequence")
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

    network = _QNetwork(len(FEATURE_NAMES) + 3, config.hidden_size)
    target = _QNetwork(len(FEATURE_NAMES) + 3, config.hidden_size)
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
                online_action = network(following).argmax(dim=1)
                next_value = target_next.gather(1, online_action[:, None]).squeeze(1)
                next_value = torch.where(next_can_continue, next_value, target_next[:, 0])
                continue_target = -cost + next_value

            predicted = network(current)
            stop_loss = F.smooth_l1_loss(predicted[:, 0], stop_reward)
            continue_loss = F.smooth_l1_loss(
                predicted[can_continue, 1], continue_target[can_continue]
            )
            loss = stop_loss + continue_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1

        epoch_loss = total_loss / max(1, batches)
        if (epoch + 1) % 4 == 0:
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
