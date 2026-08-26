from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from branchpilot.features import FEATURE_NAMES, prefix_correct, prefix_state
from branchpilot.policy import BranchPilotPolicy
from branchpilot.schema import Rollout


@dataclass(frozen=True, slots=True)
class TrainConfig:
    max_samples: int = 8
    hidden_size: int = 64
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 3e-4
    seed: int = 7
    costs: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25)


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validate_config(config: TrainConfig) -> tuple[float, ...]:
    if config.max_samples < 1 or config.hidden_size < 1:
        raise ValueError("max_samples and hidden_size must be positive")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    costs = tuple(float(cost) for cost in config.costs)
    if not costs or any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("training costs must be a non-empty finite non-negative sequence")
    if any(left >= right for left, right in pairwise(costs)):
        raise ValueError("training costs must be strictly increasing")
    return costs


def _exact_q_targets(stop_rewards: np.ndarray, cost: float) -> np.ndarray:
    """Solve one fully observed finite-horizon trajectory by backward induction."""
    targets = np.empty((len(stop_rewards), 2), dtype=np.float32)
    targets[:, 0] = stop_rewards
    targets[-1, 1] = stop_rewards[-1]
    best_next = float(stop_rewards[-1])
    for index in range(len(stop_rewards) - 2, -1, -1):
        continue_value = -cost + best_next
        targets[index, 1] = continue_value
        best_next = max(float(stop_rewards[index]), continue_value)
    return targets


def _prepare_training_tensors(
    rollouts: list[Rollout], config: TrainConfig
) -> tuple[TensorDataset, np.ndarray, np.ndarray, int]:
    states: list[np.ndarray] = []
    remaining: list[float] = []
    segments: list[tuple[int, np.ndarray]] = []

    for rollout in rollouts:
        horizon = min(config.max_samples, len(rollout.samples))
        start = len(states)
        stop_rewards = np.empty(horizon, dtype=np.float32)
        for count in range(1, horizon + 1):
            states.append(prefix_state(rollout, count, horizon).features)
            remaining.append(float(horizon - count))
            stop_rewards[count - 1] = float(prefix_correct(rollout, count))
        segments.append((start, stop_rewards))

    state_matrix = np.stack(states).astype(np.float32)
    feature_mean = state_matrix.mean(axis=0, dtype=np.float32)
    feature_std = state_matrix.std(axis=0, dtype=np.float32)
    feature_std[feature_std < 1e-5] = 1.0
    standardized = (state_matrix - feature_mean) / feature_std
    remaining_array = np.asarray(remaining, dtype=np.float32)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    continue_masks: list[np.ndarray] = []
    for cost in config.costs:
        cost_column = np.full((len(states), 1), cost, dtype=np.float32)
        inputs.append(
            np.concatenate(
                [
                    standardized,
                    cost_column,
                    (remaining_array / config.max_samples)[:, None],
                    (cost_column[:, 0] * remaining_array)[:, None],
                ],
                axis=1,
            )
        )
        cost_targets = np.empty((len(states), 2), dtype=np.float32)
        cost_mask = np.zeros(len(states), dtype=bool)
        for start, stop_rewards in segments:
            end = start + len(stop_rewards)
            cost_targets[start:end] = _exact_q_targets(stop_rewards, float(cost))
            cost_mask[start : end - 1] = True
        targets.append(cost_targets)
        continue_masks.append(cost_mask)

    dataset = TensorDataset(
        torch.from_numpy(np.concatenate(inputs, axis=0)),
        torch.from_numpy(np.concatenate(targets, axis=0)),
        torch.from_numpy(np.concatenate(continue_masks, axis=0)),
    )
    return dataset, feature_mean, feature_std, len(states)


def _export_weights(network: _QNetwork) -> dict[str, np.ndarray]:
    state = network.state_dict()
    mapping = {
        "input.weight": "layers.0.weight",
        "input.bias": "layers.0.bias",
        "norm.weight": "layers.1.weight",
        "norm.bias": "layers.1.bias",
        "hidden.weight": "layers.3.weight",
        "hidden.bias": "layers.3.bias",
        "output.weight": "layers.5.weight",
        "output.bias": "layers.5.bias",
    }
    return {
        destination: np.ascontiguousarray(
            state[source].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        for destination, source in mapping.items()
    }


def train_policy(
    rollouts: list[Rollout], config: TrainConfig | None = None
) -> tuple[BranchPilotPolicy, dict[str, Any]]:
    """Fit one universal Q approximator to exact offline backward-induction targets."""
    config = config or TrainConfig()
    if not rollouts:
        raise ValueError("training requires at least one rollout")
    costs = _validate_config(config)
    _seed_everything(config.seed)
    dataset, feature_mean, feature_std, base_states = _prepare_training_tensors(rollouts, config)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    network = _QNetwork(len(FEATURE_NAMES) + 3, config.hidden_size)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    epoch_loss = 0.0

    for epoch in range(config.epochs):
        network.train()
        total_loss = 0.0
        batches = 0
        for batch_index, (model_input, target, can_continue) in enumerate(loader):
            predicted = network(model_input)
            stop_loss = F.smooth_l1_loss(predicted[:, 0], target[:, 0])
            if bool(can_continue.any()):
                continue_loss = F.smooth_l1_loss(
                    predicted[can_continue, 1], target[can_continue, 1]
                )
            else:
                continue_loss = predicted[:, 1].sum() * 0.0
            loss = stop_loss + continue_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"non-finite training loss at epoch {epoch}, batch {batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError(f"non-finite gradient at epoch {epoch}, batch {batch_index}")
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1

        epoch_loss = total_loss / max(1, batches)

    training = {
        "algorithm": "exact-backward-q-regression-v1",
        "config": asdict(config),
        "rollouts": len(rollouts),
        "base_states": base_states,
        "state_cost_pairs": len(dataset),
        "final_loss": epoch_loss,
    }
    policy = BranchPilotPolicy(
        _export_weights(network.eval()),
        feature_mean,
        feature_std,
        config.max_samples,
        config.hidden_size,
        costs,
        training,
    )
    return policy, training
