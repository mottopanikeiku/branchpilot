from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save as save_safetensors

from branchpilot.artifacts import atomic_write_bytes
from branchpilot.features import FEATURE_NAMES, observed_state
from branchpilot.schema import Rollout, Sample

ARTIFACT_VERSION = 3
ARCHITECTURE = "mlp-layernorm-silu-v1"
COST_MODEL = "additional-samples-v1"
TRAINING_ALGORITHM = "exact-backward-q-regression-v1"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_COSTS = 256
MAX_HIDDEN_SIZE = 4096
MAX_SAMPLES = 1024
_LAYER_NORM_EPSILON = np.float32(1e-5)
_NETWORK_TENSORS = (
    "input.weight",
    "input.bias",
    "norm.weight",
    "norm.bias",
    "hidden.weight",
    "hidden.bias",
    "output.weight",
    "output.bias",
)
_METADATA_KEYS = {
    "format",
    "artifact_version",
    "architecture",
    "cost_model",
    "feature_names",
    "hidden_size",
    "max_samples",
    "costs",
    "training",
}


@dataclass(frozen=True, slots=True)
class Decision:
    action: str
    q_stop: float
    q_continue: float
    sample_count: int
    majority_answer: str | None

    @property
    def margin(self) -> float:
        """Positive values favor STOP; negative values favor CONTINUE."""
        return self.q_stop - self.q_continue


def _expected_shapes(hidden_size: int) -> dict[str, tuple[int, ...]]:
    input_size = len(FEATURE_NAMES) + 3
    return {
        "input.weight": (hidden_size, input_size),
        "input.bias": (hidden_size,),
        "norm.weight": (hidden_size,),
        "norm.bias": (hidden_size,),
        "hidden.weight": (hidden_size, hidden_size),
        "hidden.bias": (hidden_size,),
        "output.weight": (2, hidden_size),
        "output.bias": (2,),
    }


def _positive_metadata_int(metadata: Mapping[str, str], key: str, maximum: int) -> int:
    raw = metadata.get(key)
    if raw is None or re.fullmatch(r"[1-9]\d*", raw) is None:
        raise ValueError(f"policy metadata {key!r} must be a positive integer")
    value = int(raw)
    if value > maximum:
        raise ValueError(f"policy metadata {key!r} exceeds the supported maximum {maximum}")
    return value


def _validated_costs(values: Sequence[float]) -> tuple[float, ...]:
    costs = tuple(float(value) for value in values)
    if len(costs) > MAX_COSTS:
        raise ValueError(f"policy costs cannot exceed {MAX_COSTS} entries")
    if not costs or any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("policy costs must be a non-empty finite non-negative sequence")
    if any(left >= right for left, right in pairwise(costs)):
        raise ValueError("policy costs must be strictly increasing")
    return costs


def _silu(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, np.float32(-80.0), np.float32(80.0))
    return values / (np.float32(1.0) + np.exp(-clipped))


def _project_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares isotonic projection with one unit-weight observation per cost."""
    levels: list[float] = []
    weights: list[int] = []
    for value in -np.asarray(values, dtype=np.float64):
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / combined_weight
            levels[-2:] = [combined_level]
            weights[-2:] = [combined_weight]
    projected = np.concatenate(
        [
            np.full(weight, level, dtype=np.float64)
            for level, weight in zip(levels, weights, strict=True)
        ]
    )
    return -projected


class BranchPilotPolicy:
    """Torch-free inference for a cost-conditioned BranchPilot Q-policy."""

    def __init__(
        self,
        weights: Mapping[str, np.ndarray],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        max_samples: int,
        hidden_size: int,
        costs: Sequence[float],
        training: Mapping[str, Any] | None = None,
    ) -> None:
        if max_samples < 1 or max_samples > MAX_SAMPLES:
            raise ValueError(f"max_samples must be in [1, {MAX_SAMPLES}]")
        if hidden_size < 1 or hidden_size > MAX_HIDDEN_SIZE:
            raise ValueError(f"hidden_size must be in [1, {MAX_HIDDEN_SIZE}]")

        feature_mean = np.asarray(feature_mean)
        feature_std = np.asarray(feature_std)
        expected_feature_shape = (len(FEATURE_NAMES),)
        if (
            feature_mean.shape != expected_feature_shape
            or feature_std.shape != expected_feature_shape
        ):
            raise ValueError(f"normalization tensors must have shape {expected_feature_shape}")
        if feature_mean.dtype != np.float32 or feature_std.dtype != np.float32:
            raise ValueError("normalization tensors must use float32")
        if not np.isfinite(feature_mean).all():
            raise ValueError("feature means must be finite")
        if not np.isfinite(feature_std).all() or np.any(feature_std <= 0):
            raise ValueError("feature standard deviations must be finite and positive")

        expected_shapes = _expected_shapes(hidden_size)
        if set(weights) != set(_NETWORK_TENSORS):
            missing = sorted(set(_NETWORK_TENSORS) - set(weights))
            extra = sorted(set(weights) - set(_NETWORK_TENSORS))
            raise ValueError(
                f"policy tensors do not match the architecture; missing={missing}, extra={extra}"
            )
        checked: dict[str, np.ndarray] = {}
        for name in _NETWORK_TENSORS:
            value = np.asarray(weights[name])
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"policy tensor {name!r} has shape {value.shape}, "
                    f"expected {expected_shapes[name]}"
                )
            if value.dtype != np.float32:
                raise ValueError(f"policy tensor {name!r} must use float32")
            if not np.isfinite(value).all():
                raise ValueError(f"policy tensor {name!r} must be finite")
            checked[name] = np.ascontiguousarray(value).copy()

        self._weights = checked
        self.feature_mean = np.ascontiguousarray(feature_mean).copy()
        self.feature_std = np.ascontiguousarray(feature_std).copy()
        self.max_samples = max_samples
        self.hidden_size = hidden_size
        self.costs = _validated_costs(costs)
        self._training = json.loads(json.dumps(dict(training or {}), sort_keys=True))

    @property
    def training(self) -> dict[str, Any]:
        """Return a defensive copy of the artifact's training provenance."""
        return json.loads(json.dumps(self._training, sort_keys=True))

    def _validate_cost(self, cost: float) -> None:
        if not math.isfinite(cost) or cost < self.costs[0] or cost > self.costs[-1]:
            raise ValueError(
                f"cost must be finite and within the trained range "
                f"[{self.costs[0]:g}, {self.costs[-1]:g}]"
            )

    def _model_input(
        self,
        features: np.ndarray,
        cost: float,
        remaining_samples: float,
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("features must be a finite BranchPilot state vector")
        self._validate_cost(cost)
        if (
            not math.isfinite(remaining_samples)
            or remaining_samples < 0
            or remaining_samples > self.max_samples
        ):
            raise ValueError(f"remaining_samples must be finite and within [0, {self.max_samples}]")
        standardized = (features - self.feature_mean) / self.feature_std
        return np.concatenate(
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
        ).astype(np.float32, copy=False)

    def _forward(self, model_input: np.ndarray) -> np.ndarray:
        weights = self._weights
        hidden = weights["input.weight"] @ model_input + weights["input.bias"]
        centered = hidden - np.mean(hidden, dtype=np.float32)
        variance = np.mean(centered * centered, dtype=np.float32)
        hidden = centered / np.sqrt(variance + _LAYER_NORM_EPSILON)
        hidden = hidden * weights["norm.weight"] + weights["norm.bias"]
        hidden = _silu(hidden)
        hidden = weights["hidden.weight"] @ hidden + weights["hidden.bias"]
        hidden = _silu(hidden)
        return weights["output.weight"] @ hidden + weights["output.bias"]

    def q_values(
        self,
        features: np.ndarray,
        cost: float,
        remaining_samples: float | None = None,
    ) -> tuple[float, float]:
        self._validate_cost(cost)
        if remaining_samples is None:
            progress = float(features[0])
            remaining_samples = max(0.0, (1.0 - progress) * self.max_samples)
        raw = np.asarray(
            [
                self._forward(self._model_input(features, trained_cost, remaining_samples))
                for trained_cost in self.costs
            ],
            dtype=np.float64,
        )
        grid = np.asarray(self.costs, dtype=np.float64)
        q_stop = float(np.interp(cost, grid, raw[:, 0]))
        advantage = _project_nonincreasing(raw[:, 1] - raw[:, 0])
        q_continue = q_stop + float(np.interp(cost, grid, advantage))
        return q_stop, q_continue

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        """Decide from an unlabeled prefix without reading future samples."""
        observed = tuple(samples)
        horizon = self.max_samples if max_samples is None else max_samples
        if horizon < 1 or horizon > self.max_samples:
            raise ValueError(f"max_samples must be in [1, {self.max_samples}]")
        state = observed_state(question, observed, horizon, prompt_tokens)
        count = len(observed)
        q_stop, q_continue = self.q_values(state.features, cost, horizon - count)
        action = "stop" if count >= horizon or q_stop >= q_continue else "continue"
        return Decision(action, q_stop, q_continue, count, state.majority_answer)

    def decide(self, rollout: Rollout, count: int, cost: float) -> Decision:
        horizon = min(self.max_samples, len(rollout.samples))
        if count < 1 or count > horizon:
            raise ValueError(f"count must be in [1, {horizon}], got {count}")
        return self.decide_observed(
            rollout.question,
            rollout.samples[:count],
            cost,
            prompt_tokens=rollout.prompt_tokens,
            max_samples=horizon,
        )

    def run(self, rollout: Rollout, cost: float) -> Decision:
        horizon = min(self.max_samples, len(rollout.samples))
        for count in range(1, horizon + 1):
            decision = self.decide(rollout, count, cost)
            if decision.action == "stop":
                return decision
        raise RuntimeError("policy failed to stop at its horizon")

    def start(
        self,
        question: str,
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ):
        """Create a label-free incremental inference session."""
        from branchpilot.runtime import PilotSession

        self._validate_cost(cost)

        return PilotSession(
            self,
            question,
            cost,
            prompt_tokens=prompt_tokens,
            max_samples=max_samples,
        )

    def save(self, path: str | Path, training: Mapping[str, Any] | None = None) -> None:
        training_payload = dict(self._training if training is None else training)
        metadata = {
            "format": "branchpilot-policy",
            "artifact_version": str(ARTIFACT_VERSION),
            "architecture": ARCHITECTURE,
            "cost_model": COST_MODEL,
            "feature_names": json.dumps(FEATURE_NAMES, separators=(",", ":")),
            "hidden_size": str(self.hidden_size),
            "max_samples": str(self.max_samples),
            "costs": json.dumps(self.costs, separators=(",", ":")),
            "training": json.dumps(training_payload, separators=(",", ":"), sort_keys=True),
        }
        tensors = {
            **self._weights,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
        }
        encoded = save_safetensors(tensors, metadata=metadata)
        if len(encoded) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"policy artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
        atomic_write_bytes(path, encoded)

    @classmethod
    def load(cls, path: str | Path) -> BranchPilotPolicy:
        source = Path(path)
        size = source.stat().st_size
        if size < 1 or size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"policy artifact size must be in [1, {MAX_ARTIFACT_BYTES}] bytes")
        try:
            with safe_open(source, framework="np") as artifact:
                metadata = artifact.metadata() or {}
                if set(metadata) != _METADATA_KEYS:
                    missing = sorted(_METADATA_KEYS - set(metadata))
                    extra = sorted(set(metadata) - _METADATA_KEYS)
                    raise ValueError(
                        f"policy metadata does not match schema; missing={missing}, extra={extra}"
                    )
                if metadata["format"] != "branchpilot-policy":
                    raise ValueError("file is not a BranchPilot policy artifact")
                if metadata["artifact_version"] != str(ARTIFACT_VERSION):
                    raise ValueError(
                        f"unsupported policy artifact {metadata['artifact_version']}; "
                        f"expected {ARTIFACT_VERSION}"
                    )
                if metadata["architecture"] != ARCHITECTURE:
                    raise ValueError("unsupported policy architecture")
                if metadata["cost_model"] != COST_MODEL:
                    raise ValueError("unsupported policy cost model")
                feature_names = tuple(json.loads(metadata["feature_names"]))
                if feature_names != FEATURE_NAMES:
                    raise ValueError("policy features do not match this BranchPilot version")
                hidden_size = _positive_metadata_int(metadata, "hidden_size", MAX_HIDDEN_SIZE)
                max_samples = _positive_metadata_int(metadata, "max_samples", MAX_SAMPLES)
                costs = _validated_costs(json.loads(metadata["costs"]))
                training = json.loads(metadata["training"])
                if not isinstance(training, dict):
                    raise ValueError("policy training metadata must be a JSON object")

                expected_shapes = {
                    **_expected_shapes(hidden_size),
                    "feature_mean": (len(FEATURE_NAMES),),
                    "feature_std": (len(FEATURE_NAMES),),
                }
                if set(artifact.keys()) != set(expected_shapes):
                    missing = sorted(set(expected_shapes) - set(artifact.keys()))
                    extra = sorted(set(artifact.keys()) - set(expected_shapes))
                    raise ValueError(
                        f"policy tensors do not match schema; missing={missing}, extra={extra}"
                    )
                for name, expected_shape in expected_shapes.items():
                    shape = tuple(artifact.get_slice(name).get_shape())
                    if shape != expected_shape:
                        raise ValueError(
                            f"policy tensor {name!r} has shape {shape}, expected {expected_shape}"
                        )
                tensors = {name: artifact.get_tensor(name) for name in expected_shapes}
            return cls(
                {name: tensors[name] for name in _NETWORK_TENSORS},
                tensors["feature_mean"],
                tensors["feature_std"],
                max_samples,
                hidden_size,
                costs,
                training,
            )
        except (SafetensorError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid policy artifact: {exc}") from exc
