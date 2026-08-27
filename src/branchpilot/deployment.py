from __future__ import annotations

import errno
import hashlib
import hmac
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from branchpilot.calibration import DeploymentPlan
from branchpilot.evaluate import Interval
from branchpilot.policy import MAX_ARTIFACT_BYTES, BranchPilotPolicy
from branchpilot.strategies import StoppingStrategy, strategy_from_spec

_PLAN_SCHEMA_VERSION = 1
_BENCHMARK_SCHEMA_VERSION = 2
_LEARNED_KEYS = frozenset({"type", "cost", "policy_artifact", "policy_sha256"})
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "selection_source",
        "family",
        "policy",
        "strategy_spec",
        "expected_accuracy",
        "accuracy_interval",
        "expected_samples",
        "samples_interval",
        "expected_tokens",
        "tokens_interval",
        "requested_sample_budget",
        "conservative",
        "budget_satisfied",
    }
)
_INTERVAL_KEYS = frozenset({"lower", "upper"})
_SELECTION_SOURCE_KEYS = frozenset({"benchmark_schema_version", "payload_sha256"})
_PLAN_FAMILY_BY_STRATEGY = {
    "learned": "offline-rl",
    "fixed": "fixed",
    "vote_confidence": "heuristic",
    "consecutive_agreement": "heuristic",
}


@dataclass(frozen=True, slots=True)
class LoadedDeployment:
    """A strategy and its server-controlled cost, captured from a strict spec."""

    strategy: StoppingStrategy
    cost: float
    spec: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec", MappingProxyType(dict(self.spec)))


def _require_exact_keys(
    value: dict[str, object], expected: frozenset[str], description: str
) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing keys: {', '.join(missing)}")
    if extra:
        details.append(f"extra keys: {', '.join(extra)}")
    raise ValueError(f"invalid {description} (" + "; ".join(details) + ")")


def _open_regular(path: Path, description: str) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{description} must not be a symbolic link") from exc
        raise ValueError(f"cannot open {description} {path!s}: {exc.strerror}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"{description} must be a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _load_private_policy(path: Path, expected_sha256: str) -> BranchPilotPolicy:
    descriptor = _open_regular(path, "policy artifact")
    try:
        size = os.fstat(descriptor).st_size
        if size < 1 or size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"policy artifact size must be in [1, {MAX_ARTIFACT_BYTES}] bytes")
        digest = hashlib.sha256()
        copied = 0
        with tempfile.TemporaryDirectory(prefix="branchpilot-policy-") as directory:
            private_path = Path(directory) / "policy.safetensors"
            private_descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(private_descriptor, "wb") as private_file:
                    private_descriptor = -1
                    while True:
                        chunk = os.read(descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > MAX_ARTIFACT_BYTES:
                            raise ValueError("policy artifact exceeds the supported size limit")
                        digest.update(chunk)
                        private_file.write(chunk)
            finally:
                if private_descriptor >= 0:
                    os.close(private_descriptor)
            if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
                raise ValueError("policy artifact SHA-256 does not match policy_sha256")
            return BranchPilotPolicy.load(private_path)
    finally:
        os.close(descriptor)


def load_strategy_spec(spec: object, *, base_dir: str | Path = ".") -> LoadedDeployment:
    """Load a strict built-in or content-bound learned deployment spec."""

    if type(spec) is not dict:
        raise TypeError("strategy spec must be a dictionary")
    snapshot = dict(spec)
    kind = snapshot.get("type")
    if kind != "learned":
        strategy = strategy_from_spec(snapshot)
        return LoadedDeployment(strategy=strategy, cost=0.0, spec=snapshot)

    _require_exact_keys(snapshot, _LEARNED_KEYS, "learned strategy spec")
    if type(kind) is not str:
        raise TypeError("strategy spec type must be a string")
    cost = snapshot["cost"]
    if type(cost) is not float:
        raise TypeError("learned strategy cost must be a float")
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("learned strategy cost must be finite and non-negative")
    artifact = snapshot["policy_artifact"]
    if type(artifact) is not str:
        raise TypeError("policy_artifact must be a string")
    if not artifact.strip():
        raise ValueError("policy_artifact must not be empty")
    expected_digest = snapshot["policy_sha256"]
    if type(expected_digest) is not str:
        raise TypeError("policy_sha256 must be a string")
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_digest
    ):
        raise ValueError("policy_sha256 must be 64 hexadecimal characters")

    policy = _load_private_policy(Path(base_dir) / artifact, expected_digest.lower())
    if cost < policy.costs[0] or cost > policy.costs[-1]:
        raise ValueError(
            "cost must be finite and within the trained range "
            f"[{policy.costs[0]:g}, {policy.costs[-1]:g}]"
        )
    return LoadedDeployment(strategy=policy, cost=cost, spec=snapshot)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _read_strict_json(path: Path) -> object:
    descriptor = _open_regular(path, "deployment plan")
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        text = b"".join(chunks).decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid deployment plan JSON: {exc}") from exc


def _text_field(payload: dict[str, object], field: str) -> str:
    value = payload[field]
    if type(value) is not str:
        raise TypeError(f"deployment plan field {field!r} must be a string")
    if not value:
        raise ValueError(f"deployment plan field {field!r} must not be empty")
    return value


def _float_field(
    payload: dict[str, object],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = payload[field]
    if type(value) is not float:
        raise TypeError(f"deployment plan field {field!r} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"deployment plan field {field!r} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"deployment plan field {field!r} must be at least {minimum:g}")
    if maximum is not None and value > maximum:
        raise ValueError(f"deployment plan field {field!r} must be at most {maximum:g}")
    return value


def _interval_field(
    payload: dict[str, object],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Interval:
    raw = payload[field]
    if type(raw) is not dict:
        raise TypeError(f"deployment plan field {field!r} must be a dictionary")
    _require_exact_keys(raw, _INTERVAL_KEYS, f"deployment plan field {field!r}")
    lower = _float_field(raw, "lower", minimum=minimum, maximum=maximum)
    upper = _float_field(raw, "upper", minimum=minimum, maximum=maximum)
    try:
        return Interval(lower, upper)
    except ValueError as exc:
        raise ValueError(f"invalid deployment plan field {field!r}: {exc}") from exc


def _bool_field(payload: dict[str, object], field: str) -> bool:
    value = payload[field]
    if type(value) is not bool:
        raise TypeError(f"deployment plan field {field!r} must be a boolean")
    return value


def _selection_source_field(payload: dict[str, object]) -> dict[str, object]:
    raw = payload["selection_source"]
    if type(raw) is not dict:
        raise TypeError("deployment plan field 'selection_source' must be a dictionary")
    _require_exact_keys(raw, _SELECTION_SOURCE_KEYS, "deployment plan selection source")

    benchmark_schema_version = raw["benchmark_schema_version"]
    if type(benchmark_schema_version) is not int:
        raise TypeError(
            "deployment plan selection source field 'benchmark_schema_version' must be an integer"
        )
    if benchmark_schema_version != _BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            "deployment plan selection source field 'benchmark_schema_version' "
            f"must be {_BENCHMARK_SCHEMA_VERSION}"
        )

    payload_sha256 = raw["payload_sha256"]
    if type(payload_sha256) is not str:
        raise TypeError("deployment plan selection source field 'payload_sha256' must be a string")
    if len(payload_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in payload_sha256
    ):
        raise ValueError(
            "deployment plan selection source field 'payload_sha256' must be "
            "64 lowercase hexadecimal characters"
        )
    return raw


def _require_estimate_in_interval(name: str, estimate: float, interval: Interval) -> None:
    if estimate < interval.lower or estimate > interval.upper:
        raise ValueError(f"deployment plan {name} must fall within its reported interval")


def load_deployment_plan(
    path: str | Path,
) -> tuple[LoadedDeployment, DeploymentPlan]:
    """Capture and load a serialized :class:`DeploymentPlan` and its strategy."""

    plan_path = Path(path)
    payload = _read_strict_json(plan_path)
    if type(payload) is not dict:
        raise TypeError("deployment plan must be a JSON object")
    _require_exact_keys(payload, _PLAN_KEYS, "deployment plan")

    schema_version = payload["schema_version"]
    if type(schema_version) is not int:
        raise TypeError("deployment plan field 'schema_version' must be an integer")
    if schema_version != _PLAN_SCHEMA_VERSION:
        raise ValueError(f"deployment plan field 'schema_version' must be {_PLAN_SCHEMA_VERSION}")
    selection_source = _selection_source_field(payload)

    family = _text_field(payload, "family")
    policy_name = _text_field(payload, "policy")
    strategy_spec = payload["strategy_spec"]
    if type(strategy_spec) is not dict:
        raise TypeError("deployment plan field 'strategy_spec' must be a dictionary")

    expected_accuracy = _float_field(payload, "expected_accuracy", minimum=0.0, maximum=1.0)
    accuracy_interval = _interval_field(payload, "accuracy_interval", minimum=0.0, maximum=1.0)
    expected_samples = _float_field(payload, "expected_samples", minimum=1.0)
    samples_interval = _interval_field(payload, "samples_interval", minimum=1.0)
    expected_tokens = _float_field(payload, "expected_tokens", minimum=0.0)
    tokens_interval = _interval_field(payload, "tokens_interval", minimum=0.0)
    requested_sample_budget = _float_field(payload, "requested_sample_budget", minimum=0.0)
    conservative = _bool_field(payload, "conservative")
    budget_satisfied = _bool_field(payload, "budget_satisfied")

    _require_estimate_in_interval("expected_accuracy", expected_accuracy, accuracy_interval)
    _require_estimate_in_interval("expected_samples", expected_samples, samples_interval)
    _require_estimate_in_interval("expected_tokens", expected_tokens, tokens_interval)

    deployment = load_strategy_spec(strategy_spec, base_dir=plan_path.parent)
    strategy_type = deployment.spec["type"]
    if type(strategy_type) is not str:
        raise TypeError("loaded strategy spec type must be a string")
    expected_family = _PLAN_FAMILY_BY_STRATEGY[strategy_type]
    if family != expected_family:
        raise ValueError(
            f"deployment plan family {family!r} does not match strategy type {strategy_type!r}"
        )
    if (
        expected_samples > deployment.strategy.max_samples
        or samples_interval.upper > deployment.strategy.max_samples
    ):
        raise ValueError("deployment plan sample metrics exceed the strategy's max_samples")

    plan = DeploymentPlan(
        schema_version=schema_version,
        selection_source=selection_source,
        family=family,
        policy=policy_name,
        strategy_spec=deployment.spec,
        expected_accuracy=expected_accuracy,
        accuracy_interval=accuracy_interval,
        expected_samples=expected_samples,
        samples_interval=samples_interval,
        expected_tokens=expected_tokens,
        tokens_interval=tokens_interval,
        requested_sample_budget=requested_sample_budget,
        conservative=conservative,
        budget_satisfied=budget_satisfied,
    )
    return deployment, plan
