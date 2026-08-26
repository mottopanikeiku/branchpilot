from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "FileSnapshot",
    "Protocol",
    "capture_file",
    "load_manifest",
    "validate_manifest_protocol",
    "verify_manifest_artifact",
]

_PathLike = str | os.PathLike[str]
_PROTOCOL_KEYS = {
    "protocol_schema",
    "status",
    "evidence_tier",
    "scope",
    "collection",
    "controller",
    "evaluation",
    "decision_rule",
    "limitations_declared_in_advance",
}
_SECTION_KEYS = {
    "scope": {
        "dataset",
        "dataset_config",
        "dataset_revision",
        "dataset_train_sha256",
        "dataset_test_sha256",
        "model",
        "model_revision",
        "runtime_image",
        "runtime_overlay",
    },
    "collection": {
        "train_records",
        "validation_records",
        "test_records",
        "test_split",
        "samples_per_prompt",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "logprobs",
        "sampling_seed",
        "selection",
        "system_prompt_sha256",
        "parser",
        "completed_output_fallback",
        "truncated_outputs_vote",
    },
    "controller": {
        "algorithm",
        "hidden_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "seed",
        "max_samples",
        "training_costs",
    },
    "evaluation": {
        "primary_costs",
        "reported_costs",
        "objective",
        "bootstrap_unit",
        "bootstrap_resamples",
        "bootstrap_seed",
        "confidence",
        "comparator_selection",
        "fixed_counts",
        "confidence_thresholds",
        "agreement_streaks",
    },
    "decision_rule": {"success", "failure_handling"},
}
_MISSING = object()
MANIFEST_SCHEMA_VERSION = 3
DATA_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str

    @property
    def size(self) -> int:
        return len(self.payload)

    def metadata(self) -> dict[str, str | int]:
        return {"path": self.path.name, "bytes": self.size, "sha256": self.sha256}


def capture_file(path: _PathLike, label: str = "file") -> FileSnapshot:
    """Read one non-symlink regular-file descriptor and hash the exact bytes returned."""
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError(f"{label} is not a readable regular file: {candidate}") from exc
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise ValueError(f"{label} must be a regular file, not a symlink: {candidate}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    return FileSnapshot(candidate, payload, hashlib.sha256(payload).hexdigest())


def _load_object(snapshot: FileSnapshot, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {snapshot.path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {snapshot.path}")
    return payload


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys do not match schema; missing={missing}, extra={extra}")


def _validate_protocol_schema(payload: dict[str, Any]) -> None:
    _require_exact_keys(payload, _PROTOCOL_KEYS, "protocol")
    if type(payload["protocol_schema"]) is not int or payload["protocol_schema"] != 1:
        raise ValueError("protocol.protocol_schema must be integer 1")
    for section, keys in _SECTION_KEYS.items():
        value = payload[section]
        if not isinstance(value, Mapping):
            raise ValueError(f"protocol.{section} must be an object")
        _require_exact_keys(value, keys, f"protocol.{section}")
    limitations = payload["limitations_declared_in_advance"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item for item in limitations)
    ):
        raise ValueError("protocol.limitations_declared_in_advance must be non-empty strings")


def _value_at(root: Mapping[str, Any], dotted_path: str, label: str) -> Any:
    if not dotted_path:
        raise ValueError(f"{label} path must not be empty")
    current: Any = root
    traversed: list[str] = []
    for key in dotted_path.split("."):
        traversed.append(key)
        location = ".".join(traversed)
        if not isinstance(current, Mapping):
            parent = ".".join(traversed[:-1]) or "<root>"
            raise ValueError(
                f"{label}.{parent} must be an object (required by {label}.{dotted_path})"
            )
        if key not in current:
            raise ValueError(f"missing {label}.{location} (required by {label}.{dotted_path})")
        current = current[key]
    return current


def _exact(expected: Any, actual: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _require_type(value: Any, expected_type: type[Any], dotted_path: str) -> None:
    if type(value) is not expected_type:
        raise ValueError(
            f"protocol.{dotted_path} must be {expected_type.__name__}, got {type(value).__name__}"
        )


@dataclass(frozen=True)
class Protocol:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    raw: bytes = b""

    @classmethod
    def load(cls, path: _PathLike) -> Protocol:
        snapshot = capture_file(path, "protocol")
        payload = _load_object(snapshot, "protocol")
        _validate_protocol_schema(payload)
        return cls(
            path=snapshot.path,
            sha256=snapshot.sha256,
            payload=payload,
            raw=snapshot.payload,
        )

    def metadata(self) -> dict[str, str]:
        status = _value_at(self.payload, "status", "protocol")
        evidence_tier = _value_at(self.payload, "evidence_tier", "protocol")
        _require_type(status, str, "status")
        _require_type(evidence_tier, str, "evidence_tier")
        return {
            "path": self.path.name,
            "sha256": self.sha256,
            "status": status,
            "evidence_tier": evidence_tier,
        }

    def require(self, dotted_path: str, actual: Any) -> None:
        expected = _value_at(self.payload, dotted_path, "protocol")
        if not _exact(expected, actual):
            raise ValueError(
                f"protocol mismatch at {dotted_path}: expected {expected!r} "
                f"({type(expected).__name__}), got {actual!r} ({type(actual).__name__})"
            )


def load_manifest(path: _PathLike) -> dict[str, Any]:
    return _load_object(capture_file(path, "manifest"), "manifest")


def validate_manifest_protocol(manifest: Mapping[str, Any], protocol: Protocol) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    if (
        type(manifest.get("manifest_schema")) is not int
        or manifest["manifest_schema"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"manifest.manifest_schema must be integer {MANIFEST_SCHEMA_VERSION}")
    if (
        type(manifest.get("data_schema")) is not int
        or manifest["data_schema"] != DATA_SCHEMA_VERSION
    ):
        raise ValueError(f"manifest.data_schema must be integer {DATA_SCHEMA_VERSION}")
    comparisons = (
        ("dataset.id", "scope.dataset", str),
        ("dataset.config", "scope.dataset_config", str),
        ("dataset.revision", "scope.dataset_revision", str),
        ("dataset.source_files.train.sha256", "scope.dataset_train_sha256", str),
        ("dataset.source_files.test.sha256", "scope.dataset_test_sha256", str),
        ("dataset.selection", "collection.selection", str),
        ("dataset.test_split", "collection.test_split", str),
        ("model.id", "scope.model", str),
        ("model.revision", "scope.model_revision", str),
        ("sampling.samples_per_prompt", "collection.samples_per_prompt", int),
        ("sampling.max_tokens", "collection.max_completion_tokens", int),
        ("sampling.temperature", "collection.temperature", float),
        ("sampling.top_p", "collection.top_p", float),
        ("sampling.logprobs", "collection.logprobs", int),
        ("sampling.seed", "collection.sampling_seed", int),
        ("prompt.sha256", "collection.system_prompt_sha256", str),
        ("prompt.completed_output_fallback", "collection.completed_output_fallback", str),
        ("runtime.image", "scope.runtime_image", str),
        ("runtime.overlay", "scope.runtime_overlay", str),
        ("splits.train.records", "collection.train_records", int),
        ("splits.validation.records", "collection.validation_records", int),
        ("splits.test.records", "collection.test_records", int),
        ("prompt.parser", "collection.parser", str),
        ("prompt.truncated_outputs_vote", "collection.truncated_outputs_vote", bool),
    )
    for manifest_path, protocol_path, expected_type in comparisons:
        expected = _value_at(protocol.payload, protocol_path, "protocol")
        _require_type(expected, expected_type, protocol_path)
        actual = _value_at(manifest, manifest_path, "manifest")
        if type(actual) is not expected_type:
            raise ValueError(
                f"manifest.{manifest_path} must be {expected_type.__name__}, "
                f"got {type(actual).__name__}"
            )
        protocol.require(protocol_path, actual)


def _require_exact_field(
    mapping: Mapping[str, Any], field: str, expected: Any, location: str
) -> None:
    actual = mapping.get(field, _MISSING)
    field_path = f"{location}.{field}"
    if actual is _MISSING:
        raise ValueError(f"missing {field_path}")
    if not _exact(expected, actual):
        raise ValueError(
            f"mismatch at {field_path}: expected {expected!r} "
            f"({type(expected).__name__}), got {actual!r} ({type(actual).__name__})"
        )


def verify_manifest_artifact(
    manifest_path: _PathLike,
    data_path: _PathLike,
    split: str,
    protocol: Protocol,
    *,
    manifest_snapshot: FileSnapshot | None = None,
    data_snapshot: FileSnapshot | None = None,
) -> dict[str, Any]:
    if type(split) is not str or split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported artifact split: {split!r}")
    manifest_snapshot = manifest_snapshot or capture_file(manifest_path, "manifest")
    data_snapshot = data_snapshot or capture_file(data_path, f"{split} artifact")
    if os.path.abspath(manifest_snapshot.path) != os.path.abspath(manifest_path):
        raise ValueError("manifest snapshot does not belong to the requested path")
    if os.path.abspath(data_snapshot.path) != os.path.abspath(data_path):
        raise ValueError(f"{split} snapshot does not belong to the requested path")
    manifest = _load_object(manifest_snapshot, "manifest")
    validate_manifest_protocol(manifest, protocol)

    manifest_protocol = _value_at(manifest, "protocol", "manifest")
    if not isinstance(manifest_protocol, Mapping):
        raise ValueError("manifest.protocol must be an object")
    for field, expected in protocol.metadata().items():
        _require_exact_field(manifest_protocol, field, expected, "manifest.protocol")

    artifacts = _value_at(manifest, "artifacts", "manifest")
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest.artifacts must be an object")
    artifact = _value_at(artifacts, split, "manifest.artifacts")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"manifest.artifacts.{split} must be an object")
    for field, expected in data_snapshot.metadata().items():
        _require_exact_field(artifact, field, expected, f"manifest.artifacts.{split}")
    return manifest
