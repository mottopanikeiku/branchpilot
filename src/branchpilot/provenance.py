from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from branchpilot.artifacts import sha256_file

__all__ = [
    "Protocol",
    "load_manifest",
    "validate_manifest_protocol",
    "verify_manifest_artifact",
]

_PathLike = str | os.PathLike[str]
_MISSING = object()


def _regular_file(path: _PathLike, label: str) -> Path:
    candidate = Path(path)
    try:
        mode = candidate.lstat().st_mode
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file: {candidate}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file and must not be a symlink: {candidate}")
    return candidate


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


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

    @classmethod
    def load(cls, path: _PathLike) -> Protocol:
        protocol_path = _regular_file(path, "protocol")
        payload = _load_object(protocol_path, "protocol")
        return cls(path=protocol_path, sha256=sha256_file(protocol_path), payload=payload)

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
    manifest_path = _regular_file(path, "manifest")
    return _load_object(manifest_path, "manifest")


def validate_manifest_protocol(manifest: Mapping[str, Any], protocol: Protocol) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")

    comparisons = (
        ("dataset.id", "scope.dataset", str),
        ("dataset.config", "scope.dataset_config", str),
        ("dataset.revision", "scope.dataset_revision", str),
        ("dataset.source_files.train.sha256", "scope.dataset_train_sha256", str),
        ("dataset.source_files.test.sha256", "scope.dataset_test_sha256", str),
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
        ("splits.train.records", "collection.train_records", int),
        ("splits.validation.records", "collection.validation_records", int),
        ("splits.test.records", "collection.test_records", int),
        ("prompt.parser", "collection.parser", str),
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
    mapping: Mapping[str, Any],
    field: str,
    expected: Any,
    location: str,
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
) -> dict[str, Any]:
    if type(split) is not str or split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported artifact split: {split!r}")

    manifest = load_manifest(manifest_path)
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

    data_file = _regular_file(data_path, f"{split} artifact")
    _require_exact_field(artifact, "path", data_file.name, f"manifest.artifacts.{split}")
    _require_exact_field(
        artifact,
        "bytes",
        data_file.stat().st_size,
        f"manifest.artifacts.{split}",
    )
    _require_exact_field(
        artifact,
        "sha256",
        sha256_file(data_file),
        f"manifest.artifacts.{split}",
    )
    return manifest
