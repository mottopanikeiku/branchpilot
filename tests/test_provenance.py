import json
from dataclasses import FrozenInstanceError
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from branchpilot.artifacts import sha256_file
from branchpilot.provenance import (
    Protocol,
    load_manifest,
    validate_manifest_protocol,
    verify_manifest_artifact,
)


@pytest.fixture
def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_schema": 1,
        "status": "frozen-before-canonical-generation",
        "evidence_tier": "scoped-single-model-single-task",
        "scope": {
            "dataset": "openai/grade-school-math",
            "dataset_config": "main",
            "dataset_revision": "dataset-revision",
            "dataset_train_sha256": "1" * 64,
            "dataset_test_sha256": "2" * 64,
            "model": "Qwen/model",
            "model_revision": "model-revision",
            "runtime_image": "vllm/image@sha256:" + "3" * 64,
        },
        "collection": {
            "train_records": 1,
            "validation_records": 1,
            "test_records": 1,
            "samples_per_prompt": 8,
            "max_completion_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "logprobs": 1,
            "sampling_seed": 17,
            "system_prompt_sha256": "a" * 64,
            "parser": "branchpilot.numeric-complete-v3",
            "completed_output_fallback": "last-numeric-only-when-finish-reason-is-stop",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _set_dotted(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _load_protocol(tmp_path: Path, payload: dict[str, Any]) -> Protocol:
    path = tmp_path / "protocol.json"
    _write_json(path, payload)
    return Protocol.load(path)


def _manifest(protocol: Protocol, data_path: Path, split: str = "train") -> dict[str, Any]:
    scope = protocol.payload["scope"]
    collection = protocol.payload["collection"]
    return {
        "manifest_schema": 3,
        "dataset": {
            "id": scope["dataset"],
            "config": scope["dataset_config"],
            "revision": scope["dataset_revision"],
            "source_files": {
                "train": {"sha256": scope["dataset_train_sha256"]},
                "test": {"sha256": scope["dataset_test_sha256"]},
            },
        },
        "model": {
            "id": scope["model"],
            "revision": scope["model_revision"],
        },
        "sampling": {
            "samples_per_prompt": collection["samples_per_prompt"],
            "max_tokens": collection["max_completion_tokens"],
            "temperature": collection["temperature"],
            "top_p": collection["top_p"],
            "logprobs": collection["logprobs"],
            "seed": collection["sampling_seed"],
        },
        "prompt": {
            "sha256": collection["system_prompt_sha256"],
            "parser": collection["parser"],
            "completed_output_fallback": collection["completed_output_fallback"],
        },
        "runtime": {"image": scope["runtime_image"]},
        "splits": {
            "train": {"records": collection["train_records"]},
            "validation": {"records": collection["validation_records"]},
            "test": {"records": collection["test_records"]},
        },
        "protocol": protocol.metadata(),
        "artifacts": {
            split: {
                "path": data_path.name,
                "bytes": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
            }
        },
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def test_protocol_load_hashes_exact_bytes_and_serializes_metadata(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    path = tmp_path / "frozen.json"
    _write_json(path, protocol_payload)

    protocol = Protocol.load(path)

    assert protocol.path == path
    assert protocol.sha256 == sha256_file(path)
    assert protocol.metadata() == {
        "path": "frozen.json",
        "sha256": sha256_file(path),
        "status": "frozen-before-canonical-generation",
        "evidence_tier": "scoped-single-model-single-task",
    }
    assert json.loads(json.dumps(protocol.metadata())) == protocol.metadata()
    with pytest.raises(FrozenInstanceError):
        protocol.sha256 = "0" * 64  # type: ignore[misc]


def test_protocol_require_uses_exact_mapping_keys_and_types(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)

    protocol.require("collection.samples_per_prompt", 8)
    with pytest.raises(ValueError, match=r"mismatch at collection\.samples_per_prompt"):
        protocol.require("collection.samples_per_prompt", True)
    with pytest.raises(ValueError, match=r"missing protocol\.collection\.missing"):
        protocol.require("collection.missing", 8)
    with pytest.raises(ValueError, match=r"protocol\.scope must be an object"):
        Protocol(protocol.path, protocol.sha256, {"scope": "gsm8k"}).require(
            "scope.dataset", "openai/gsm8k"
        )


def test_load_manifest_requires_json_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_json(path, ["not", "an", "object"])

    with pytest.raises(ValueError, match="manifest must be a JSON object"):
        load_manifest(path)


def test_valid_manifest_binds_exact_artifact_bytes(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b'{"id":1}\n')
    expected = _manifest(protocol, data_path)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, expected)

    actual = verify_manifest_artifact(manifest_path, data_path, "train", protocol)

    assert actual == expected


def test_one_byte_data_tampering_is_rejected(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"alpha\n")
    manifest = _manifest(protocol, data_path)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    data_path.write_bytes(b"alphb\n")

    with pytest.raises(ValueError, match=r"artifacts\.train\.sha256"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


def test_renamed_artifact_is_rejected_even_when_bytes_match(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    original = tmp_path / "train.jsonl"
    original.write_bytes(b"same bytes\n")
    manifest = _manifest(protocol, original)
    renamed = tmp_path / "renamed.jsonl"
    original.rename(renamed)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=r"artifacts\.train\.path"):
        verify_manifest_artifact(manifest_path, renamed, "train", protocol)


@pytest.mark.parametrize(
    ("field", "wrong"),
    (("bytes", 0), ("sha256", "0" * 64)),
)
def test_wrong_artifact_size_or_hash_is_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    field: str,
    wrong: Any,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    manifest["artifacts"]["train"][field] = wrong
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=rf"artifacts\.train\.{field}"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


@pytest.mark.parametrize("field", ("path", "sha256", "status", "evidence_tier"))
def test_wrong_protocol_binding_is_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    field: str,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    manifest["protocol"][field] = "wrong"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=rf"manifest\.protocol\.{field}"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


@pytest.mark.parametrize("split", ("training", "", 1, True, None))
def test_unsupported_split_is_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    split: Any,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(protocol, data_path))

    with pytest.raises(ValueError, match="unsupported artifact split"):
        verify_manifest_artifact(manifest_path, data_path, split, protocol)


def test_missing_allowed_split_is_rejected(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "test.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(protocol, data_path, "train"))

    with pytest.raises(ValueError, match=r"missing manifest\.artifacts\.test"):
        verify_manifest_artifact(manifest_path, data_path, "test", protocol)


@pytest.mark.parametrize(
    ("manifest_path", "protocol_path", "wrong"),
    (
        ("dataset.id", "scope.dataset", "other/dataset"),
        ("dataset.config", "scope.dataset_config", "other"),
        ("dataset.revision", "scope.dataset_revision", "other-revision"),
        (
            "dataset.source_files.train.sha256",
            "scope.dataset_train_sha256",
            "4" * 64,
        ),
        (
            "dataset.source_files.test.sha256",
            "scope.dataset_test_sha256",
            "5" * 64,
        ),
        ("model.id", "scope.model", "other/model"),
        ("model.revision", "scope.model_revision", "other-revision"),
        ("sampling.samples_per_prompt", "collection.samples_per_prompt", 7),
        ("sampling.max_tokens", "collection.max_completion_tokens", 256),
        ("sampling.temperature", "collection.temperature", 0.7000000000000001),
        ("sampling.top_p", "collection.top_p", 0.9500000000000001),
        ("sampling.logprobs", "collection.logprobs", 2),
        ("sampling.seed", "collection.sampling_seed", 18),
        ("prompt.sha256", "collection.system_prompt_sha256", "b" * 64),
        ("prompt.parser", "collection.parser", "other-parser"),
        (
            "prompt.completed_output_fallback",
            "collection.completed_output_fallback",
            "unsafe-fallback",
        ),
        ("runtime.image", "scope.runtime_image", "mutable-image"),
        ("splits.train.records", "collection.train_records", 2),
        ("splits.validation.records", "collection.validation_records", 2),
        ("splits.test.records", "collection.test_records", 2),
    ),
)
def test_manifest_protocol_drift_is_rejected_at_every_compared_field(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    manifest_path: str,
    protocol_path: str,
    wrong: Any,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    _set_dotted(manifest, manifest_path, wrong)

    with pytest.raises(ValueError, match=protocol_path.replace(".", r"\.")):
        validate_manifest_protocol(manifest, protocol)


def test_float_comparisons_are_exact_and_do_not_coerce_ints(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    manifest["sampling"]["temperature"] = 0.7000000000000001

    with pytest.raises(ValueError, match=r"collection\.temperature"):
        validate_manifest_protocol(manifest, protocol)

    manifest["sampling"]["temperature"] = 0.7
    manifest["sampling"]["top_p"] = 1
    protocol.payload["collection"]["top_p"] = 1.0
    with pytest.raises(ValueError, match=r"manifest\.sampling\.top_p must be float"):
        validate_manifest_protocol(manifest, protocol)


@pytest.mark.parametrize(
    "field",
    ("samples_per_prompt", "max_tokens", "logprobs", "seed"),
)
def test_booleans_are_rejected_for_manifest_integer_fields(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    field: str,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    manifest["sampling"][field] = True

    with pytest.raises(ValueError, match=rf"manifest\.sampling\.{field} must be int"):
        validate_manifest_protocol(manifest, protocol)


def test_boolean_is_rejected_for_artifact_byte_size(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"x")
    manifest = _manifest(protocol, data_path)
    manifest["artifacts"]["train"]["bytes"] = True
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=r"artifacts\.train\.bytes"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


@pytest.mark.parametrize("section", ("dataset", "model", "sampling", "prompt"))
@pytest.mark.parametrize("replacement", (None, "not-an-object", []))
def test_missing_or_non_object_manifest_sections_are_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    section: str,
    replacement: Any,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    if replacement is None:
        del manifest[section]
    else:
        manifest[section] = replacement

    with pytest.raises(ValueError, match=rf"manifest\.{section}"):
        validate_manifest_protocol(manifest, protocol)


@pytest.mark.parametrize("section", ("protocol", "artifacts"))
@pytest.mark.parametrize("replacement", (None, "not-an-object", []))
def test_missing_or_non_object_binding_sections_are_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    section: str,
    replacement: Any,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    if replacement is None:
        del manifest[section]
    else:
        manifest[section] = replacement
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=rf"manifest\.{section}"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


@pytest.mark.parametrize("field", ("status", "evidence_tier"))
def test_protocol_metadata_requires_string_fields(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    field: str,
) -> None:
    protocol_payload[field] = None
    protocol = _load_protocol(tmp_path, protocol_payload)

    with pytest.raises(ValueError, match=rf"protocol\.{field} must be str"):
        protocol.metadata()


def test_missing_protocol_comparison_field_is_rejected(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    del protocol.payload["collection"]["parser"]

    with pytest.raises(ValueError, match=r"missing protocol\.collection\.parser"):
        validate_manifest_protocol(manifest, protocol)


@pytest.mark.parametrize("kind", ("protocol", "manifest", "data"))
def test_symlink_inputs_are_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    kind: str,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(protocol, data_path))

    if kind == "protocol":
        link = tmp_path / "protocol-link.json"
        link.symlink_to(protocol.path)
        call = partial(Protocol.load, link)
    elif kind == "manifest":
        link = tmp_path / "manifest-link.json"
        link.symlink_to(manifest_path)
        call = partial(verify_manifest_artifact, link, data_path, "train", protocol)
    else:
        link = tmp_path / "data-link.jsonl"
        link.symlink_to(data_path)
        call = partial(verify_manifest_artifact, manifest_path, link, "train", protocol)

    with pytest.raises(ValueError, match="regular file"):
        call()


@pytest.mark.parametrize("kind", ("protocol", "manifest", "data"))
def test_non_regular_inputs_are_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    kind: str,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(protocol, data_path))
    directory = tmp_path / "directory"
    directory.mkdir()

    if kind == "protocol":
        call = partial(Protocol.load, directory)
    elif kind == "manifest":
        call = partial(verify_manifest_artifact, directory, data_path, "train", protocol)
    else:
        call = partial(verify_manifest_artifact, manifest_path, directory, "train", protocol)

    with pytest.raises(ValueError, match="regular file"):
        call()


def test_protocol_load_requires_json_object(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    _write_json(path, ["not", "an", "object"])

    with pytest.raises(ValueError, match="protocol must be a JSON object"):
        Protocol.load(path)


@pytest.mark.parametrize("field", ("path", "bytes", "sha256"))
def test_missing_artifact_binding_field_is_rejected(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    field: str,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    del manifest["artifacts"]["train"][field]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=rf"missing manifest\.artifacts\.train\.{field}"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


def test_non_object_artifact_binding_is_rejected(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    manifest["artifacts"]["train"] = []
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=r"manifest\.artifacts\.train must be an object"):
        verify_manifest_artifact(manifest_path, data_path, "train", protocol)


def test_non_object_protocol_section_is_rejected(
    tmp_path: Path, protocol_payload: dict[str, Any]
) -> None:
    protocol_payload["scope"] = "not-an-object"
    protocol = _load_protocol(tmp_path, protocol_payload)

    with pytest.raises(ValueError, match=r"protocol\.scope must be an object"):
        validate_manifest_protocol({}, protocol)


@pytest.mark.parametrize(
    "field",
    ("samples_per_prompt", "max_completion_tokens", "logprobs", "sampling_seed"),
)
def test_booleans_are_rejected_for_protocol_integer_fields(
    tmp_path: Path,
    protocol_payload: dict[str, Any],
    field: str,
) -> None:
    protocol = _load_protocol(tmp_path, protocol_payload)
    data_path = tmp_path / "train.jsonl"
    data_path.write_bytes(b"artifact\n")
    manifest = _manifest(protocol, data_path)
    protocol.payload["collection"][field] = True

    with pytest.raises(ValueError, match=rf"protocol\.collection\.{field} must be int"):
        validate_manifest_protocol(manifest, protocol)
