import json
from argparse import Namespace
from pathlib import Path

from branchpilot.artifacts import atomic_write_text, sha256_file
from branchpilot.cli import command_evaluate, command_train
from branchpilot.evaluate import AGREEMENT_STREAKS, CONFIDENCE_THRESHOLDS
from branchpilot.provenance import Protocol
from branchpilot.schema import write_jsonl
from branchpilot.synthetic import make_synthetic_rollouts


def _protocol(path: Path) -> Protocol:
    payload = {
        "protocol_schema": 1,
        "status": "frozen-before-canonical-generation",
        "evidence_tier": "test-fixture",
        "scope": {
            "dataset": "fixture/dataset",
            "dataset_config": "main",
            "dataset_revision": "fixture-revision",
            "dataset_train_sha256": "1" * 64,
            "dataset_test_sha256": "2" * 64,
            "model": "fixture/model",
            "model_revision": "fixture-model-revision",
            "runtime_image": "fixture/image@sha256:" + "3" * 64,
        },
        "collection": {
            "train_records": 16,
            "validation_records": 8,
            "test_records": 8,
            "samples_per_prompt": 3,
            "max_completion_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "logprobs": 1,
            "sampling_seed": 17,
            "system_prompt_sha256": "4" * 64,
            "parser": "fixture-parser",
            "completed_output_fallback": "finish-reason-stop-only",
        },
        "controller": {
            "algorithm": "exact-backward-q-regression-v1",
            "hidden_size": 16,
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.0003,
            "seed": 7,
            "max_samples": 3,
            "training_costs": [0.0, 0.05, 0.1, 0.2],
        },
        "evaluation": {
            "primary_costs": [0.05, 0.1, 0.2],
            "reported_costs": [0.05, 0.1, 0.2],
            "bootstrap_resamples": 50,
            "bootstrap_seed": 17,
            "fixed_counts": [1, 2, 3],
            "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
            "agreement_streaks": list(AGREEMENT_STREAKS),
        },
        "decision_rule": {
            "success": (
                "paired utility interval lower bound above zero at two or more primary costs "
                "and nonnegative at the third"
            )
        },
        "limitations_declared_in_advance": ["fixture limitation"],
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return Protocol.load(path)


def _artifact(path: Path) -> dict[str, str | int]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _manifest(path: Path, protocol: Protocol, artifacts: dict[str, Path]) -> None:
    scope = protocol.payload["scope"]
    collection = protocol.payload["collection"]
    payload = {
        "manifest_schema": 3,
        "data_schema": 2,
        "protocol": protocol.metadata(),
        "dataset": {
            "id": scope["dataset"],
            "config": scope["dataset_config"],
            "revision": scope["dataset_revision"],
            "source_files": {
                "train": {"sha256": scope["dataset_train_sha256"]},
                "test": {"sha256": scope["dataset_test_sha256"]},
            },
        },
        "model": {"id": scope["model"], "revision": scope["model_revision"]},
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
        "artifacts": {split: _artifact(data_path) for split, data_path in artifacts.items()},
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_hash_bound_validation_freeze_and_test_evaluation(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol(protocol_path)
    artifacts = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    write_jsonl(artifacts["train"], make_synthetic_rollouts(16, max_samples=3, seed=101))
    write_jsonl(artifacts["validation"], make_synthetic_rollouts(8, max_samples=3, seed=102))
    write_jsonl(artifacts["test"], make_synthetic_rollouts(8, max_samples=3, seed=103))
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, protocol, artifacts)

    policy_path = tmp_path / "policy.safetensors"
    command_train(
        Namespace(
            data=str(artifacts["train"]),
            output=str(policy_path),
            max_samples=3,
            hidden_size=16,
            epochs=2,
            batch_size=32,
            learning_rate=0.0003,
            seed=7,
            costs=(0.0, 0.05, 0.1, 0.2),
            protocol=str(protocol_path),
            manifest=str(manifest_path),
        )
    )

    validation_benchmark = tmp_path / "validation-benchmark.json"
    selection_path = tmp_path / "frozen-baselines.json"
    command_evaluate(
        Namespace(
            data=str(artifacts["validation"]),
            policy=str(policy_path),
            output=str(validation_benchmark),
            svg=None,
            html=None,
            costs=(0.05, 0.1, 0.2),
            bootstrap_samples=50,
            bootstrap_seed=17,
            frozen_baselines=None,
            protocol=str(protocol_path),
            manifest=str(manifest_path),
            split="validation",
            export_baselines=str(selection_path),
        )
    )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selection"] == "observed-best-on-validation"
    assert selection["validation_benchmark"] == _artifact(validation_benchmark)

    test_benchmark = tmp_path / "test-benchmark.json"
    report_path = tmp_path / "report.html"
    command_evaluate(
        Namespace(
            data=str(artifacts["test"]),
            policy=str(policy_path),
            output=str(test_benchmark),
            svg=None,
            html=str(report_path),
            costs=(0.05, 0.1, 0.2),
            bootstrap_samples=50,
            bootstrap_seed=17,
            frozen_baselines=str(selection_path),
            protocol=str(protocol_path),
            manifest=str(manifest_path),
            split="test",
            export_baselines=None,
        )
    )

    benchmark = json.loads(test_benchmark.read_text(encoding="utf-8"))
    assert {comparison["selection"] for comparison in benchmark["comparisons"]} == {
        "validation-frozen"
    }
    assert benchmark["protocol_result"]["eligible"] is True
    assert benchmark["protocol_result"]["status"] in {"pass", "fail"}
    assert benchmark["baseline_selection"]["artifact"] == _artifact(selection_path)
    assert report_path.is_file()
