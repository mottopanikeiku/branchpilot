from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from itertools import combinations
from pathlib import Path

from rich.console import Console
from rich.table import Table

from branchpilot.artifacts import atomic_write_bytes, atomic_write_text, paths_alias
from branchpilot.calibration import load_deployment_plan
from branchpilot.evaluate import (
    AGREEMENT_STREAKS,
    BOOTSTRAP_CONFIDENCE,
    CONFIDENCE_THRESHOLDS,
    benchmark,
    decision_trace,
    pareto_frontier,
)
from branchpilot.features import FEATURE_NAMES
from branchpilot.integrity import (
    profile_rollouts,
    validate_disjoint,
    validate_unique,
)
from branchpilot.policy import (
    ARTIFACT_VERSION,
    COST_MODEL,
    TRAINING_ALGORITHM,
    BranchPilotPolicy,
)
from branchpilot.provenance import (
    FileSnapshot,
    Protocol,
    capture_file,
    verify_manifest_artifact,
)
from branchpilot.report import render_svg, write_report
from branchpilot.schema import SCHEMA_VERSION, read_jsonl, read_jsonl_bytes, write_jsonl
from branchpilot.synthetic import make_synthetic_rollouts

_SUCCESS_RULE = (
    "paired utility interval lower bound above zero at two or more primary costs "
    "and nonnegative at the third"
)
_OBJECTIVE = "accuracy - lambda * (samples - 1)"
_BOOTSTRAP_UNIT = "prompt"
_COMPARATOR_SELECTION = "best-validation-utility-per-cost-frozen-before-test"
_FAILURE_HANDLING = (
    "publish every result unchanged; do not retune, change seeds, or regenerate the canonical bank"
)
console = Console()


def _assert_distinct_paths(**paths: str | Path | None) -> None:
    present = [(label, Path(path)) for label, path in paths.items() if path is not None]
    for (left_label, left), (right_label, right) in combinations(present, 2):
        if paths_alias(left, right):
            raise ValueError(f"{left_label} and {right_label} must be distinct paths: {left}")


def _training_api():
    try:
        from branchpilot.training import TrainConfig, train_policy
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Training requires PyTorch. Install BranchPilot with the 'train' extra."
            ) from exc
        raise
    return TrainConfig, train_policy


def _costs(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("costs must be comma-separated numbers") from exc
    if not parsed or any(not math.isfinite(cost) or cost < 0 for cost in parsed):
        raise argparse.ArgumentTypeError("costs must be non-empty, finite, and non-negative")
    return parsed


def _cost_key(cost: float) -> str:
    return repr(float(cost))


def _capture_file(path: str | Path, label: str) -> FileSnapshot:
    return capture_file(path, label)


def _snapshot_file(path: str | Path, label: str) -> dict[str, str | int]:
    return _capture_file(path, label).metadata()


def _load_policy_snapshot(snapshot: FileSnapshot) -> BranchPilotPolicy:
    with tempfile.TemporaryDirectory(prefix="branchpilot-policy-") as directory:
        private_path = Path(directory) / snapshot.path.name
        atomic_write_bytes(private_path, snapshot.payload)
        return BranchPilotPolicy.load(private_path)


def _verify_unchanged(
    path: str | Path,
    expected: dict[str, str | int],
    label: str,
) -> None:
    actual = _snapshot_file(path, label)
    if actual != expected:
        raise RuntimeError(f"{label} changed while it was being used")


def _validate_policy_protocol(
    policy: BranchPilotPolicy,
    protocol: Protocol,
    manifest_snapshot: dict[str, str | int],
    manifest: dict,
) -> None:
    training = policy.training
    config = training.get("config")
    if not isinstance(config, dict):
        raise ValueError("policy training metadata is missing its config object")
    requirements = {
        "controller.algorithm": training.get("algorithm"),
        "controller.max_samples": config.get("max_samples"),
        "controller.hidden_size": config.get("hidden_size"),
        "controller.epochs": config.get("epochs"),
        "controller.batch_size": config.get("batch_size"),
        "controller.learning_rate": config.get("learning_rate"),
        "controller.seed": config.get("seed"),
        "controller.training_costs": config.get("costs"),
    }
    for dotted_path, actual in requirements.items():
        protocol.require(dotted_path, actual)
    provenance = training.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("policy training metadata is missing provenance")
    if provenance.get("protocol") != protocol.metadata():
        raise ValueError("policy was not trained under this frozen protocol")
    if provenance.get("manifest") != manifest_snapshot:
        raise ValueError("policy was not trained from this generation manifest")
    if provenance.get("data") != manifest["artifacts"]["train"]:
        raise ValueError("policy was not trained from this manifest's train artifact")


def _protocol_result(protocol: Protocol, result, split: str | None) -> dict:
    protocol.require("decision_rule.success", _SUCCESS_RULE)
    protocol.require("decision_rule.failure_handling", _FAILURE_HANDLING)
    evaluation = protocol.payload.get("evaluation")
    limitations = protocol.payload.get("limitations_declared_in_advance")
    if not isinstance(evaluation, dict) or not isinstance(limitations, list):
        raise ValueError("protocol is missing evaluation or limitation declarations")
    primary_costs = evaluation.get("primary_costs")
    if not isinstance(primary_costs, list) or len(primary_costs) != 3:
        raise ValueError("protocol evaluation.primary_costs must contain exactly three costs")
    comparisons = {comparison.scoring_cost: comparison for comparison in result.comparisons}
    if any(cost not in comparisons for cost in primary_costs):
        raise ValueError("protocol primary costs are missing from benchmark comparisons")
    lower_bounds = {
        _cost_key(cost): comparisons[cost].utility_delta_interval.lower for cost in primary_costs
    }
    eligible = split == "test" and all(
        comparisons[cost].selection == "validation-frozen" for cost in primary_costs
    )
    positive = sum(value > 0 for value in lower_bounds.values())
    passed = eligible and positive >= 2 and all(value >= 0 for value in lower_bounds.values())
    return {
        "eligible": eligible,
        "passed": passed,
        "status": "pass" if passed else ("fail" if eligible else "not-applicable"),
        "rule": _SUCCESS_RULE,
        "primary_costs": primary_costs,
        "utility_delta_lower_bounds": lower_bounds,
        "limitations": limitations,
    }


def _load_frozen_baselines(
    path: str | Path,
    *,
    costs: tuple[float, ...],
    policy_capture: FileSnapshot,
    protocol: Protocol,
) -> tuple[
    dict[float, str],
    FileSnapshot,
    FileSnapshot,
    FileSnapshot,
    FileSnapshot,
    dict,
]:
    selection_path = Path(path)
    selection_capture = _capture_file(selection_path, "baseline selection")
    payload = json.loads(selection_capture.payload.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("baseline selection must be a schema-version 1 JSON object")
    if payload.get("selection") != "observed-best-on-validation":
        raise ValueError("baseline selection has an unsupported selection method")
    if payload.get("protocol") != protocol.metadata():
        raise ValueError("baseline selection is not bound to this frozen protocol")
    policy_snapshot = policy_capture.metadata()
    if payload.get("policy") != policy_snapshot:
        raise ValueError("baseline selection is not bound to this policy artifact")
    if payload.get("costs") != list(costs):
        raise ValueError("baseline selection costs do not match evaluation costs")

    def capture_binding(field: str, label: str) -> tuple[dict, FileSnapshot]:
        binding = payload.get(field)
        if not isinstance(binding, dict):
            raise ValueError(f"baseline selection is missing {field}")
        relative_path = binding.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"baseline selection {field} path is invalid")
        captured = _capture_file(selection_path.parent / relative_path, label)
        if any(binding.get(key) != captured.metadata()[key] for key in ("bytes", "sha256")):
            raise ValueError(f"baseline selection {field} binding is invalid")
        return binding, captured

    source, source_capture = capture_binding("validation_benchmark", "validation benchmark")
    validation_data, validation_data_capture = capture_binding("validation_data", "validation data")
    validation_manifest, validation_manifest_capture = capture_binding(
        "validation_manifest", "validation manifest"
    )
    manifest = verify_manifest_artifact(
        validation_manifest_capture.path,
        validation_data_capture.path,
        "validation",
        protocol,
        manifest_snapshot=validation_manifest_capture,
        data_snapshot=validation_data_capture,
    )
    validation_payload = json.loads(source_capture.payload.decode("utf-8"))
    if not isinstance(validation_payload, dict):
        raise ValueError("validation benchmark must be a JSON object")
    if validation_payload.get("protocol") != protocol.metadata():
        raise ValueError("validation benchmark protocol binding is invalid")
    validation_policy = validation_payload.get("policy")
    if not isinstance(validation_policy, dict) or any(
        validation_policy.get(key) != value for key, value in policy_snapshot.items()
    ):
        raise ValueError("validation benchmark policy binding is invalid")
    validation_payload_data = validation_payload.get("data")
    if not isinstance(validation_payload_data, dict) or any(
        validation_payload_data.get(key) != validation_data_capture.metadata()[key]
        for key in ("bytes", "sha256")
    ):
        raise ValueError("validation benchmark data binding is invalid")
    if validation_payload_data.get("path") != validation_data_capture.path.name:
        raise ValueError("validation benchmark data path is invalid")
    if validation_payload_data.get("manifest") != validation_manifest_capture.metadata():
        raise ValueError("validation benchmark generation manifest binding is invalid")

    validation_rollouts = read_jsonl_bytes(
        validation_data_capture.payload,
        str(validation_data_capture.path),
    )
    validate_unique(validation_rollouts)
    protocol.require("collection.validation_records", len(validation_rollouts))
    validation_policy_object = _load_policy_snapshot(policy_capture)
    _validate_policy_protocol(
        validation_policy_object,
        protocol,
        validation_manifest_capture.metadata(),
        manifest,
    )
    evaluation = protocol.payload["evaluation"]
    recomputed = benchmark(
        validation_rollouts,
        validation_policy_object,
        costs,
        bootstrap_samples=evaluation["bootstrap_resamples"],
        bootstrap_seed=evaluation["bootstrap_seed"],
    ).to_dict()
    for field, value in recomputed.items():
        if validation_payload.get(field) != value:
            raise ValueError(
                f"validation benchmark field {field!r} does not match replayed evidence"
            )
    render_svg(validation_payload)

    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, dict) or not raw_baselines:
        raise ValueError("baseline selection must contain a non-empty baselines object")
    expected_baselines = {
        _cost_key(comparison["scoring_cost"]): comparison["baseline_policy"]
        for comparison in recomputed["comparisons"]
    }
    if raw_baselines != expected_baselines:
        raise ValueError("baseline names do not match replayed validation selection")
    baselines = {float(cost): str(policy) for cost, policy in raw_baselines.items()}
    if set(baselines) != set(costs) or any(not policy for policy in baselines.values()):
        raise ValueError("baseline selection must name one policy for every evaluation cost")
    return (
        baselines,
        selection_capture,
        source_capture,
        validation_data_capture,
        validation_manifest_capture,
        payload,
    )


def _write_benchmark(
    data_path: Path,
    policy_path: Path,
    output: Path,
    costs: tuple[float, ...],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
    frozen_baseline_path: Path | None = None,
    protocol: Protocol | None = None,
    manifest_path: Path | None = None,
    split: str | None = None,
) -> dict:
    data_capture = _capture_file(data_path, "evaluation data")
    data_snapshot = data_capture.metadata()
    policy_capture = _capture_file(policy_path, "policy artifact")
    policy_snapshot = policy_capture.metadata()
    manifest_capture = None
    manifest_snapshot = None
    if protocol is None:
        if manifest_path is not None or split is not None or frozen_baseline_path is not None:
            raise ValueError("manifest, split, and frozen baselines require a frozen protocol")
        frozen_baselines = None
        selection_snapshot = None
        selection_payload = None
        selection_capture = None
        validation_benchmark_capture = None
        validation_data_capture = None
        validation_manifest_capture = None
        validation_source = None
        manifest = None
    else:
        if manifest_path is None or split not in {"validation", "test"}:
            raise ValueError("protocol evaluation requires --manifest and --split validation|test")
        manifest_capture = _capture_file(manifest_path, "generation manifest")
        manifest_snapshot = manifest_capture.metadata()
        manifest = verify_manifest_artifact(
            manifest_path,
            data_path,
            split,
            protocol,
            manifest_snapshot=manifest_capture,
            data_snapshot=data_capture,
        )
        protocol.require("evaluation.reported_costs", list(costs))
        protocol.require("evaluation.bootstrap_resamples", bootstrap_samples)
        protocol.require("evaluation.bootstrap_seed", bootstrap_seed)
        protocol.require("evaluation.objective", _OBJECTIVE)
        protocol.require("evaluation.bootstrap_unit", _BOOTSTRAP_UNIT)
        protocol.require("evaluation.confidence", BOOTSTRAP_CONFIDENCE)
        protocol.require("evaluation.comparator_selection", _COMPARATOR_SELECTION)
        if frozen_baseline_path is None:
            if split == "test":
                raise ValueError("test evaluation requires a bound validation baseline selection")
            frozen_baselines = None
            selection_snapshot = None
            selection_payload = None
            selection_capture = None
            validation_benchmark_capture = None
            validation_data_capture = None
            validation_manifest_capture = None
            validation_source = None
        else:
            if split != "test":
                raise ValueError("frozen baselines may only be consumed on the test split")
            (
                frozen_baselines,
                selection_capture,
                validation_benchmark_capture,
                validation_data_capture,
                validation_manifest_capture,
                selection_payload,
            ) = _load_frozen_baselines(
                frozen_baseline_path,
                costs=costs,
                policy_capture=policy_capture,
                protocol=protocol,
            )
            selection_snapshot = selection_capture.metadata()
            if selection_payload["validation_data"]["sha256"] == data_snapshot["sha256"]:
                raise ValueError("validation and test data must have different SHA-256 values")
            validation_source = validation_benchmark_capture.path
            if paths_alias(output, validation_source):
                raise ValueError("test benchmark output cannot overwrite validation benchmark")

    rollouts = read_jsonl_bytes(data_capture.payload, str(data_path))
    validate_unique(rollouts)
    if protocol is not None:
        protocol.require(f"collection.{split}_records", len(rollouts))
    policy = _load_policy_snapshot(policy_capture)
    if protocol is not None:
        _validate_policy_protocol(policy, protocol, manifest_snapshot, manifest)
        protocol.require(
            "evaluation.fixed_counts",
            list(range(1, policy.max_samples + 1)),
        )
        protocol.require(
            "evaluation.confidence_thresholds",
            list(CONFIDENCE_THRESHOLDS),
        )
        protocol.require(
            "evaluation.agreement_streaks",
            list(AGREEMENT_STREAKS),
        )
    result = benchmark(
        rollouts,
        policy,
        costs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        frozen_baselines=frozen_baselines,
    )
    protocol_result = None if protocol is None else _protocol_result(protocol, result, split)
    _verify_unchanged(data_path, data_snapshot, "evaluation data")
    _verify_unchanged(policy_path, policy_snapshot, "policy artifact")
    if protocol is not None:
        if Protocol.load(protocol.path).metadata() != protocol.metadata():
            raise RuntimeError("frozen protocol changed during evaluation")
        _verify_unchanged(manifest_path, manifest_snapshot, "generation manifest")
        if frozen_baseline_path is not None:
            _verify_unchanged(
                frozen_baseline_path,
                selection_capture.metadata(),
                "baseline selection",
            )
            _verify_unchanged(
                validation_benchmark_capture.path,
                validation_benchmark_capture.metadata(),
                "validation benchmark",
            )
            _verify_unchanged(
                validation_data_capture.path,
                validation_data_capture.metadata(),
                "validation data",
            )
            _verify_unchanged(
                validation_manifest_capture.path,
                validation_manifest_capture.metadata(),
                "validation manifest",
            )

    payload = {
        "schema_version": 2,
        **result.to_dict(),
        "objective": {
            "name": "additional-sample utility",
            "formula": "accuracy - lambda * (samples - 1)",
            "cost_unit": "additional_samples",
        },
        "protocol": None if protocol is None else protocol.metadata(),
        "protocol_raw": None if protocol is None else protocol.raw.decode("utf-8"),
        "protocol_snapshot": (
            None if protocol is None else json.loads(json.dumps(protocol.payload, sort_keys=True))
        ),
        "protocol_result": protocol_result,
        "data": {
            **data_snapshot,
            "schema_version": SCHEMA_VERSION,
            "profile": profile_rollouts(rollouts).to_dict(),
            "split": split,
            "manifest": manifest_snapshot,
        },
        "policy": {
            **policy_snapshot,
            "artifact_version": ARTIFACT_VERSION,
            "cost_model": COST_MODEL,
            "feature_names": list(FEATURE_NAMES),
            "training": policy.training,
        },
        "baseline_selection": (
            None
            if selection_payload is None
            else {
                "artifact": selection_snapshot,
                "validation_data": selection_payload["validation_data"],
                "validation_benchmark": selection_payload["validation_benchmark"],
            }
        ),
        "pareto_frontier": [row.to_dict() for row in pareto_frontier(result)],
    }
    render_svg(payload)
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _print_benchmark(payload: dict) -> None:
    table = Table(title="Paired accuracy–compute evaluation", header_style="bold magenta")
    table.add_column("λ", justify="right")
    table.add_column("policy")
    table.add_column("accuracy", justify="right")
    table.add_column("samples", justify="right")
    table.add_column("Δ utility [95% CI]", justify="right")
    comparisons = {row["scoring_cost"]: row for row in payload["comparisons"]}
    rows = payload["rows"]
    for cost in payload["costs"]:
        comparison = comparisons[cost]
        at_cost = [row for row in rows if abs(row["scoring_cost"] - cost) < 1e-9]
        learned = next(row for row in at_cost if row["policy"] == comparison["learned_policy"])
        baseline = next(row for row in at_cost if row["policy"] == comparison["baseline_policy"])
        delta = comparison["utility_delta"]
        interval = comparison["utility_delta_interval"]
        table.add_row(
            f"{cost:g}",
            learned["policy"],
            f"{learned['accuracy']:.1%}",
            f"{learned['average_samples']:.2f}",
            f"{delta:+.3f} [{interval['lower']:+.3f}, {interval['upper']:+.3f}]",
            style="bold magenta",
        )
        table.add_row(
            "",
            f"↳ {baseline['policy']} · {comparison['selection']}",
            f"{baseline['accuracy']:.1%}",
            f"{baseline['average_samples']:.2f}",
            "reference",
            style="dim",
        )
    console.print(table)


def command_synthetic(args: argparse.Namespace) -> None:
    train = make_synthetic_rollouts(args.train_size, args.max_samples, args.seed)
    test = make_synthetic_rollouts(args.test_size, args.max_samples, args.seed + 1)
    output = Path(args.output_dir)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "test.jsonl", test)
    console.print(
        f"Wrote [bold]{len(train)}[/] train and [bold]{len(test)}[/] test trajectories to {output}"
    )


def command_split(args: argparse.Namespace) -> None:
    train_path = Path(args.train_output)
    test_path = Path(args.test_output)
    _assert_distinct_paths(
        data=args.data,
        train_output=train_path,
        test_output=test_path,
        manifest=args.manifest,
    )
    source_capture = _capture_file(args.data, "split source")
    source_snapshot = source_capture.metadata()
    records = read_jsonl_bytes(source_capture.payload, str(args.data))
    validate_unique(records)
    source_profile = profile_rollouts(records)
    requested = args.train_size + args.test_size
    if args.train_size < 1 or args.test_size < 1 or requested > len(records):
        raise ValueError(
            f"requested positive splits totaling {requested} from {len(records)} records"
        )
    random.Random(args.seed).shuffle(records)
    train_records = records[: args.train_size]
    test_records = records[args.train_size : requested]
    validate_disjoint(train_records, test_records)
    _verify_unchanged(args.data, source_snapshot, "split source")
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)
    if args.manifest:
        manifest = {
            "schema_version": 1,
            "seed": args.seed,
            "source": {
                **source_profile.to_dict(),
                **source_snapshot,
            },
            "train": {
                **profile_rollouts(train_records).to_dict(),
                **_snapshot_file(train_path, "train split"),
            },
            "test": {
                **profile_rollouts(test_records).to_dict(),
                **_snapshot_file(test_path, "test split"),
            },
        }
        atomic_write_text(
            args.manifest,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    console.print(
        f"Wrote [bold]{args.train_size}[/] train and [bold]{args.test_size}[/] "
        f"verified-disjoint trajectories with seed {args.seed}"
    )


def command_audit(args: argparse.Namespace) -> None:
    _assert_distinct_paths(
        data=args.data,
        comparison=args.compare,
        json_output=args.json_output,
    )
    data_capture = _capture_file(args.data, "audit data")
    records = read_jsonl_bytes(data_capture.payload, str(args.data))
    validate_unique(records)
    profile = profile_rollouts(records)
    payload: dict = {"data": str(args.data), "profile": profile.to_dict()}
    if args.compare:
        comparison_capture = _capture_file(args.compare, "audit comparison")
        comparison = read_jsonl_bytes(comparison_capture.payload, str(args.compare))
        validate_unique(comparison)
        validate_disjoint(records, comparison)
        payload["comparison"] = {
            "data": str(args.compare),
            "profile": profile_rollouts(comparison).to_dict(),
            "disjoint": True,
        }
    table = Table(title="Trajectory integrity audit", header_style="bold cyan")
    table.add_column("measure")
    table.add_column("value", justify="right")
    for label, value in (
        ("records", profile.record_count),
        ("samples", profile.sample_count),
        ("horizon", f"{profile.horizon_min} / {profile.horizon_median:g} / {profile.horizon_max}"),
        ("unique prompts", f"{profile.prompt_uniqueness:.1%}"),
        ("parsed samples", f"{profile.parse_rate:.1%}"),
        ("logprob coverage", f"{profile.logprob_coverage:.1%}"),
        ("completion tokens", f"{profile.completion_token_total:,}"),
        ("dataset SHA-256", profile.dataset_fingerprint),
    ):
        table.add_row(label, str(value))
    console.print(table)
    if args.json_output:
        atomic_write_text(
            args.json_output,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        console.print(f"Wrote audit record to [cyan]{args.json_output}[/]")


def command_train(args: argparse.Namespace) -> None:
    _assert_distinct_paths(
        data=args.data,
        output=args.output,
        protocol=args.protocol,
        manifest=args.manifest,
    )
    data_capture = _capture_file(args.data, "training data")
    data_snapshot = data_capture.metadata()
    protocol = None if args.protocol is None else Protocol.load(args.protocol)
    manifest_capture = None
    manifest_snapshot = None
    manifest = None
    if protocol is None:
        if args.manifest is not None:
            raise ValueError("--manifest requires --protocol")
    else:
        if args.manifest is None:
            raise ValueError("canonical training requires --manifest")
        manifest_capture = _capture_file(args.manifest, "generation manifest")
        manifest_snapshot = manifest_capture.metadata()
        manifest = verify_manifest_artifact(
            args.manifest,
            args.data,
            "train",
            protocol,
            manifest_snapshot=manifest_capture,
            data_snapshot=data_capture,
        )
        requirements = {
            "controller.algorithm": TRAINING_ALGORITHM,
            "controller.max_samples": args.max_samples,
            "controller.hidden_size": args.hidden_size,
            "controller.epochs": args.epochs,
            "controller.batch_size": args.batch_size,
            "controller.learning_rate": args.learning_rate,
            "controller.seed": args.seed,
            "controller.training_costs": list(args.costs),
        }
        for dotted_path, actual in requirements.items():
            protocol.require(dotted_path, actual)

    rollouts = read_jsonl_bytes(data_capture.payload, str(args.data))
    validate_unique(rollouts)
    if protocol is not None:
        protocol.require("collection.train_records", len(rollouts))
    TrainConfig, train_policy = _training_api()
    config = TrainConfig(
        max_samples=args.max_samples,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        costs=args.costs,
    )
    policy, training = train_policy(rollouts, config)
    _verify_unchanged(args.data, data_snapshot, "training data")
    if protocol is not None:
        if Protocol.load(protocol.path).metadata() != protocol.metadata():
            raise RuntimeError("frozen protocol changed during training")
        _verify_unchanged(args.manifest, manifest_snapshot, "generation manifest")
        training["provenance"] = {
            "protocol": protocol.metadata(),
            "manifest": manifest_snapshot,
            "data": manifest["artifacts"]["train"],
        }
    policy.save(args.output, training)
    console.print(
        f"Trained on [bold]{len(rollouts)}[/] trajectories / "
        f"[bold]{training['state_cost_pairs']}[/] state-cost pairs; final exact-target loss "
        f"[bold]{training['final_loss']:.5f}[/]"
    )
    console.print(f"Saved policy to [cyan]{args.output}[/]")


def command_evaluate(args: argparse.Namespace) -> None:
    _assert_distinct_paths(
        data=args.data,
        policy=args.policy,
        output=args.output,
        svg=args.svg,
        html=args.html,
        protocol=args.protocol,
        manifest=args.manifest,
        frozen_baselines=args.frozen_baselines,
        export_baselines=args.export_baselines,
    )
    if args.frozen_baselines and args.export_baselines:
        raise ValueError("cannot consume and export baseline selections in one evaluation")
    if args.frozen_baselines:
        selection_preview = json.loads(
            _capture_file(args.frozen_baselines, "baseline selection").payload.decode("utf-8")
        )
        if not isinstance(selection_preview, dict):
            raise ValueError("baseline selection must be a JSON object")
        indirect_inputs = {}
        for field in ("validation_benchmark", "validation_data", "validation_manifest"):
            binding = selection_preview.get(field)
            if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
                raise ValueError(f"baseline selection {field} path is invalid")
            indirect_inputs[field] = Path(args.frozen_baselines).parent / binding["path"]
        _assert_distinct_paths(
            **indirect_inputs,
            output=args.output,
            svg=args.svg,
            html=args.html,
        )
    protocol = None if args.protocol is None else Protocol.load(args.protocol)
    payload = _write_benchmark(
        Path(args.data),
        Path(args.policy),
        Path(args.output),
        args.costs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        frozen_baseline_path=(
            None if args.frozen_baselines is None else Path(args.frozen_baselines)
        ),
        protocol=protocol,
        manifest_path=None if args.manifest is None else Path(args.manifest),
        split=args.split,
    )
    _print_benchmark(payload)
    if args.export_baselines:
        if protocol is None or args.split != "validation":
            raise ValueError("baseline export requires protocol-bound validation evaluation")
        selected = {
            _cost_key(comparison["scoring_cost"]): comparison["baseline_policy"]
            for comparison in payload["comparisons"]
        }
        benchmark_binding = _snapshot_file(args.output, "validation benchmark")
        benchmark_binding["path"] = os.path.relpath(
            args.output,
            Path(args.export_baselines).parent,
        )
        selection_parent = Path(args.export_baselines).parent
        validation_data_binding = {key: payload["data"][key] for key in ("path", "bytes", "sha256")}
        validation_data_binding["path"] = os.path.relpath(args.data, selection_parent)
        validation_manifest_binding = dict(payload["data"]["manifest"])
        validation_manifest_binding["path"] = os.path.relpath(
            args.manifest,
            selection_parent,
        )
        selection = {
            "schema_version": 1,
            "selection": "observed-best-on-validation",
            "protocol": payload["protocol"],
            "policy": {key: payload["policy"][key] for key in ("path", "bytes", "sha256")},
            "costs": list(args.costs),
            "validation_data": validation_data_binding,
            "validation_manifest": validation_manifest_binding,
            "validation_benchmark": benchmark_binding,
            "baselines": selected,
        }
        atomic_write_text(
            args.export_baselines,
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
        )
        console.print(f"Froze hash-bound validation baselines to [cyan]{args.export_baselines}[/]")
    if args.svg or args.html:
        write_report(args.output, args.svg, args.html)
        console.print("Rendered " + ", ".join(path for path in (args.svg, args.html) if path))


def command_report(args: argparse.Namespace) -> None:
    _assert_distinct_paths(benchmark=args.benchmark, svg=args.svg, html=args.html)
    if not args.svg and not args.html:
        raise ValueError("report requires --svg, --html, or both")
    write_report(args.benchmark, args.svg, args.html)
    console.print("Rendered " + ", ".join(path for path in (args.svg, args.html) if path))


def command_plan(args: argparse.Namespace) -> None:
    _assert_distinct_paths(benchmark=args.benchmark, json_output=args.json_output)
    aliases = {
        "learned": "offline-rl",
        "fixed": "fixed",
        "heuristic": "heuristic",
    }
    families = None if not args.family else tuple(aliases[name] for name in args.family)
    plan = load_deployment_plan(
        args.benchmark,
        args.sample_budget,
        conservative=not args.point_estimate,
        families=families,
    )
    table = Table(title="Validation-calibrated deployment plan", header_style="bold cyan")
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("family", plan.family)
    table.add_row("policy", plan.policy)
    table.add_row(
        "strategy spec",
        json.dumps(plan.to_dict()["strategy_spec"], separators=(",", ":"), sort_keys=True),
    )
    table.add_row(
        "accuracy",
        f"{plan.expected_accuracy:.1%} "
        f"[{plan.accuracy_interval.lower:.1%}, {plan.accuracy_interval.upper:.1%}]",
    )
    table.add_row(
        "average samples",
        f"{plan.expected_samples:.2f} "
        f"[{plan.samples_interval.lower:.2f}, {plan.samples_interval.upper:.2f}]",
    )
    table.add_row("requested budget", f"{plan.requested_sample_budget:.2f} average samples")
    table.add_row("validation feasibility", "satisfied" if plan.budget_satisfied else "not met")
    console.print(table)
    if not plan.budget_satisfied:
        console.print(
            "[yellow]No measured deployable strategy met this validation budget; "
            "showing the minimum-compute point without claiming feasibility.[/]"
        )
    if args.json_output:
        atomic_write_text(
            args.json_output,
            json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        console.print(f"Wrote deployment plan to [cyan]{args.json_output}[/]")


def command_demo(args: argparse.Namespace) -> None:
    rollouts = read_jsonl(args.data)
    rollout = rollouts[args.index % len(rollouts)]
    policy = BranchPilotPolicy.load(args.policy)
    trace = decision_trace(rollout, policy, args.cost)
    table = Table(title=f"Live policy trace · λ={args.cost:g}", header_style="bold cyan")
    table.add_column("sample", justify="right")
    table.add_column("parsed answer", justify="right")
    table.add_column("vote winner", justify="right")
    table.add_column("Q(stop)", justify="right")
    table.add_column("Q(continue)", justify="right")
    table.add_column("action")
    for decision in trace:
        sample = rollout.samples[decision.sample_count - 1]
        style = "bold magenta" if decision.action == "stop" else None
        table.add_row(
            str(decision.sample_count),
            sample.answer or "∅",
            decision.majority_answer or "∅",
            f"{decision.q_stop:+.3f}",
            f"{decision.q_continue:+.3f}",
            decision.action.upper(),
            style=style,
        )
    console.print(f"[dim]{rollout.question}[/]")
    console.print(table)
    final = trace[-1]
    verdict = "correct" if final.majority_answer == rollout.gold else "incorrect"
    console.print(
        f"Stopped after [bold]{final.sample_count}[/] samples with "
        f"[bold]{final.majority_answer}[/] — {verdict}; gold={rollout.gold}"
    )


def command_quickstart(args: argparse.Namespace) -> None:
    TrainConfig, train_policy = _training_api()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path, test_path = output / "train.jsonl", output / "test.jsonl"
    policy_path, benchmark_path = output / "policy.safetensors", output / "benchmark.json"
    train = make_synthetic_rollouts(args.train_size, args.max_samples, args.seed)
    test = make_synthetic_rollouts(args.test_size, args.max_samples, args.seed + 1)
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    config = TrainConfig(
        max_samples=args.max_samples,
        epochs=args.epochs,
        seed=args.seed,
        costs=args.costs,
    )
    policy, training = train_policy(train, config)
    policy.save(policy_path, training)
    payload = _write_benchmark(test_path, policy_path, benchmark_path, args.costs)
    write_report(benchmark_path, output / "pareto.svg", output / "report.html")
    _print_benchmark(payload)
    demo_args = argparse.Namespace(
        data=str(test_path),
        policy=str(policy_path),
        index=0,
        cost=args.costs[len(args.costs) // 2],
    )
    command_demo(demo_args)
    console.print(f"Open [cyan]{output / 'report.html'}[/] for the standalone report.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="branchpilot",
        description="Learn when another LLM reasoning sample is worth its inference cost.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="create reproducible local trajectories")
    synthetic.add_argument("--output-dir", default="artifacts/synthetic")
    synthetic.add_argument("--train-size", type=int, default=512)
    synthetic.add_argument("--test-size", type=int, default=256)
    synthetic.add_argument("--max-samples", type=int, default=8)
    synthetic.add_argument("--seed", type=int, default=17)
    synthetic.set_defaults(handler=command_synthetic)

    split = subparsers.add_parser("split", help="create deterministic disjoint data splits")
    split.add_argument("--data", required=True)
    split.add_argument("--train-output", required=True)
    split.add_argument("--test-output", required=True)
    split.add_argument("--train-size", type=int, required=True)
    split.add_argument("--test-size", type=int, required=True)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--manifest")
    split.set_defaults(handler=command_split)

    audit = subparsers.add_parser(
        "audit", help="validate trajectory integrity and profile observation coverage"
    )
    audit.add_argument("--data", required=True)
    audit.add_argument("--compare", help="second dataset that must be disjoint")
    audit.add_argument("--json-output")
    audit.set_defaults(handler=command_audit)

    train = subparsers.add_parser("train", help="fit the offline Q-controller")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="artifacts/policy.safetensors")
    train.add_argument("--max-samples", type=int, default=8)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument(
        "--costs",
        type=_costs,
        default=_costs("0,0.01,0.025,0.05,0.075,0.1,0.15,0.25"),
    )
    train.add_argument("--protocol", help="frozen benchmark protocol JSON")
    train.add_argument("--manifest", help="hash-bound generation manifest")
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="compare the learned controller with exhaustive baselines"
    )
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--output", default="artifacts/benchmark.json")
    evaluate.add_argument("--svg")
    evaluate.add_argument("--html")
    evaluate.add_argument(
        "--costs",
        type=_costs,
        default=_costs("0.01,0.025,0.05,0.075,0.1,0.15"),
    )
    evaluate.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=17)
    evaluate.add_argument(
        "--frozen-baselines",
        help="hash-bound validation baseline selection record",
    )
    evaluate.add_argument("--protocol", help="frozen benchmark protocol JSON")
    evaluate.add_argument("--manifest", help="hash-bound generation manifest")
    evaluate.add_argument("--split", choices=("validation", "test"))
    evaluate.add_argument(
        "--export-baselines",
        help="write the observed-best comparator names for later frozen test evaluation",
    )
    evaluate.set_defaults(handler=command_evaluate)

    report = subparsers.add_parser(
        "report", help="render deterministic standalone evidence from benchmark JSON"
    )
    report.add_argument("--benchmark", required=True)
    report.add_argument("--svg")
    report.add_argument("--html")
    report.set_defaults(handler=command_report)

    plan = subparsers.add_parser(
        "plan", help="select a validation-measured cost for an average sample budget"
    )
    plan.add_argument("--benchmark", required=True)
    plan.add_argument("--sample-budget", type=float, required=True)
    plan.add_argument(
        "--point-estimate",
        action="store_true",
        help="use measured average samples instead of the conservative 95%% upper bound",
    )
    plan.add_argument(
        "--family",
        action="append",
        choices=("learned", "fixed", "heuristic"),
        help="limit candidates; repeat to allow several families (default: all)",
    )
    plan.add_argument("--json-output")
    plan.set_defaults(handler=command_plan)

    demo = subparsers.add_parser("demo", help="inspect every stop/continue decision")
    demo.add_argument("--data", required=True)
    demo.add_argument("--policy", required=True)
    demo.add_argument("--index", type=int, default=0)
    demo.add_argument("--cost", type=float, default=0.05)
    demo.set_defaults(handler=command_demo)

    quickstart = subparsers.add_parser("quickstart", help="run the complete zero-GPU pipeline")
    quickstart.add_argument("--output-dir", default="artifacts/quickstart")
    quickstart.add_argument("--train-size", type=int, default=384)
    quickstart.add_argument("--test-size", type=int, default=192)
    quickstart.add_argument("--max-samples", type=int, default=8)
    quickstart.add_argument("--epochs", type=int, default=60)
    quickstart.add_argument("--seed", type=int, default=17)
    quickstart.add_argument(
        "--costs", type=_costs, default=_costs("0.01,0.025,0.05,0.075,0.1,0.15")
    )
    quickstart.set_defaults(handler=command_quickstart)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
