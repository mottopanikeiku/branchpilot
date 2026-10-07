"""Select on train/validation, commit selection, then evaluate the held-out MATH split.

Run ``select`` before ``test --selection-commit COMMIT``. All settings come from
benchmarks/math500-protocol.json; there are no training or bootstrap overrides.
The test command also writes the comparison SVG from committed GSM8K results.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from branchpilot.evaluate import BenchmarkResult, benchmark
from branchpilot.policy import BranchPilotPolicy
from branchpilot.schema import Rollout, read_jsonl_bytes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("benchmarks/math500-protocol.json")
BANK = Path("benchmarks/math500")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def resolve_commit(root: Path, commit: str) -> str:
    return git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()


def protocol_at_commit(root: Path, commit: str) -> str:
    resolved = resolve_commit(root, commit)
    committed = git(root, "show", f"{resolved}:{PROTOCOL.as_posix()}")
    if hashlib.sha256(committed).hexdigest() != digest(root / PROTOCOL):
        raise ValueError("protocol differs from the recorded preregistration commit")
    return resolved


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Check the declared partitions without opening any response bank."""
    indices: set[int] = set()
    uids: set[str] = set()
    for split in ("train", "validation", "test"):
        spec = protocol["splits"][split]
        if not spec["indices"] or len(spec["indices"]) != len(spec["uids"]):
            raise ValueError(f"invalid {split} partition")
        if any(type(index) is not int for index in spec["indices"]):
            raise ValueError("source indices must be integers")
        if len(set(spec["indices"])) != len(spec["indices"]):
            raise ValueError("duplicate source indices")
        if len(set(spec["uids"])) != len(spec["uids"]):
            raise ValueError("duplicate dataset UIDs")
        if indices.intersection(spec["indices"]) or uids.intersection(spec["uids"]):
            raise ValueError("partitions must have disjoint source indices and UIDs")
        indices.update(spec["indices"])
        uids.update(spec["uids"])
    if indices != set(range(protocol["dataset"]["rows"])):
        raise ValueError("partitions must cover every dataset source index exactly once")
    if protocol["collection"]["samples_per_prompt"] != protocol["controller"]["max_samples"]:
        raise ValueError("collection and controller sample limits differ")
    evaluation = protocol["evaluation"]
    if evaluation["confidence"] != 0.95:
        raise ValueError("existing benchmark uses 95% confidence intervals")
    if not set(evaluation["primary_costs"]).issubset(evaluation["reported_costs"]):
        raise ValueError("primary costs must be reported")


def load_bank(root: Path, protocol: dict[str, Any], split: str) -> list[Rollout]:
    path = root / BANK / f"{split}.jsonl.gz"
    records = read_jsonl_bytes(gzip.decompress(path.read_bytes()), source=path.name)
    spec = protocol["splits"][split]
    expected = dict(zip(spec["indices"], spec["uids"], strict=True))
    found: dict[int, Rollout] = {}
    rollout_uids: set[str] = set()
    for record in records:
        metadata = record.metadata
        index = metadata.get("source_index")
        if type(index) is not int or index not in expected:
            raise ValueError(f"unexpected {split} source index")
        if index in found or record.uid in rollout_uids:
            raise ValueError(f"duplicate {split} source index or rollout UID")
        if metadata.get("dataset_uid") != expected[index] or record.uid != expected[index]:
            raise ValueError(f"{split} dataset UID does not match protocol source index")
        if metadata.get("output_split") != split:
            raise ValueError(f"wrong output_split in {split} bank")
        if len(record.samples) != protocol["collection"]["samples_per_prompt"]:
            raise ValueError(f"wrong sample count in {split} bank")
        correctness = metadata.get("answer_correctness")
        labels = {sample.answer for sample in record.samples if sample.answer is not None}
        if (
            record.gold != "math500-gold-unused"
            or not isinstance(correctness, dict)
            or set(correctness) != labels
            or any(type(value) is not bool for value in correctness.values())
        ):
            raise ValueError(f"{split} bank must be parsed with sample-label correctness")
        found[index] = record
        rollout_uids.add(record.uid)
    if set(found) != set(expected):
        raise ValueError(f"{split} bank does not contain the complete protocol partition")
    # File row order cannot change training shuffle or prompt bootstrap order.
    return [found[index] for index in spec["indices"]]


def train_from_protocol(records: list[Rollout], protocol: dict[str, Any]):
    import torch

    from branchpilot.training import TrainConfig, train_policy

    torch.set_num_threads(min(2, torch.get_num_threads()))

    settings = dict(protocol["controller"])
    settings["costs"] = tuple(settings["costs"])
    config = TrainConfig(**settings)
    policy, training = train_policy(records, config)
    if training["config"] != asdict(config):
        raise ValueError("training configuration differs from protocol")
    return policy, training


def evaluate_bank(records, policy, protocol, frozen=None) -> BenchmarkResult:
    evaluation = protocol["evaluation"]
    return benchmark(
        records,
        policy,
        evaluation["reported_costs"],
        bootstrap_samples=evaluation["bootstrap_resamples"],
        bootstrap_seed=evaluation["bootstrap_seed"],
        frozen_baselines=frozen,
    )


def summary(result: BenchmarkResult) -> dict[str, Any]:
    # Per-prompt arrays are stored once in the compressed outcomes file.
    return {
        "records": result.records,
        "max_samples": result.max_samples,
        "costs": list(result.costs),
        "rows": [row.to_dict() for row in result.rows],
        "comparisons": [item.to_dict() for item in result.comparisons],
        "bootstrap": {
            "resamples": result.bootstrap_samples,
            "seed": result.bootstrap_seed,
            "confidence": result.confidence,
        },
    }


def select(root: Path, preregistration_commit: str) -> dict[str, Any]:
    protocol = read_json(root / PROTOCOL)
    validate_protocol(protocol)
    preregistration_commit = protocol_at_commit(root, preregistration_commit)
    train = load_bank(root, protocol, "train")
    validation = load_bank(root, protocol, "validation")
    if {row.uid for row in train}.intersection(row.uid for row in validation):
        raise ValueError("train and validation rollout UIDs must be disjoint")
    policy, training = train_from_protocol(train, protocol)
    destination = root / BANK
    destination.mkdir(parents=True, exist_ok=True)
    policy.save(destination / "policy.safetensors", training)
    write_json(destination / "training.json", training)
    result = evaluate_bank(validation, policy, protocol)
    # benchmark() uses max() over fixed, confidence, agreement in existing order;
    # equal validation utilities therefore retain the earliest existing row.
    baselines = {str(item.scoring_cost): item.baseline_policy for item in result.comparisons}
    files = [
        PROTOCOL,
        BANK / "train.jsonl.gz",
        BANK / "validation.jsonl.gz",
        BANK / "policy.safetensors",
        BANK / "training.json",
    ]
    selection = {
        "schema": 1,
        "preregistration_commit": preregistration_commit,
        "hashes": {path.as_posix(): digest(root / path) for path in files},
        "controller": protocol["controller"],
        "baselines": baselines,
        "selection": "best validation utility; ties retain existing benchmark row order",
        "validation": summary(result),
    }
    write_json(destination / "selection.json", selection)
    return {
        "command": "select",
        "baselines": baselines,
        "policy_sha256": selection["hashes"][(BANK / "policy.safetensors").as_posix()],
    }


def verify_selection(root: Path, protocol: dict[str, Any], selection_commit: str):
    path = BANK / "selection.json"
    selection = read_json(root / path)
    commit = resolve_commit(root, selection_commit)
    preregistration = protocol_at_commit(root, selection["preregistration_commit"])
    if commit == preregistration:
        raise ValueError("selection must be committed after preregistration")
    git(root, "merge-base", "--is-ancestor", preregistration, commit)
    git(root, "merge-base", "--is-ancestor", commit, "HEAD")
    if git(root, "show", f"{commit}:{path.as_posix()}") != (root / path).read_bytes():
        raise ValueError("selection differs from the recorded selection commit")
    required = {
        PROTOCOL.as_posix(),
        (BANK / "train.jsonl.gz").as_posix(),
        (BANK / "validation.jsonl.gz").as_posix(),
        (BANK / "policy.safetensors").as_posix(),
        (BANK / "training.json").as_posix(),
    }
    if set(selection["hashes"]) != required:
        raise ValueError("selection must hash protocol, train, validation, policy and training")
    for filename, expected in selection["hashes"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"changed selection input: {filename}")
    if selection["controller"] != protocol["controller"]:
        raise ValueError("selection controller differs from protocol")
    costs = protocol["evaluation"]["reported_costs"]
    if set(selection["baselines"]) != {str(cost) for cost in costs}:
        raise ValueError("selection must contain exactly the reported costs")
    validation = selection["validation"]
    for cost in costs:
        rows = [
            row
            for row in validation["rows"]
            if row["scoring_cost"] == cost and row["family"] != "offline-rl"
        ]
        best = max(rows, key=lambda row: row["utility"])
        if selection["baselines"][str(cost)] != best["policy"]:
            raise ValueError("recorded comparator differs from best validation utility")
    return selection, commit


def primary_rule(comparisons: list[dict[str, Any]], primary_costs: list[float]):
    by_cost = {item["scoring_cost"]: item for item in comparisons}
    lowers = {str(cost): by_cost[cost]["utility_delta_interval"]["lower"] for cost in primary_costs}
    success = sum(value > 0 for value in lowers.values()) >= 2 and all(
        value >= 0 for value in lowers.values()
    )
    return {
        "success": success,
        "utility_interval_lower_bounds": lowers,
        "rule": "lower bound > 0 at at least two primary costs and >= 0 at the third",
        "scope": "new MATH-trained policy only; transfer and matched budgets descriptive",
    }


def count_mixture(target: float, low: float, high: float) -> tuple[float, float]:
    """Return low/high weights using counts only, including degenerate boundaries."""
    if not all(math.isfinite(value) for value in (target, low, high)):
        raise ValueError("mixture counts must be finite")
    if low > high or target < low or target > high:
        raise ValueError("target must be bracketed by mixture counts")
    if low == high:
        return 1.0, 0.0
    high_weight = (target - low) / (high - low)
    return 1.0 - high_weight, high_weight


def matched_budgets(result: BenchmarkResult, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = {item.policy: item for item in result.policy_outcomes}
    uids = result.policy_outcomes[0].uids
    if any(item.uids != uids for item in result.policy_outcomes):
        raise ValueError("matched comparisons require identical prompt UID order")
    evaluation = protocol["evaluation"]
    indices = np.random.default_rng(evaluation["bootstrap_seed"]).integers(
        0, len(uids), size=(evaluation["bootstrap_resamples"], len(uids))
    )
    agreement_mean = float(np.mean(outcomes["agreement-2"].samples))
    comparisons = []
    for cost in result.costs:
        learned = outcomes[f"BranchPilot λ={cost!r}"]
        counts = np.asarray(learned.samples, dtype=float)
        correct = np.asarray(learned.correct, dtype=float)
        target = float(counts.mean())
        low, high = math.floor(target), math.ceil(target)
        mixtures = [("adjacent-fixed", f"fixed-{low}", f"fixed-{high}")]
        if target <= agreement_mean:
            mixtures.append(("agreement-2-budget", "fixed-1", "agreement-2"))
        else:
            mixtures.append(("agreement-2-budget", "agreement-2", "fixed-8"))
        for name, low_name, high_name in mixtures:
            a, b = outcomes[low_name], outcomes[high_name]
            a_count, b_count = float(np.mean(a.samples)), float(np.mean(b.samples))
            a_weight, b_weight = count_mixture(target, a_count, b_count)
            mixed_correct = a_weight * np.asarray(a.correct) + b_weight * np.asarray(b.correct)
            mixed_counts = a_weight * np.asarray(a.samples) + b_weight * np.asarray(b.samples)
            differences = correct - mixed_correct
            bounds = np.quantile(differences[indices].mean(axis=1), (0.025, 0.975))
            comparisons.append(
                {
                    "scoring_cost": cost,
                    "learned_policy": learned.policy,
                    "comparison": name,
                    "learned_accuracy": float(correct.mean()),
                    "learned_mean_samples": target,
                    "components": [
                        {
                            "policy": low_name,
                            "weight": a_weight,
                            "mean_samples": a_count,
                            "accuracy": float(np.mean(a.correct)),
                        },
                        {
                            "policy": high_name,
                            "weight": b_weight,
                            "mean_samples": b_count,
                            "accuracy": float(np.mean(b.correct)),
                        },
                    ],
                    "mixture_expected_mean_samples": float(mixed_counts.mean()),
                    "mixture_expected_accuracy": float(mixed_correct.mean()),
                    "accuracy_delta": float(differences.mean()),
                    "accuracy_delta_interval": {
                        "lower": float(bounds[0]),
                        "upper": float(bounds[1]),
                    },
                    "interpretation": "Descriptive paired prompt bootstrap conditional on fixed "
                    "whole-test count weights; no mixture randomization or "
                    "weight-estimation uncertainty. Expected budgets match on "
                    "the whole test, not necessarily in each bootstrap resample.",
                }
            )
    return comparisons


def compact_outcomes(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    uids = next(iter(results.values())).policy_outcomes[0].uids
    policies: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for label, result in results.items():
        names = []
        for outcome in result.policy_outcomes:
            if outcome.uids != uids:
                raise ValueError("evaluations must share prompt UID order")
            name = (
                f"{label}: {outcome.policy}" if outcome.family == "offline-rl" else outcome.policy
            )
            item = {
                "family": outcome.family,
                "correct": list(outcome.correct),
                "samples": list(outcome.samples),
                "tokens": list(outcome.tokens),
            }
            previous = policies.setdefault(name, item)
            if previous != item:
                raise ValueError("shared baseline outcomes differ between policies")
            names.append(name)
        evaluations[label] = {"policy_names": names, "costs": list(result.costs)}
    return {"schema": 1, "uids": list(uids), "policies": policies, "evaluations": evaluations}


def render_svg(gsm: dict[str, Any], result: dict[str, Any]) -> str:
    """Two separate axes: different tasks and differently sized held-out sets."""
    width, height = 1080, 530
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Stopping on GSM8K and MATH-500</title>',
        '<desc id="desc">Accuracy versus average samples. GSM8K full official test, '
        "1319 prompts; MATH-500 internal held-out test, 200 prompts. Error bars are "
        "marginal 95% prompt bootstrap intervals. These are different tasks.</desc>",
        '<rect width="1080" height="530" fill="white"/>',
        '<g font-family="sans-serif" fill="#17212e">',
    ]

    def text(x, y, value, size=14):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')

    text(35, 32, "Stopping accuracy versus sample count", 23)
    colors = {
        "fixed": "#334155",
        "agreement": "#b46b00",
        "learned": "#95358d",
        "transfer": "#087e8b",
    }
    for index, (name, color) in enumerate(colors.items()):
        x = 40 + 230 * index
        parts.append(f'<circle cx="{x}" cy="57" r="4" fill="{color}"/>')
        text(
            x + 12,
            62,
            {
                "fixed": "Fixed counts",
                "agreement": "Agreement-2",
                "learned": "Task-trained policy",
                "transfer": "GSM8K transfer",
            }[name],
        )
    panels = [
        ("GSM8K · full official test · N=1,319", gsm["splits"]["test"]["rows"], None),
        ("MATH-500 · internal held-out · N=200", result["test"]["rows"], result.get("transfer")),
    ]
    for panel, (title, raw_rows, transfer) in enumerate(panels):
        left, right, top, bottom = 70 + panel * 530, 505 + panel * 530, 145, 410
        text(left - 20, 104, title, 17)
        if panel == 0:
            text(left - 20, 125, "Earlier success rule: not met", 12)
        else:
            label = "met" if result["decision"]["success"] else "not met"
            text(left - 20, 125, f"Fixed primary success rule: {label}", 12)
        rows, seen = [], set()
        for row in raw_rows:
            policy = row["policy"]
            if policy in seen:
                continue
            seen.add(policy)
            if policy.startswith("fixed-"):
                family = "fixed"
            elif policy == "agreement-2":
                family = "agreement"
            elif policy.startswith("BranchPilot"):
                family = "learned"
            else:
                continue
            rows.append((row, family))
        if transfer is not None:
            rows.extend(
                (row, "transfer") for row in transfer["rows"] if row["family"] == "offline-rl"
            )
        lower = max(0.0, min(row["accuracy_interval"]["lower"] for row, _ in rows) - 0.025)
        upper = min(1.0, max(row["accuracy_interval"]["upper"] for row, _ in rows) + 0.025)

        def x(value, left=left, right=right):
            return left + (value - 1) / 7 * (right - left)

        def y(value, bottom=bottom, lower=lower, upper=upper, top=top):
            return bottom - (value - lower) / (upper - lower) * (bottom - top)

        for tick in np.linspace(lower, upper, 5):
            parts.append(f'<path d="M {left} {y(tick):.2f} H {right}" stroke="#e5e7eb"/>')
            text(left - 45, round(y(tick) + 4, 2), f"{tick * 100:.0f}%", 12)
        for count in range(1, 9):
            text(round(x(count) - 4, 2), bottom + 20, count, 12)
        fixed = sorted(
            (row for row, family in rows if family == "fixed"),
            key=lambda row: row["average_samples"],
        )
        points = " ".join(
            f"{x(row['average_samples']):.2f},{y(row['accuracy']):.2f}" for row in fixed
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="#334155"/>')
        for row, family in rows:
            cx, cy = x(row["average_samples"]), y(row["accuracy"])
            interval = row["accuracy_interval"]
            count_interval = row["average_samples_interval"]
            color = colors[family]
            parts.append(
                f'<path d="M {cx:.2f} {y(interval["lower"]):.2f} V '
                f"{y(interval['upper']):.2f} M {x(count_interval['lower']):.2f} "
                f'{cy:.2f} H {x(count_interval["upper"]):.2f}" '
                f'stroke="{color}" opacity="0.5"/>'
            )
            parts.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{color}">'
                f"<title>{escape(row['policy'])}: accuracy {row['accuracy']:.6f}, "
                f"mean samples {row['average_samples']:.6f}</title></circle>"
            )
        text(left + 90, 456, "Average samples per prompt", 14)
    text(
        35,
        490,
        "95% prompt-bootstrap intervals · 10,000 resamples · seed 17. Separate accuracy scales.",
        13,
    )
    text(
        35,
        513,
        "Response-bank replay; sample counts are not token, latency, or dollar budgets.",
        13,
    )
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def test(root: Path, selection_commit: str) -> dict[str, Any]:
    protocol = read_json(root / PROTOCOL)
    validate_protocol(protocol)
    selection, commit = verify_selection(root, protocol, selection_commit)
    train = load_bank(root, protocol, "train")
    validation = load_bank(root, protocol, "validation")
    records = load_bank(root, protocol, "test")
    all_uids = [row.uid for row in train + validation + records]
    if len(set(all_uids)) != len(all_uids):
        raise ValueError("rollout UIDs must be disjoint across all partitions")
    policy = BranchPilotPolicy.load(root / BANK / "policy.safetensors")
    if (
        policy.max_samples != protocol["controller"]["max_samples"]
        or policy.hidden_size != protocol["controller"]["hidden_size"]
        or list(policy.costs) != protocol["controller"]["costs"]
    ):
        raise ValueError("trained policy architecture differs from protocol")
    frozen = {float(cost): name for cost, name in selection["baselines"].items()}
    primary = evaluate_bank(records, policy, protocol, frozen)
    result = {
        "schema": 1,
        "question": protocol["question"],
        "scope": protocol["decision_rule"]["scope"],
        "preregistration_commit": selection["preregistration_commit"],
        "selection_commit": commit,
        "selection_sha256": digest(root / BANK / "selection.json"),
        "hashes": {
            **selection["hashes"],
            (BANK / "test.jsonl.gz").as_posix(): digest(root / BANK / "test.jsonl.gz"),
        },
        "test": summary(primary),
        "decision": primary_rule(
            [item.to_dict() for item in primary.comparisons],
            protocol["evaluation"]["primary_costs"],
        ),
        "matched_budget": matched_budgets(primary, protocol),
        "limitations": protocol["limitations"],
    }
    evaluations = {"math-trained": primary}
    transfer_path = root / BANK / "gsm8k-policy.safetensors"
    if transfer_path.exists():
        transfer_policy = BranchPilotPolicy.load(transfer_path)
        transfer = evaluate_bank(records, transfer_policy, protocol, frozen)
        result["transfer"] = {
            **summary(transfer),
            "descriptive_only": True,
            "policy_sha256": digest(transfer_path),
            "matched_budget": matched_budgets(transfer, protocol),
        }
        evaluations["gsm8k-transfer"] = transfer
    else:
        result["transfer_status"] = "original GSM8K policy unavailable; no substitute trained"
    outcome_path = root / BANK / "outcomes.json.gz"
    outcome_path.write_bytes(
        gzip.compress(
            json.dumps(
                compact_outcomes(evaluations),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            mtime=0,
        )
    )
    result["outcomes"] = {
        "path": (BANK / outcome_path.name).as_posix(),
        "sha256": digest(outcome_path),
        "format": "shared UIDs; arrays per policy",
    }
    gsm_path = root / "benchmarks/gsm8k-sampling-bootstrap.json"
    gsm = read_json(gsm_path)
    result["figure_inputs"] = {"benchmarks/gsm8k-sampling-bootstrap.json": digest(gsm_path)}
    svg_path = root / "assets/math500-generalization.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(gsm, result), encoding="utf-8")
    write_json(root / BANK / "result.json", result)
    return {
        "command": "test",
        "records": primary.records,
        "decision": result["decision"],
        "comparisons": [
            {
                key: item[key]
                for key in (
                    "scoring_cost",
                    "baseline_policy",
                    "utility_delta",
                    "utility_delta_interval",
                )
            }
            for item in result["test"]["comparisons"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select", help="train and select using train/validation only")
    selection.add_argument("--preregistration-commit", default="d475eeb")
    testing = commands.add_parser("test", help="evaluate after committing selection.json")
    testing.add_argument("--selection-commit", required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (
        select(root, args.preregistration_commit)
        if args.command == "select"
        else test(root, args.selection_commit)
    )
    print(json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
