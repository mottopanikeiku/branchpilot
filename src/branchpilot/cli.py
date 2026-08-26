from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from branchpilot.artifacts import atomic_write_text, paths_alias, sha256_file
from branchpilot.calibration import load_operating_point
from branchpilot.evaluate import benchmark, decision_trace, pareto_frontier
from branchpilot.features import FEATURE_NAMES
from branchpilot.integrity import (
    profile_rollouts,
    validate_disjoint,
    validate_unique,
)
from branchpilot.policy import ARTIFACT_VERSION, COST_MODEL, BranchPilotPolicy
from branchpilot.report import write_report
from branchpilot.schema import SCHEMA_VERSION, read_jsonl, write_jsonl
from branchpilot.synthetic import make_synthetic_rollouts

console = Console()


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


def _load_frozen_baselines(path: str | None) -> dict[float, str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("frozen baseline file must contain a non-empty JSON object")
    try:
        return {float(cost): str(policy) for cost, policy in payload.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen baseline keys must be numeric costs") from exc


def _write_benchmark(
    data_path: Path,
    policy_path: Path,
    output: Path,
    costs: tuple[float, ...],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
    frozen_baselines: dict[float, str] | None = None,
    protocol_path: Path | None = None,
) -> dict:
    rollouts = read_jsonl(data_path)
    validate_unique(rollouts)
    policy = BranchPilotPolicy.load(policy_path)
    result = benchmark(
        rollouts,
        policy,
        costs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        frozen_baselines=frozen_baselines,
    )
    protocol = None
    if protocol_path is not None:
        protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(protocol_payload, dict):
            raise ValueError("benchmark protocol must be a JSON object")
        protocol = {
            "path": protocol_path.name,
            "sha256": sha256_file(protocol_path),
            "status": protocol_payload.get("status"),
            "evidence_tier": protocol_payload.get("evidence_tier"),
        }
    payload = {
        "schema_version": 2,
        **result.to_dict(),
        "objective": {
            "name": "additional-sample utility",
            "formula": "accuracy - lambda * (samples - 1)",
            "cost_unit": "additional_samples",
        },
        "protocol": protocol,
        "data": {
            "path": data_path.name,
            "sha256": sha256_file(data_path),
            "schema_version": SCHEMA_VERSION,
            "profile": profile_rollouts(rollouts).to_dict(),
        },
        "policy": {
            "path": policy_path.name,
            "sha256": sha256_file(policy_path),
            "artifact_version": ARTIFACT_VERSION,
            "cost_model": COST_MODEL,
            "feature_names": list(FEATURE_NAMES),
            "training": policy.training,
        },
        "pareto_frontier": [row.to_dict() for row in pareto_frontier(result)],
    }
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
    if paths_alias(train_path, test_path):
        raise ValueError("train and test outputs must be distinct paths")
    records = read_jsonl(args.data)
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
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)
    if args.manifest:
        manifest = {
            "schema_version": 1,
            "seed": args.seed,
            "source": {
                **source_profile.to_dict(),
                "path": Path(args.data).name,
                "sha256": sha256_file(args.data),
            },
            "train": {
                **profile_rollouts(train_records).to_dict(),
                "path": train_path.name,
                "sha256": sha256_file(train_path),
            },
            "test": {
                **profile_rollouts(test_records).to_dict(),
                "path": test_path.name,
                "sha256": sha256_file(test_path),
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
    records = read_jsonl(args.data)
    validate_unique(records)
    profile = profile_rollouts(records)
    payload: dict = {"data": str(args.data), "profile": profile.to_dict()}
    if args.compare:
        comparison = read_jsonl(args.compare)
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
    TrainConfig, train_policy = _training_api()
    rollouts = read_jsonl(args.data)
    validate_unique(rollouts)
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
    policy.save(args.output, training)
    console.print(
        f"Trained on [bold]{len(rollouts)}[/] trajectories / "
        f"[bold]{training['state_cost_pairs']}[/] state-cost pairs; final exact-target loss "
        f"[bold]{training['final_loss']:.5f}[/]"
    )
    console.print(f"Saved policy to [cyan]{args.output}[/]")


def command_evaluate(args: argparse.Namespace) -> None:
    payload = _write_benchmark(
        Path(args.data),
        Path(args.policy),
        Path(args.output),
        args.costs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        frozen_baselines=_load_frozen_baselines(args.frozen_baselines),
        protocol_path=(None if args.protocol is None else Path(args.protocol)),
    )
    _print_benchmark(payload)
    if args.export_baselines:
        selected = {
            f"{comparison['scoring_cost']:g}": comparison["baseline_policy"]
            for comparison in payload["comparisons"]
        }
        atomic_write_text(
            args.export_baselines,
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
        )
        console.print(f"Froze validation-selected baselines to [cyan]{args.export_baselines}[/]")
    if args.svg or args.html:
        write_report(args.output, args.svg, args.html)
        console.print("Rendered " + ", ".join(path for path in (args.svg, args.html) if path))


def command_report(args: argparse.Namespace) -> None:
    if not args.svg and not args.html:
        raise ValueError("report requires --svg, --html, or both")
    write_report(args.benchmark, args.svg, args.html)
    console.print("Rendered " + ", ".join(path for path in (args.svg, args.html) if path))


def command_plan(args: argparse.Namespace) -> None:
    point = load_operating_point(
        args.benchmark,
        args.sample_budget,
        conservative=not args.point_estimate,
    )
    table = Table(title="Validation-calibrated operating point", header_style="bold cyan")
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("decision cost λ", f"{point.cost:g}")
    table.add_row(
        "accuracy",
        f"{point.expected_accuracy:.1%} "
        f"[{point.accuracy_interval.lower:.1%}, {point.accuracy_interval.upper:.1%}]",
    )
    table.add_row(
        "average samples",
        f"{point.expected_samples:.2f} "
        f"[{point.samples_interval.lower:.2f}, {point.samples_interval.upper:.2f}]",
    )
    table.add_row("requested budget", f"{point.requested_sample_budget:.2f} average samples")
    table.add_row("validation feasibility", "satisfied" if point.budget_satisfied else "not met")
    console.print(table)
    if not point.budget_satisfied:
        console.print(
            "[yellow]No measured learned point met this validation budget; "
            "showing the minimum-compute point without claiming feasibility.[/]"
        )
    if args.json_output:
        atomic_write_text(
            args.json_output,
            json.dumps(point.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        console.print(f"Wrote operating point to [cyan]{args.json_output}[/]")


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
    demo_args = argparse.Namespace(data=str(test_path), policy=str(policy_path), index=0, cost=0.05)
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
        help="JSON object mapping every evaluation cost to a validation-selected policy name",
    )
    evaluate.add_argument("--protocol", help="frozen benchmark protocol JSON")
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
        help="use measured average samples instead of the conservative 95% upper bound",
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
