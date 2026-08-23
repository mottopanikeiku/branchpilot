from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from branchpilot.evaluate import benchmark, decision_trace, pareto_frontier
from branchpilot.policy import BranchPilotPolicy, TrainConfig, train_policy
from branchpilot.report import write_report
from branchpilot.schema import read_jsonl, write_jsonl
from branchpilot.synthetic import make_synthetic_rollouts

console = Console()


def _costs(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("costs must be comma-separated numbers") from exc
    if not parsed or any(cost < 0 for cost in parsed):
        raise argparse.ArgumentTypeError("costs must be non-empty and non-negative")
    return parsed


def _write_benchmark(data_path: Path, policy_path: Path, output: Path, costs: tuple[float, ...]) -> dict:
    rollouts = read_jsonl(data_path)
    policy = BranchPilotPolicy.load(policy_path)
    rows = benchmark(rollouts, policy, costs)
    payload = {
        "schema_version": 1,
        "records": len(rollouts),
        "data": data_path.name,
        "policy": policy_path.name,
        "max_samples": policy.max_samples,
        "costs": costs,
        "rows": [row.to_dict() for row in rows],
        "pareto_frontier": [row.to_dict() for row in pareto_frontier(rows)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _print_benchmark(payload: dict) -> None:
    table = Table(title="Held-out accuracy–compute benchmark", header_style="bold magenta")
    table.add_column("inference cost λ", justify="right")
    table.add_column("policy")
    table.add_column("accuracy", justify="right")
    table.add_column("samples", justify="right")
    table.add_column("utility", justify="right")
    rows = payload["rows"]
    for cost in payload["costs"]:
        at_cost = [row for row in rows if abs(row["scoring_cost"] - cost) < 1e-9]
        learned = next(row for row in at_cost if row["family"] == "offline-rl")
        baseline = max((row for row in at_cost if row["family"] != "offline-rl"), key=lambda row: row["utility"])
        for row, style in ((learned, "bold magenta"), (baseline, "dim")):
            table.add_row(
                f"{cost:g}",
                row["policy"],
                f"{row['accuracy']:.1%}",
                f"{row['average_samples']:.2f}",
                f"{row['utility']:.3f}",
                style=style,
            )
    console.print(table)


def command_synthetic(args: argparse.Namespace) -> None:
    train = make_synthetic_rollouts(args.train_size, args.max_samples, args.seed)
    test = make_synthetic_rollouts(args.test_size, args.max_samples, args.seed + 1)
    output = Path(args.output_dir)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "test.jsonl", test)
    console.print(f"Wrote [bold]{len(train)}[/] train and [bold]{len(test)}[/] test trajectories to {output}")


def command_train(args: argparse.Namespace) -> None:
    rollouts = read_jsonl(args.data)
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
        f"Trained on [bold]{len(rollouts)}[/] trajectories / [bold]{training['states']}[/] state-cost pairs; final TD loss [bold]{training['final_loss']:.5f}[/]"
    )
    console.print(f"Saved policy to [cyan]{args.output}[/]")


def command_evaluate(args: argparse.Namespace) -> None:
    payload = _write_benchmark(Path(args.data), Path(args.policy), Path(args.output), args.costs)
    _print_benchmark(payload)
    if args.svg or args.html:
        write_report(args.output, args.svg, args.html)
        console.print("Rendered " + ", ".join(path for path in (args.svg, args.html) if path))


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
        f"Stopped after [bold]{final.sample_count}[/] samples with [bold]{final.majority_answer}[/] — {verdict}; gold={rollout.gold}"
    )


def command_quickstart(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path, test_path = output / "train.jsonl", output / "test.jsonl"
    policy_path, benchmark_path = output / "policy.pt", output / "benchmark.json"
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

    train = subparsers.add_parser("train", help="fit the offline Q-controller")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="artifacts/policy.pt")
    train.add_argument("--max-samples", type=int, default=8)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--costs", type=_costs, default=_costs("0,0.01,0.025,0.05,0.075,0.1,0.15,0.25"))
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate", help="compare RL with fixed and heuristic policies")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--output", default="artifacts/benchmark.json")
    evaluate.add_argument("--svg")
    evaluate.add_argument("--html")
    evaluate.add_argument("--costs", type=_costs, default=_costs("0.01,0.025,0.05,0.075,0.1,0.15"))
    evaluate.set_defaults(handler=command_evaluate)

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
    quickstart.add_argument("--costs", type=_costs, default=_costs("0.01,0.025,0.05,0.075,0.1,0.15"))
    quickstart.set_defaults(handler=command_quickstart)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
