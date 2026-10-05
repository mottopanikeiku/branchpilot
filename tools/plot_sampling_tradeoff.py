"""Plot committed per-prompt outcomes; no model loading, training, or new inference.

Run from the repository root:
    nice -n 19 uv run --frozen --no-dev python tools/plot_sampling_tradeoff.py

Outputs: assets/gsm8k-sampling-bootstrap.svg and
benchmarks/gsm8k-sampling-bootstrap.json. All strategies and both splits are
retained in JSON. The SVG shows test outcomes, including every learned cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = (
    "gsm8k-v2.json",
    "gsm8k-v2-validation.json",
    "manifest-v2.json",
    "gsm8k-v2-baselines.json",
    "protocol.json",
    "checksums-v2.txt",
)
COLORS = {"fixed": "#334155", "confidence": "#087e8b", "agreement": "#a65c00", "learned": "#9d317b"}


@dataclass(frozen=True)
class Outcomes:
    names: tuple[str, ...]
    families: tuple[str, ...]
    uids: tuple[str, ...]
    correct: np.ndarray
    samples: np.ndarray
    costs: tuple[tuple[float, ...], ...]


def family(name: str) -> str:
    for prefix in ("fixed", "confidence", "agreement"):
        if name.startswith(prefix + "-"):
            return prefix
    if name.startswith("BranchPilot λ="):
        return "learned"
    raise ValueError(f"Unknown strategy: {name}")


def extract(payload: dict[str, Any], evaluation: dict[str, Any]) -> Outcomes:
    """Align every strategy by prompt UID, never by an unchecked row position."""
    raw = payload["policy_outcomes"]
    names = tuple(item["policy"] for item in raw)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate policy outcomes")
    expected = {f"fixed-{n}" for n in evaluation["fixed_counts"]}
    expected |= {f"confidence-{t:g}" for t in evaluation["confidence_thresholds"]}
    expected |= {f"agreement-{n}" for n in evaluation["agreement_streaks"]}
    expected |= {f"BranchPilot λ={c!r}" for c in payload["costs"]}
    if set(names) != expected:
        raise ValueError(f"Missing or unexpected strategies: {set(names) ^ expected}")
    if set(evaluation["fixed_counts"]) != set(range(1, 9)):
        raise ValueError("This figure requires committed fixed counts 1–8")
    uids = tuple(raw[0]["uids"])
    n = payload["records"]
    if len(uids) != n or len(set(uids)) != n:
        raise ValueError("Prompt UIDs must be unique and match records")
    correct = np.empty((n, len(raw)), dtype=np.float64)
    samples = np.empty_like(correct)
    costs = []
    for j, item in enumerate(raw):
        if any(len(item[key]) != n for key in ("uids", "correct", "samples")):
            raise ValueError(f"Outcome length mismatch: {names[j]}")
        if len(set(item["uids"])) != n or set(item["uids"]) != set(uids):
            raise ValueError(f"Unpaired prompt UIDs: {names[j]}")
        if any(type(value) is not bool for value in item["correct"]):
            raise ValueError("Correctness must contain boolean outcomes")
        if any(type(value) is not int or not 1 <= value <= 8 for value in item["samples"]):
            raise ValueError("Sample counts must be integers within 1–8")
        positions = {uid: i for i, uid in enumerate(item["uids"])}
        order = [positions[uid] for uid in uids]
        correct[:, j] = np.asarray(item["correct"], dtype=float)[order]
        samples[:, j] = np.asarray(item["samples"], dtype=float)[order]
        if names[j].startswith("fixed-") and not np.all(samples[:, j] == int(names[j][6:])):
            raise ValueError(f"Nonconstant fixed count: {names[j]}")
        rows = [row for row in payload["rows"] if row["policy"] == names[j]]
        if not rows:
            raise ValueError(f"Missing aggregate rows: {names[j]}")
        for row in rows:
            if not np.isclose(row["accuracy"], correct[:, j].mean(), rtol=0, atol=1e-12):
                raise ValueError(f"Accuracy disagrees with outcomes: {names[j]}")
            if not np.isclose(row["average_samples"], samples[:, j].mean(), rtol=0, atol=1e-12):
                raise ValueError(f"Average samples disagree with outcomes: {names[j]}")
        costs.append(tuple(row["scoring_cost"] for row in rows))
    return Outcomes(names, tuple(map(family, names)), uids, correct, samples, tuple(costs))


def bootstrap(data: Outcomes, resamples: int, seed: int, batch_size: int = 128):
    """Same sampled UID indices for every strategy and both axes, in bounded batches.

    int64 indices and NumPy default_rng match the original evaluation's draws.
    Changing batch_size does not change the resampling sequence.
    """
    if resamples < 1 or seed < 0 or batch_size < 1:
        raise ValueError("Require positive resamples/batch size and nonnegative seed")
    n, k = data.correct.shape
    accuracy = np.empty((resamples, k))
    samples = np.empty_like(accuracy)
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, batch_size):
        end = min(start + batch_size, resamples)
        indices = rng.integers(0, n, size=(end - start, n), dtype=np.int64)
        for j in range(k):
            accuracy[start:end, j] = data.correct[:, j][indices].mean(axis=1)
            samples[start:end, j] = data.samples[:, j][indices].mean(axis=1)
    return accuracy, samples


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {"lower": float(lower), "upper": float(upper)}


def summarize(
    payload: dict[str, Any],
    evaluation: dict[str, Any],
    seed: int,
    resamples: int,
    baselines: dict[str, str],
) -> dict[str, Any]:
    data = extract(payload, evaluation)
    accuracy, samples = bootstrap(data, resamples, seed)
    rows = []
    for j, name in enumerate(data.names):
        rows.append(
            {
                "policy": name,
                "family": data.families[j],
                "prompts": len(data.uids),
                "correct_prompts": int(data.correct[:, j].sum()),
                "total_samples": int(data.samples[:, j].sum()),
                "accuracy": float(data.correct[:, j].mean()),
                "accuracy_interval": interval(accuracy[:, j]),
                "average_samples": float(data.samples[:, j].mean()),
                "average_samples_interval": interval(samples[:, j]),
                "source_scoring_costs": list(data.costs[j]),
            }
        )
    comparisons = []
    for cost in payload["costs"]:
        learned = f"BranchPilot λ={cost!r}"
        baseline = baselines[str(cost)]
        a, b = data.names.index(learned), data.names.index(baseline)
        da = accuracy[:, a] - accuracy[:, b]
        ds = samples[:, a] - samples[:, b]
        point_a = float((data.correct[:, a] - data.correct[:, b]).mean())
        point_s = float((data.samples[:, a] - data.samples[:, b]).mean())
        comparisons.append(
            {
                "cost": cost,
                "learned": learned,
                "baseline": baseline,
                "baseline_selection": (
                    "committed validation selection; not selected from this figure"
                ),
                "accuracy_delta": point_a,
                "accuracy_delta_interval": interval(da),
                "average_samples_delta": point_s,
                "average_samples_delta_interval": interval(ds),
                "utility_delta": point_a - cost * point_s,
                "utility_delta_interval": interval(da - cost * ds),
            }
        )
    uid_bytes = json.dumps(data.uids, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "prompts": len(data.uids),
        "prompt_uid_order_sha256": hashlib.sha256(uid_bytes).hexdigest(),
        "rows": rows,
        "paired_comparisons": comparisons,
    }


def fingerprint(path: Path, root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_summary(root: Path = ROOT, seed: int = 17, resamples: int = 10_000):
    paths = {name: root / "benchmarks" / name for name in INPUT_NAMES}
    fingerprints = {name: fingerprint(path, root) for name, path in paths.items()}
    checksums = dict(
        line.split(None, 1)[::-1] for line in paths["checksums-v2.txt"].read_text().splitlines()
    )
    for name in INPUT_NAMES[:-1]:
        if fingerprints[name]["sha256"] != checksums[name]:
            raise ValueError(f"Input differs from committed checksum record: {name}")
    inputs = {name: json.loads(paths[name].read_text()) for name in INPUT_NAMES[:-1]}
    test, validation = inputs["gsm8k-v2.json"], inputs["gsm8k-v2-validation.json"]
    manifest, selection = inputs["manifest-v2.json"], inputs["gsm8k-v2-baselines.json"]
    protocol = inputs["protocol.json"]
    for split, payload in (("test", test), ("validation", validation)):
        if payload["data"]["manifest"]["sha256"] != fingerprints["manifest-v2.json"]["sha256"]:
            raise ValueError(f"Manifest reference mismatch: {split}")
        if payload["data"]["sha256"] != manifest["artifacts"][split]["sha256"]:
            raise ValueError(f"Response bank reference mismatch: {split}")
        if payload["protocol"]["sha256"] != fingerprints["protocol.json"]["sha256"]:
            raise ValueError(f"Protocol reference mismatch: {split}")
        if json.loads(payload["protocol_raw"]) != protocol:
            raise ValueError(f"Protocol snapshot mismatch: {split}")
    if (
        selection["validation_benchmark"]["sha256"]
        != fingerprints["gsm8k-v2-validation.json"]["sha256"]
    ):
        raise ValueError("Baseline selection references a different validation benchmark")
    if (
        test["baseline_selection"]["artifact"]["sha256"]
        != fingerprints["gsm8k-v2-baselines.json"]["sha256"]
    ):
        raise ValueError("Test references a different baseline selection")
    if test["policy"]["sha256"] != validation["policy"]["sha256"]:
        raise ValueError("Test and validation use different learned policies")
    splits = {
        split: summarize(payload, protocol["evaluation"], seed, resamples, selection["baselines"])
        for split, payload in (("test", test), ("validation", validation))
    }
    return {
        "schema_version": 1,
        "question": (
            "Accuracy versus average samples per prompt in a committed response-bank replay"
        ),
        "inputs": list(fingerprints.values()),
        "generator": fingerprint(root / "tools" / "plot_sampling_tradeoff.py", root),
        "software": {"numpy": np.__version__},
        "provenance": {
            "dataset": manifest["dataset"],
            "model": {key: manifest["model"][key] for key in ("id", "revision", "dtype")},
            "sampling": manifest["sampling"],
            "response_banks": manifest["artifacts"],
            "learned_policy": test["policy"],
            "unavailable_in_clone": [
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "policy.safetensors",
            ],
            "scope": (
                "Recomputed from committed correctness and sample-count arrays, "
                "not raw generations or model weights."
            ),
        },
        "bootstrap": {
            "unit": "prompt",
            "paired": True,
            "resamples": resamples,
            "seed": seed,
            "confidence": 0.95,
            "rng": "numpy.random.default_rng (PCG64), int64 indices",
            "method": (
                "Percentile interval of prompt means; NumPy linear quantiles at 0.025 and 0.975"
            ),
            "coupling": (
                "Each resampled prompt retains all strategy outcomes and both axes. "
                "Splits resampled separately."
            ),
            "interpretation": (
                "Marginal 95% intervals, not simultaneous coverage or uncertainty "
                "over training/generation seeds."
            ),
        },
        "figure": {
            "split": "test",
            "families": list(COLORS),
            "learned_selection": "All six committed reported costs; no test-based filtering.",
            "fixed_counts": list(range(1, 9)),
            "available": True,
            "display": (
                "Shaded vertical bands and horizontal/vertical whiskers are marginal "
                "95% bootstrap intervals. All heuristic rules shown; "
                "overlapping rules may coincide."
            ),
        },
        "rules": {
            "fixed": "Vote after exactly n samples, n=1–8.",
            "confidence": (
                "After at least two samples, stop when leading parsed-answer votes / "
                "all observed samples reaches the threshold, or at 8."
            ),
            "confidence_thresholds": protocol["evaluation"]["confidence_thresholds"],
            "agreement": (
                "Stop after 2 or 3 consecutive identical parsed answers, or at 8; "
                "unparsed answers reset the streak."
            ),
            "learned": (
                "Use committed learned-policy outcomes at each reported lambda; "
                "objective accuracy - lambda * (samples - 1). Lambda is not a dollar price."
            ),
            "source": ["src/branchpilot/evaluate.py", "src/branchpilot/strategies.py"],
        },
        "limitations": [
            "One model, one task, one generation seed; no new model inference or training.",
            "Offline response-bank replay is not a sequential deployment experiment.",
            "Sample count is not token cost, latency, energy, GPU time, or dollars.",
            (
                "Correctness labels and stopping counts are inherited; "
                "absent raw banks/weights prevent regenerating them here."
            ),
            (
                "Prompt bootstrap conditions on the collected bank and trained policy, "
                "not model-training or generation variability."
            ),
        ],
        "splits": splits,
    }


def render_svg(summary: dict[str, Any]) -> str:
    rows = summary["splits"]["test"]["rows"]
    width, height = 1160, 790
    left, right, top, bottom = 90, 845, 160, 630

    def x(value):
        return left + (value - 0.7) / 7.6 * (right - left)

    def y(value):
        return bottom - (value - 0.65) / 0.19 * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GSM8K: accuracy versus average samples per prompt</title>',
        '<desc id="desc">Fixed counts 1 through 8, vote-confidence thresholds, '
        "agreement streaks, and all six learned-policy costs. Test outcomes for "
        "1319 prompts; marginal 95% paired prompt bootstrap intervals on both axes. "
        "Offline replay, not measured speed or dollar savings.</desc>",
        '<rect width="1160" height="790" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#17212e">',
    ]

    def text(tx, ty, value, size=14, **attrs):
        extra = " ".join(
            f'{key.replace("_", "-")}="{escape(str(val))}"' for key, val in attrs.items()
        )
        parts.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="{size}" {extra}>'
            f"{escape(str(value))}</text>"
        )

    text(55, 42, "GSM8K: accuracy vs. average samples per prompt", 25, font_weight="bold")
    text(55, 70, "Qwen2.5-1.5B-Instruct · 1,319 test prompts · committed response-bank replay", 16)
    text(
        55,
        94,
        f"95% paired prompt bootstrap · {summary['bootstrap']['resamples']:,} resamples "
        f"· seed {summary['bootstrap']['seed']}",
        14,
    )
    for i, (group, label) in enumerate(
        (
            ("fixed", "Fixed count 1–8"),
            ("confidence", "Vote confidence"),
            ("agreement", "Consecutive agreement"),
            ("learned", "Learned policy"),
        )
    ):
        lx = 60 + i * 270
        parts.append(f'<circle cx="{lx}" cy="125" r="5" fill="{COLORS[group]}"/>')
        text(lx + 13, 130, label)
    for n in range(1, 9):
        parts.append(f'<path d="M{x(n):.2f} {top}V{bottom}" stroke="#e2e8f0"/>')
        text(x(n), bottom + 25, n, text_anchor="middle")
    for tick in (0.65, 0.70, 0.75, 0.80):
        parts.append(f'<path d="M{left} {y(tick):.2f}H{right}" stroke="#e2e8f0"/>')
        text(left - 12, y(tick) + 5, f"{tick:.0%}", text_anchor="end")
    text(
        (left + right) / 2,
        bottom + 59,
        "Average samples per prompt (not time or dollars)",
        17,
        text_anchor="middle",
    )
    parts.append(
        f'<text transform="translate(28 {(top + bottom) / 2}) rotate(-90)" '
        'text-anchor="middle" font-size="17">Answer accuracy</text>'
    )
    parts.append(f'<path d="M{left} {top}V{bottom}H{right}" fill="none" stroke="#64748b"/>')
    for group in COLORS:
        members = sorted(
            (row for row in rows if row["family"] == group), key=lambda row: row["average_samples"]
        )
        color = COLORS[group]
        upper = [
            (x(row["average_samples"]), y(row["accuracy_interval"]["upper"])) for row in members
        ]
        lower = [
            (x(row["average_samples"]), y(row["accuracy_interval"]["lower"]))
            for row in reversed(members)
        ]
        points = " ".join(f"{px:.2f},{py:.2f}" for px, py in upper + lower)
        parts.append(f'<polygon points="{points}" fill="{color}" fill-opacity="0.07"/>')
        line = " ".join(
            f"{x(row['average_samples']):.2f},{y(row['accuracy']):.2f}" for row in members
        )
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        for row in members:
            px, py = x(row["average_samples"]), y(row["accuracy"])
            lo_x, hi_x = map(x, row["average_samples_interval"].values())
            lo_y, hi_y = map(y, row["accuracy_interval"].values())
            parts.append(
                f'<g stroke="{color}" stroke-opacity="0.4">'
                f'<path d="M{lo_x:.2f} {py:.2f}H{hi_x:.2f}'
                f'M{px:.2f} {lo_y:.2f}V{hi_y:.2f}"/>'
                f'<path d="M{lo_x:.2f} {py - 3:.2f}v6M{hi_x:.2f} {py - 3:.2f}v6'
                f'M{px - 3:.2f} {lo_y:.2f}h6M{px - 3:.2f} {hi_y:.2f}h6"/></g>'
            )
            parts.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}">'
                f"<title>{escape(row['policy'])}: accuracy {row['accuracy']:.6f}; "
                f"average samples {row['average_samples']:.6f}</title></circle>"
            )
            if group == "fixed":
                text(px + 8, py + 15, row["policy"][6:], 13, fill=color)
            elif group == "learned":
                text(px + 7, py + 18, row["policy"].split(" ")[1], 12, fill=color)
            elif group == "agreement":
                text(px + 8, py - 8, row["policy"], 12, fill=color)
    text(880, 177, "Committed rule settings", 16, font_weight="bold")
    text(880, 208, "Confidence thresholds", 14, fill=COLORS["confidence"])
    for i, line in enumerate(
        ("0.5, 0.55, 0.6, 0.65, 0.67,", "0.7, 0.75, 0.8, 0.85, 0.9,", "0.95, 1.0 (minimum 2)")
    ):
        text(880, 233 + i * 22, line, 13)
    text(880, 317, "Agreement streaks: 2, 3", 14, fill=COLORS["agreement"])
    text(880, 357, "Learned λ: all reported", 14, fill=COLORS["learned"])
    text(880, 382, "0.01, 0.025, 0.05,", 13)
    text(880, 404, "0.075, 0.1, 0.15", 13)
    text(880, 449, "Bands: accuracy intervals", 13)
    text(880, 472, "Whiskers: both axes", 13)
    text(880, 495, "Marginal, not simultaneous", 13)
    text(880, 541, "All rules plotted; some", 13)
    text(880, 564, "confidence points overlap.", 13)
    text(
        55,
        732,
        "One model / one task / one generation seed. "
        "Lines guide the eye; they are not additional measurements.",
        14,
    )
    text(
        55,
        758,
        "No new inference or training. Exact values, intervals and checksums: "
        "benchmarks/gsm8k-sampling-bootstrap.json",
        13,
    )
    parts.append("</g></svg>\n")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "benchmarks/gsm8k-sampling-bootstrap.json"
    )
    parser.add_argument("--svg", type=Path, default=ROOT / "assets/gsm8k-sampling-bootstrap.svg")
    args = parser.parse_args(argv)
    summary = build_summary(seed=args.seed, resamples=args.resamples)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    args.svg.write_text(render_svg(summary))
    print(f"Wrote {args.summary} and {args.svg}")
    print(
        f"{summary['splits']['test']['prompts']} test / "
        f"{summary['splits']['validation']['prompts']} validation prompts; "
        f"{args.resamples} paired resamples per split"
    )
    for row in summary["splits"]["test"]["rows"]:
        print(
            f"{row['policy']}: accuracy={row['accuracy']:.9f}, samples={row['average_samples']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
