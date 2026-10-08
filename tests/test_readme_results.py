"""The README result tables must match the committed result files they cite."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
GSM8K = json.loads((ROOT / "benchmarks/gsm8k-sampling-bootstrap.json").read_text())
MATH = json.loads((ROOT / "benchmarks/math500/result.json").read_text())

# README rule label -> committed policy name.
POLICIES = {
    "Fixed one": "fixed-1",
    "Fixed eight": "fixed-8",
    "Two consecutive matching answers": "agreement-2",
    "Learned, λ = 0.05": "BranchPilot λ=0.05",
    "MATH-trained, λ = 0.05": "BranchPilot λ=0.05",
    "GSM8K transfer, λ = 0.05": "BranchPilot λ=0.05",
}


def table_rows(header: str) -> list[list[str]]:
    lines = README[README.index(header) :].splitlines()[2:]
    rows = []
    for line in lines:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def committed_row(benchmark: str, rule: str) -> dict:
    name = POLICIES[rule]
    if benchmark == "GSM8K":
        rows = GSM8K["splits"]["test"]["rows"]
    elif rule.startswith("GSM8K transfer"):
        rows = [row for row in MATH["transfer"]["rows"] if row["scoring_cost"] == 0.05]
    else:
        rows = [row for row in MATH["test"]["rows"] if row["scoring_cost"] == 0.05]
    (row,) = [row for row in rows if row["policy"] == name]
    return row


def number(text: str) -> str:
    return text.replace("−", "-")


def test_accuracy_and_sample_table_matches_committed_results():
    rows = table_rows("| Benchmark | Rule | Accuracy | Mean samples |")
    assert len(rows) == 9
    for benchmark, rule, accuracy, samples in rows:
        row = committed_row(benchmark, rule)
        assert accuracy == f"{100 * row['accuracy']:.1f}%", (benchmark, rule)
        assert samples == f"{row['average_samples']:.2f}", (benchmark, rule)


def test_utility_table_matches_validation_frozen_comparisons():
    rows = table_rows("| λ | Learned − comparator utility | Paired 95% interval |")
    comparisons = {item["scoring_cost"]: item for item in MATH["test"]["comparisons"]}
    primary_costs = [float(cost) for cost in MATH["decision"]["utility_interval_lower_bounds"]]
    assert [float(row[0]) for row in rows] == primary_costs
    for cost, delta, interval in rows:
        comparison = comparisons[float(cost)]
        assert comparison["selection"] == "validation-frozen"
        assert comparison["baseline_policy"] == "fixed-1"
        assert number(delta) == f"{comparison['utility_delta']:.4f}"
        bounds = comparison["utility_delta_interval"]
        assert number(interval) == f"[{bounds['lower']:.4f}, {bounds['upper']:.4f}]"


def test_matched_budget_sentence_matches_committed_mixtures():
    match = re.search(
        r"At ([\d.]+) samples, the fixed-three/four mixture scored ([\d.]+)% and the "
        r"fixed-one/agreement mixture ([\d.]+)%, versus learned ([\d.]+)%",
        README,
    )
    assert match is not None
    budget, adjacent, agreement, learned = match.groups()
    items = {
        item["comparison"]: item for item in MATH["matched_budget"] if item["scoring_cost"] == 0.05
    }
    assert budget == f"{items['adjacent-fixed']['learned_mean_samples']:.2f}"
    assert adjacent == f"{100 * items['adjacent-fixed']['mixture_expected_accuracy']:.1f}"
    assert agreement == f"{100 * items['agreement-2-budget']['mixture_expected_accuracy']:.1f}"
    assert learned == f"{100 * items['adjacent-fixed']['learned_accuracy']:.1f}"
    assert [c["policy"] for c in items["adjacent-fixed"]["components"]] == ["fixed-3", "fixed-4"]
    for item in items.values():
        interval = item["accuracy_delta_interval"]
        assert interval["lower"] < 0 < interval["upper"], "README says both intervals cross zero"
