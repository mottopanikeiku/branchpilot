from __future__ import annotations

import copy
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sampling_tradeoff_plot", ROOT / "tools" / "plot_sampling_tradeoff.py"
)
assert SPEC is not None and SPEC.loader is not None
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)


def small_payload():
    evaluation = {
        "fixed_counts": list(range(1, 9)),
        "confidence_thresholds": [0.5, 0.75, 1.0],
        "agreement_streaks": [2, 3],
    }
    names = [f"fixed-{n}" for n in range(1, 9)]
    names += ["confidence-0.5", "confidence-0.75", "confidence-1"]
    names += ["agreement-2", "agreement-3", "BranchPilot λ=0.05"]
    outcomes, rows = [], []
    for name in names:
        counts = [int(name[6:])] * 4 if name.startswith("fixed-") else [1, 2, 3, 4]
        item = {
            "policy": name, "uids": ["a", "b", "c", "d"],
            "correct": [True, False, True, False], "samples": counts,
        }
        outcomes.append(item)
        rows.append({
            "policy": name, "accuracy": 0.5, "average_samples": sum(counts) / 4,
            "scoring_cost": 0.05,
        })
    return {"records": 4, "costs": [0.05], "policy_outcomes": outcomes,
            "rows": rows}, evaluation


def test_extraction_aligns_reordered_uids():
    payload, evaluation = small_payload()
    reordered = copy.deepcopy(payload)
    item = reordered["policy_outcomes"][-1]
    for key in ("uids", "correct", "samples"):
        item[key].reverse()
    original = plot.extract(payload, evaluation)
    aligned = plot.extract(reordered, evaluation)
    np.testing.assert_array_equal(aligned.correct, original.correct)
    np.testing.assert_array_equal(aligned.samples, original.samples)


def test_extraction_requires_every_fixed_count():
    payload, evaluation = small_payload()
    payload["policy_outcomes"] = [
        item for item in payload["policy_outcomes"] if item["policy"] != "fixed-7"
    ]
    with pytest.raises(ValueError, match="Missing or unexpected strategies"):
        plot.extract(payload, evaluation)


@pytest.mark.parametrize("bad", ["duplicate_uid", "wrong_length", "nonconstant_fixed"])
def test_extraction_rejects_unpaired_or_invalid_outcomes(bad):
    payload, evaluation = small_payload()
    if bad == "duplicate_uid":
        payload["policy_outcomes"][-1]["uids"][-1] = "a"
    elif bad == "wrong_length":
        payload["policy_outcomes"][-1]["correct"].pop()
    else:
        payload["policy_outcomes"][0]["samples"][-1] = 2
    with pytest.raises(ValueError):
        plot.extract(payload, evaluation)


def test_bootstrap_is_prompt_paired_and_batch_independent():
    payload, evaluation = small_payload()
    data = plot.extract(payload, evaluation)
    first = plot.bootstrap(data, resamples=73, seed=17, batch_size=7)
    second = plot.bootstrap(data, resamples=73, seed=17, batch_size=73)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    indices = np.random.default_rng(17).integers(0, 4, size=(73, 4), dtype=np.int64)
    np.testing.assert_array_equal(first[0][:, 0], data.correct[:, 0][indices].mean(axis=1))
    # All synthetic strategies have identical correctness. Independent draws
    # per strategy would break this equality and give nonzero paired deltas.
    assert np.all(first[0] == first[0][:, :1])
    for j in range(8):
        assert plot.interval(first[1][:, j]) == {"lower": j + 1, "upper": j + 1}


def test_summary_preserves_learned_costs_and_paired_deltas():
    payload, evaluation = small_payload()
    summary = plot.summarize(payload, evaluation, 17, 73, {"0.05": "fixed-1"})
    assert summary["prompts"] == 4
    assert len(summary["rows"]) == len(payload["policy_outcomes"])
    learned = next(row for row in summary["rows"] if row["family"] == "learned")
    assert learned["source_scoring_costs"] == [0.05]
    comparison = summary["paired_comparisons"][0]
    assert comparison["accuracy_delta_interval"] == {"lower": 0, "upper": 0}
    assert comparison["average_samples_delta"] == 1.5
    assert comparison["utility_delta"] == pytest.approx(-0.075)


@pytest.fixture(scope="module")
def committed_summary():
    return plot.build_summary(ROOT, seed=17, resamples=40)


def test_committed_inputs_cover_both_splits_and_all_settings(committed_summary):
    assert committed_summary["splits"]["test"]["prompts"] == 1319
    assert committed_summary["splits"]["validation"]["prompts"] == 400
    for split in committed_summary["splits"].values():
        assert len(split["rows"]) == 28
        assert sum(row["family"] == "learned" for row in split["rows"]) == 6
        assert {row["policy"] for row in split["rows"] if row["family"] == "fixed"} == {
            f"fixed-{n}" for n in range(1, 9)
        }
    assert all(len(item["sha256"]) == 64 for item in committed_summary["inputs"])


def test_script_writes_deterministic_machine_readable_json_and_svg(
    committed_summary, monkeypatch, tmp_path
):
    monkeypatch.setattr(plot, "build_summary", lambda **kwargs: committed_summary)
    summary_path, svg_path = tmp_path / "summary.json", tmp_path / "plot.svg"
    args = ["--resamples", "40", "--summary", str(summary_path), "--svg", str(svg_path)]
    assert plot.main(args) == 0
    first_json, first_svg = summary_path.read_bytes(), svg_path.read_bytes()
    assert json.loads(first_json) == committed_summary
    svg = ET.fromstring(first_svg)
    assert svg.tag == "{http://www.w3.org/2000/svg}svg"
    assert "not time or dollars" in first_svg.decode()
    assert "Marginal, not simultaneous" in first_svg.decode()
    assert plot.main(args) == 0
    assert summary_path.read_bytes() == first_json
    assert svg_path.read_bytes() == first_svg
