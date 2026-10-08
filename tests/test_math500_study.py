from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from branchpilot.evaluate import benchmark
from branchpilot.policy import Decision
from branchpilot.schema import Rollout, Sample

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("math500_study", ROOT / "tools/evaluate_math500.py")
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


class StoppingPolicy:
    max_samples = 8
    hidden_size = 128
    costs = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25)

    def __init__(self, stop_at=2):
        self.stop_at = stop_at

    def run(self, rollout, cost):
        count = min(self.stop_at, len(rollout.samples))
        return Decision("stop", 1.0, 0.0, count, rollout.samples[count - 1].answer)

    def save(self, path, training):
        path.write_bytes(b"test policy artifact")


@pytest.fixture
def protocol():
    payload = study.read_json(ROOT / study.PROTOCOL)
    payload["dataset"]["rows"] = 6
    payload["splits"] = {
        split: {"indices": [2 * i, 2 * i + 1], "uids": [f"uid-{2 * i}", f"uid-{2 * i + 1}"]}
        for i, split in enumerate(("train", "validation", "test"))
    }
    return payload


def rollout(index, split, correct=True, answers=None):
    answers = answers or ["sample-0"] * 8
    return Rollout(
        uid=f"uid-{index}",
        question=f"question {index}",
        gold="math500-gold-unused",
        samples=tuple(Sample("reasoning", answer, 10, parse_status="parsed") for answer in answers),
        metadata={
            "source_index": index,
            "dataset_uid": f"uid-{index}",
            "output_split": split,
            "answer_correctness": {label: correct for label in set(answers)},
        },
    )


def write_bank(root, split, records):
    path = root / study.BANK / f"{split}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record.to_dict()) + "\n" for record in records).encode()
    path.write_bytes(gzip.compress(payload, mtime=0))


def prepare(root, protocol, include_test=False):
    study.write_json(root / study.PROTOCOL, protocol)
    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    for split in splits:
        records = [rollout(index, split) for index in protocol["splits"][split]["indices"]]
        write_bank(root, split, list(reversed(records)))


def comparisons(lowers):
    return [
        {"scoring_cost": cost, "utility_delta_interval": {"lower": lower, "upper": 1.0}}
        for cost, lower in zip((0.05, 0.075, 0.1), lowers, strict=True)
    ]


@pytest.mark.parametrize(
    ("lowers", "success"),
    [
        ([0.001, 0.002, 0], True),
        ([0.001, 0, 0.002], True),
        ([0.001, 0.002, 0.003], True),
        ([0.001, 0.002, -0.001], False),
        ([0, 0, 0.001], False),
        ([0, 0, 0], False),
        ([-0.001, 0.002, 0.003], False),
    ],
)
def test_primary_success_rule_has_exact_zero_boundaries(lowers, success):
    decision = study.primary_rule(comparisons(lowers), [0.05, 0.075, 0.1])
    assert decision["success"] is success
    assert list(decision["utility_interval_lower_bounds"].values()) == lowers


@pytest.mark.parametrize(
    ("target", "low", "high", "expected"),
    [
        (1, 1, 8, (1, 0)),
        (8, 1, 8, (0, 1)),
        (3, 3, 3, (1, 0)),
        (2.5, 2, 3, (0.5, 0.5)),
        (4, 2, 8, (2 / 3, 1 / 3)),
    ],
)
def test_count_mixture_matches_exact_expected_budget(target, low, high, expected):
    weights = study.count_mixture(target, low, high)
    assert weights == pytest.approx(expected)
    assert sum(weights) == pytest.approx(1)
    assert weights[0] * low + weights[1] * high == pytest.approx(target)


@pytest.mark.parametrize("counts", [(0, 1, 8), (9, 1, 8), (3, 8, 1), (float("nan"), 1, 8)])
def test_mixture_rejects_unbracketed_counts(counts):
    with pytest.raises(ValueError):
        study.count_mixture(*counts)


def test_protocol_disjointness_includes_unopened_test_metadata(protocol):
    study.validate_protocol(protocol)
    duplicate = copy.deepcopy(protocol)
    duplicate["splits"]["test"]["indices"][0] = duplicate["splits"]["train"]["indices"][0]
    with pytest.raises(ValueError, match="disjoint"):
        study.validate_protocol(duplicate)
    duplicate = copy.deepcopy(protocol)
    duplicate["splits"]["test"]["uids"][0] = duplicate["splits"]["validation"]["uids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        study.validate_protocol(duplicate)


@pytest.mark.parametrize("change", ["uid", "index", "split", "unparsed", "count", "missing"])
def test_bank_validation_enforces_protocol_and_parser(tmp_path, protocol, change):
    prepare(tmp_path, protocol)
    first = rollout(0, "train").to_dict()
    second = rollout(1, "train").to_dict()
    if change == "uid":
        first["uid"] = "another-uid"
    elif change == "index":
        first["metadata"]["source_index"] = 4
    elif change == "split":
        first["metadata"]["output_split"] = "test"
    elif change == "unparsed":
        first["metadata"].pop("answer_correctness")
    elif change == "count":
        first["samples"] = first["samples"][:-1]
    records = [Rollout.from_dict(first)]
    if change != "missing":
        records.append(Rollout.from_dict(second))
    write_bank(tmp_path, "train", records)
    with pytest.raises(ValueError):
        study.load_bank(tmp_path, protocol, "train")


def test_bank_rows_are_reordered_to_protocol_indices(tmp_path, protocol):
    prepare(tmp_path, protocol)
    assert [item.uid for item in study.load_bank(tmp_path, protocol, "train")] == ["uid-0", "uid-1"]


def test_select_never_opens_test_and_keeps_comparator_tie_order(tmp_path, protocol, monkeypatch):
    prepare(tmp_path, protocol)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != "test.jsonl.gz", "selection opened held-out test"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(study, "protocol_at_commit", lambda root, commit: "protocol-commit")
    seen = []

    def training(records, specification):
        seen.extend(record.uid for record in records)
        assert specification["controller"] == protocol["controller"]
        return StoppingPolicy(), {"config": specification["controller"]}

    monkeypatch.setattr(study, "train_from_protocol", training)
    output = study.select(tmp_path, "protocol-commit")
    assert seen == ["uid-0", "uid-1"]
    selection = study.read_json(tmp_path / study.BANK / "selection.json")
    assert all(name == "fixed-1" for name in output["baselines"].values())
    assert "test.jsonl.gz" not in json.dumps(selection["hashes"])
    assert selection["controller"]["hidden_size"] == 128
    assert selection["controller"]["epochs"] == 120
    assert selection["controller"]["seed"] == 23
    assert selection["validation"]["bootstrap"] == {
        "resamples": 10000,
        "seed": 17,
        "confidence": 0.95,
    }


def test_training_uses_every_protocol_config_field(protocol, monkeypatch):
    import branchpilot.training as training

    seen = []

    def capture(records, config):
        seen.append(config)
        return StoppingPolicy(), {"config": asdict(config)}

    monkeypatch.setattr(training, "train_policy", capture)
    study.train_from_protocol([rollout(0, "train")], protocol)
    actual = asdict(seen[0])
    actual["costs"] = list(actual["costs"])
    assert actual == protocol["controller"]


def test_existing_benchmark_uses_correctness_mapping_not_gold_sentinel(protocol):
    records = [rollout(0, "test", True), rollout(1, "test", False)]
    result = study.evaluate_bank(records, StoppingPolicy(), protocol)
    learned = next(row for row in result.rows if row.policy == "BranchPilot λ=0.05")
    assert learned.accuracy == 0.5
    assert learned.utility == pytest.approx(0.5 - 0.05)
    assert result.policy_outcomes[0].correct == (True, False)


@pytest.mark.parametrize("stop_at", [1, 2, 5, 8])
def test_matched_comparisons_count_only_weights_and_conditional_pairing(protocol, stop_at):
    records = [rollout(i, "test", i % 2 == 0) for i in range(4)]
    result = benchmark(
        records, StoppingPolicy(stop_at), costs=[0.05], bootstrap_samples=50, bootstrap_seed=17
    )
    rows = study.matched_budgets(result, protocol)
    assert len(rows) == 2
    for row in rows:
        assert row["mixture_expected_mean_samples"] == pytest.approx(stop_at)
        assert row["learned_mean_samples"] == stop_at
        assert row["accuracy_delta"] == 0
        assert row["accuracy_delta_interval"] == {"lower": 0, "upper": 0}
        assert "whole-test" in row["interpretation"]
    alternate = [rollout(i, "test", i % 2 != 0) for i in range(4)]
    altered = benchmark(
        alternate, StoppingPolicy(stop_at), costs=[0.05], bootstrap_samples=50, bootstrap_seed=17
    )
    second = study.matched_budgets(altered, protocol)
    assert [[item["weight"] for item in row["components"]] for row in rows] == [
        [item["weight"] for item in row["components"]] for row in second
    ]


def test_matched_interval_is_a_paired_correctness_difference(protocol):
    records = [
        rollout(i, "test", i % 2 == 0, ["sample-0", "sample-1"] + ["sample-1"] * 6)
        for i in range(4)
    ]
    for record in records:
        record.metadata["answer_correctness"]["sample-0"] = False
        record.metadata["answer_correctness"]["sample-1"] = True
    result = benchmark(
        records, StoppingPolicy(3), costs=[0.05], bootstrap_samples=20, bootstrap_seed=17
    )
    matched = study.matched_budgets(result, protocol)
    fixed = next(row for row in matched if row["comparison"] == "adjacent-fixed")
    assert fixed["accuracy_delta_interval"] == {"lower": 0, "upper": 0}
    agreement = next(row for row in matched if row["comparison"] == "agreement-2-budget")
    assert agreement["mixture_expected_mean_samples"] == pytest.approx(3)
    assert all(item["weight"] >= 0 for item in agreement["components"])


def mock_git_selection(root, monkeypatch):
    selection_bytes = (root / study.BANK / "selection.json").read_bytes()
    monkeypatch.setattr(study, "resolve_commit", lambda root, commit: commit)
    monkeypatch.setattr(study, "protocol_at_commit", lambda root, commit: "protocol-commit")
    commands = []

    def git(root, *args):
        commands.append(args)
        if args[0] == "show":
            return selection_bytes
        return b""

    monkeypatch.setattr(study, "git", git)
    return commands


def selected_fixture(root, protocol, monkeypatch):
    prepare(root, protocol, include_test=True)
    monkeypatch.setattr(study, "protocol_at_commit", lambda root, commit: "protocol-commit")
    monkeypatch.setattr(
        study,
        "train_from_protocol",
        lambda rows, p: (StoppingPolicy(), {"config": p["controller"]}),
    )
    study.select(root, "protocol-commit")
    return mock_git_selection(root, monkeypatch)


def test_selection_requires_committed_selection_and_unchanged_inputs(
    tmp_path, protocol, monkeypatch
):
    commands = selected_fixture(tmp_path, protocol, monkeypatch)
    study.verify_selection(tmp_path, protocol, "selection-commit")
    assert ("merge-base", "--is-ancestor", "protocol-commit", "selection-commit") in commands
    assert ("merge-base", "--is-ancestor", "selection-commit", "HEAD") in commands
    with pytest.raises(ValueError, match="after preregistration"):
        study.verify_selection(tmp_path, protocol, "protocol-commit")
    path = tmp_path / study.BANK / "training.json"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="changed selection input"):
        study.verify_selection(tmp_path, protocol, "selection-commit")


def test_selection_rejects_uncommitted_edits(tmp_path, protocol, monkeypatch):
    selected_fixture(tmp_path, protocol, monkeypatch)
    path = tmp_path / study.BANK / "selection.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="selection differs"):
        study.verify_selection(tmp_path, protocol, "selection-commit")


def test_test_command_loads_both_policies_without_retraining(tmp_path, protocol, monkeypatch):
    selected_fixture(tmp_path, protocol, monkeypatch)
    monkeypatch.setattr(study, "train_from_protocol", lambda *args: pytest.fail("test retrained"))
    loads = []

    def load(path):
        loads.append(path.name)
        return StoppingPolicy(2 if path.name == "policy.safetensors" else 3)

    monkeypatch.setattr(study.BranchPilotPolicy, "load", load)
    (tmp_path / study.BANK / "gsm8k-policy.safetensors").write_bytes(b"original transfer policy")
    gsm = {
        "splits": {
            "test": {
                "rows": [
                    {
                        "policy": "fixed-1",
                        "accuracy": 0.7,
                        "average_samples": 1,
                        "accuracy_interval": {"lower": 0.68, "upper": 0.72},
                        "average_samples_interval": {"lower": 1, "upper": 1},
                    }
                ]
            }
        }
    }
    study.write_json(tmp_path / "benchmarks/gsm8k-sampling-bootstrap.json", gsm)
    output = study.test(tmp_path, "selection-commit")
    assert output["records"] == 2
    assert loads == ["policy.safetensors", "gsm8k-policy.safetensors"]
    result = study.read_json(tmp_path / study.BANK / "result.json")
    assert result["transfer"]["descriptive_only"] is True
    assert all(item["selection"] == "validation-frozen" for item in result["test"]["comparisons"])
    assert "policy_outcomes" not in result["test"]
    compressed = tmp_path / study.BANK / "outcomes.json.gz"
    outcomes = json.loads(gzip.decompress(compressed.read_bytes()))
    assert outcomes["uids"] == ["uid-4", "uid-5"]
    assert list(outcomes["policies"]).count("fixed-1") == 1
    assert any(name.startswith("gsm8k-transfer: BranchPilot") for name in outcomes["policies"])
    assert outcomes["policies"]["fixed-1"]["correct"] == [True, True]
    svg = (tmp_path / "assets/math500-generalization.svg").read_text()
    ET.fromstring(svg)
    assert "N=1,319" in svg and "N=200" in svg
    assert "Separate accuracy scales" in svg
    assert "GSM8K transfer" in svg
    monkeypatch.setattr(study, "select", lambda *_: pytest.fail("plot must not retrain"))
    monkeypatch.setattr(study, "test", lambda *_: pytest.fail("plot must not evaluate"))
    assert study.main(["--root", str(tmp_path), "plot"]) == 0
    assert (tmp_path / "assets/math500-generalization.svg").read_text() == svg


def test_cli_prohibits_epochs_override():
    with pytest.raises(SystemExit) as error:
        study.main(["select", "--epochs", "1"])
    assert error.value.code == 2


def test_missing_recorded_commit_explains_how_to_fetch_history(tmp_path):
    study.git(tmp_path, "init", "--quiet")
    with pytest.raises(ValueError, match="fetch full history"):
        study.resolve_commit(tmp_path, "29eea90")


def test_compact_outcomes_deduplicates_shared_baselines(protocol):
    result = study.evaluate_bank(
        [rollout(0, "test"), rollout(1, "test")], StoppingPolicy(), protocol
    )
    payload = study.compact_outcomes({"math-trained": result, "gsm8k-transfer": result})
    assert len(payload["policies"]) == 8 + 12 + 2 + 2 * 6
    assert np.asarray(payload["policies"]["fixed-8"]["samples"]).tolist() == [8, 8]


def test_matched_bootstrap_matches_manual_fixed_weight_paired_difference(protocol):
    records = []
    for index in range(4):
        answers = ["sample-0"] * 8 if index % 2 == 0 else (["sample-0"] + ["sample-1"] * 7)
        record = rollout(index, "test", answers=answers)
        record.metadata["answer_correctness"] = {
            "sample-0": index in (0, 3),
            **({"sample-1": index == 1} if index % 2 else {}),
        }
        records.append(record)

    class VaryingPolicy(StoppingPolicy):
        def run(self, rollout, cost):
            count = 2 + int(rollout.uid.rsplit("-", 1)[1]) % 2
            return Decision("stop", 1.0, 0.0, count, rollout.samples[count - 1].answer)

    result = benchmark(
        records, VaryingPolicy(), costs=[0.05], bootstrap_samples=50, bootstrap_seed=17
    )
    row = next(
        item
        for item in study.matched_budgets(result, protocol)
        if item["comparison"] == "adjacent-fixed"
    )
    assert [item["weight"] for item in row["components"]] == [0.5, 0.5]
    assert row["mixture_expected_mean_samples"] == 2.5
    expected_differences = np.array([0.0, 0.5, 0.0, -0.5])
    indices = np.random.default_rng(17).integers(0, 4, size=(10000, 4))
    expected_interval = np.quantile(expected_differences[indices].mean(axis=1), (0.025, 0.975))
    assert row["accuracy_delta"] == expected_differences.mean()
    assert list(row["accuracy_delta_interval"].values()) == pytest.approx(expected_interval)


@pytest.mark.parametrize(
    "filename", ["train.jsonl.gz", "validation.jsonl.gz", "policy.safetensors", "training.json"]
)
def test_selection_rejects_changed_bank_or_policy_inputs(tmp_path, protocol, monkeypatch, filename):
    selected_fixture(tmp_path, protocol, monkeypatch)
    path = tmp_path / study.BANK / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed selection input"):
        study.verify_selection(tmp_path, protocol, "selection-commit")
