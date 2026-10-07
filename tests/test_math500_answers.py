from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from branchpilot.answers import extract_answer
from branchpilot.evaluate import _measure
from branchpilot.features import FEATURE_NAMES, prefix_correct, prefix_state
from branchpilot.schema import Rollout, Sample, read_jsonl_bytes

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("math500_answers", ROOT / "tools/math500_answers.py")
assert SPEC is not None and SPEC.loader is not None
answers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = answers
SPEC.loader.exec_module(answers)


@pytest.fixture
def math_verify():
    return pytest.importorskip("math_verify", reason="requires the optional math extra")


def raw_rollout(*texts: str, gold: str = r"\frac{1}{2}") -> Rollout:
    return Rollout(
        uid="math-case",
        question="Find the answer.",
        gold="raw-gold-unused",
        samples=tuple(
            Sample(text, None, 12 + index, -0.5, "stop", "unparsed")
            for index, text in enumerate(texts)
        ),
        metadata={"gold_latex": gold, "dataset_uid": "test/algebra/case.json"},
    )


def test_equivalent_fractions_vote_together(math_verify) -> None:
    raw = raw_rollout(r"Thus $\boxed{\frac{1}{2}}$", r"Thus $\boxed{\frac{2}{4}}$")
    parsed = answers.parse_rollout(raw)
    assert [sample.answer for sample in parsed.samples] == ["math-answer-0"] * 2
    assert all(sample.parse_status == "parsed_explicit" for sample in parsed.samples)
    assert [sample.text for sample in parsed.samples] == [sample.text for sample in raw.samples]
    assert prefix_state(parsed, 2).top_votes == 2
    assert prefix_correct(parsed, 2)
    assert parsed.gold == answers.GOLD_SENTINEL


def test_symbolic_nonnumeric_answer_is_scored(math_verify) -> None:
    parsed = answers.parse_rollout(
        raw_rollout(r"$\boxed{x+1}$", r"$\boxed{1+x}$", gold="x+1")
    )
    assert parsed.samples[0].answer == parsed.samples[1].answer == "math-answer-0"
    assert prefix_correct(parsed, 2)


@pytest.mark.parametrize(
    "text",
    ("Evelyn", "east", "even", "(C)", "(E)", "(B)", "ellipse", "Navin"),
)
def test_dataset_textual_gold_is_parsed_without_fallback(math_verify, text: str) -> None:
    gold = rf"\text{{{text}}}"
    prediction = rf"$\boxed{{{gold}}}$"
    parsed = answers.parse_rollout(raw_rollout(prediction, prediction, gold=gold))
    assert parsed.samples[0].answer == parsed.samples[1].answer == "math-answer-0"
    assert parsed.samples[0].parse_status == "parsed_explicit"
    assert parsed.metadata["answer_correctness"] == {"math-answer-0": True}
    assert prefix_correct(parsed, 2)


def test_labels_do_not_depend_on_future_samples_or_gold(math_verify) -> None:
    raw = raw_rollout(r"$\boxed{2}$", r"$\boxed{\frac{1}{2}}$", gold="2")
    short = answers.parse_rollout(raw)
    extended = answers.parse_rollout(
        replace(raw, samples=raw.samples + (Sample(r"$\boxed{2}$", None, 3, -0.1, "stop"),))
    )
    changed_gold = answers.parse_rollout(
        replace(raw, metadata={**raw.metadata, "gold_latex": r"\frac{1}{2}"})
    )
    assert short.samples == extended.samples[:2] == changed_gold.samples
    assert short.metadata["answer_correctness"] == {"math-answer-0": True, "math-answer-1": False}
    assert changed_gold.metadata["answer_correctness"] == {
        "math-answer-0": False,
        "math-answer-1": True,
    }
    np.testing.assert_array_equal(
        prefix_state(short, 2, 8).features, prefix_state(extended, 2, 8).features
    )


def test_grouping_is_symmetric_and_uses_only_first_representatives(math_verify, monkeypatch) -> None:
    calls = []

    def parse(text, **kwargs):
        if text.startswith("$"):
            return ["gold"]
        assert kwargs["fallback_mode"] == "no_fallback"
        assert kwargs["extraction_config"][0].boxed_match_priority == 0
        return [text]

    def verify(left, right):
        a, b = left[0], right[0]
        calls.append((a, b))
        if a == "gold":
            return True
        # A and B match, B and C match, but A and C do not. D is asymmetric.
        return a == b or (a, b) in {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "D")}

    monkeypatch.setattr(math_verify, "parse", parse)
    monkeypatch.setattr(math_verify, "verify", verify)
    parsed = answers.parse_rollout(raw_rollout("A", "B", "C", "D"))
    assert [sample.answer for sample in parsed.samples] == [
        "math-answer-0",
        "math-answer-0",
        "math-answer-1",
        "math-answer-2",
    ]
    assert not any(a == "B" or b == "B" for a, b in calls[2:])
    gold_index = next(index for index, pair in enumerate(calls) if pair[0] == "gold")
    assert all(a != "gold" and b != "gold" for a, b in calls[:gold_index])
    assert all(a == "gold" for a, _ in calls[gold_index:])
    assert all(parsed.metadata["answer_correctness"].values())


def test_truncated_incomplete_and_unparsed_outputs_do_not_vote(math_verify) -> None:
    raw = raw_rollout(r"$\boxed{2}$", r"$\boxed{2}$", r"$\boxed{2}$", "The answer is 2.", gold="2")
    raw = replace(
        raw,
        samples=(
            replace(raw.samples[0], finish_reason="length", parse_status="truncated"),
            replace(raw.samples[1], parse_status="incomplete"),
            replace(raw.samples[2], finish_reason=None),
            raw.samples[3],
        ),
    )
    parsed = answers.parse_rollout(raw)
    assert [sample.parse_status for sample in parsed.samples] == [
        "truncated", "incomplete", "incomplete", "unparsed"
    ]
    assert all(sample.answer is None for sample in parsed.samples)
    assert parsed.metadata["answer_correctness"] == {}
    state = prefix_state(parsed, 4)
    assert state.majority_answer is None
    assert state.top_votes == 1
    assert state.features[FEATURE_NAMES.index("parse_rate")] == 0
    assert not prefix_correct(parsed, 4)


def test_correctness_metadata_never_changes_observable_features() -> None:
    base = Rollout(
        "labels",
        "Question?",
        answers.GOLD_SENTINEL,
        (
            Sample("first", "math-answer-0", 4, -0.4),
            Sample("second", "math-answer-1", 6, -0.1),
        ),
        metadata={"answer_correctness": {"math-answer-0": True, "math-answer-1": True}},
    )
    changed = replace(
        base, gold="unrelated", metadata={"answer_correctness": {"math-answer-0": False}}
    )
    for count in (1, 2):
        np.testing.assert_array_equal(
            prefix_state(base, count).features, prefix_state(changed, count).features
        )
        assert prefix_state(base, count).majority_answer == prefix_state(changed, count).majority_answer
        assert prefix_correct(base, count)
        assert not prefix_correct(changed, count)
    measurement = _measure([base], lambda rollout: 2, "fixed-2", "fixed", 0.05, np.array([[0]]))
    assert measurement.correct.tolist() == [1.0]
    assert measurement.metrics.accuracy == 1.0


def test_historical_numeric_rollouts_keep_equality_scoring() -> None:
    rollout = Rollout(
        "numeric",
        "What is one half?",
        "1/2",
        (
            Sample(r"\boxed{\frac{1}{2}}", extract_answer(r"\boxed{\frac{1}{2}}"), 5),
            Sample("#### 0.5", extract_answer("#### 0.5"), 3),
            Sample("#### 2", extract_answer("#### 2"), 3),
        ),
    )
    assert prefix_correct(rollout, 1)
    assert prefix_correct(rollout, 2)
    assert prefix_correct(rollout, 3)
    assert not prefix_correct(replace(rollout, gold="2"), 3)


def write_raw_banks(path: Path, rollout: Rollout) -> dict[str, bytes]:
    payloads = {}
    for split in answers.SPLITS:
        payloads[split] = answers.compressed_bank([rollout])
        (path / f"{split}.jsonl.gz").write_bytes(payloads[split])
    return payloads


def test_cli_application_preserves_text_hashes_and_refuses_reapplication(
    math_verify, tmp_path: Path
) -> None:
    raw = raw_rollout(r"$\boxed{\frac{1}{2}}$", r"$\boxed{\frac{2}{4}}$")
    originals = write_raw_banks(tmp_path, raw)
    summary = answers.apply_banks(tmp_path)
    assert summary["status"] == "complete"
    assert summary["package_versions"]["math-verify"] == "0.8.0"
    assert summary["package_versions"]["antlr4-python3-runtime"] == "4.13.2"
    assert summary["totals"]["equivalence_classes"] == 3
    assert summary["totals"]["equivalent_samples_joined"] == 3
    assert summary["totals"]["parse_status_counts"] == {"parsed_explicit": 6}
    parsed_bytes = {}
    for split in answers.SPLITS:
        encoded = (tmp_path / f"{split}.jsonl.gz").read_bytes()
        parsed_bytes[split] = encoded
        assert encoded[4:8] == bytes(4)
        assert summary["raw_bank_sha256"][split] == hashlib.sha256(originals[split]).hexdigest()
        assert summary["parsed_bank_sha256"][split] == hashlib.sha256(encoded).hexdigest()
        restored = read_jsonl_bytes(gzip.decompress(encoded))
        assert restored[0].samples[0].text == raw.samples[0].text
        assert answers.compressed_bank(restored) == encoded
    with pytest.raises(ValueError, match="Already parsed"):
        answers.apply_banks(tmp_path)
    assert all(
        (tmp_path / f"{split}.jsonl.gz").read_bytes() == payload
        for split, payload in parsed_bytes.items()
    )
    # A missing summary must not allow a partially applied bank to be relabelled.
    (tmp_path / "parser-summary.json").unlink()
    with pytest.raises(ValueError, match="Already parsed"):
        answers.apply_banks(tmp_path)


def test_gold_failure_reports_source_uid_and_leaves_all_banks_raw(math_verify, tmp_path) -> None:
    originals = write_raw_banks(tmp_path, raw_rollout(r"$\boxed{2}$", gold=""))
    with pytest.raises(answers.GoldParseError, match="test/algebra/case.json"):
        answers.apply_banks(tmp_path)
    report = json.loads((tmp_path / "parser-summary.json").read_text())
    assert report["status"] == "gold_parse_failed"
    assert report["gold_parse_failure"]["dataset_uid"] == "test/algebra/case.json"
    assert "parsed_bank_sha256" not in report
    assert all(
        (tmp_path / f"{split}.jsonl.gz").read_bytes() == payload
        for split, payload in originals.items()
    )


def test_unparseable_gold_from_library_is_an_error(math_verify, monkeypatch, tmp_path) -> None:
    raw = raw_rollout(r"$\boxed{2}$", gold="unparseable-gold")
    originals = write_raw_banks(tmp_path, raw)
    original_parse = math_verify.parse

    def parse(text, **kwargs):
        if text == "$unparseable-gold$":
            assert kwargs["fallback_mode"] == "no_fallback"
            return []
        return original_parse(text, **kwargs)

    monkeypatch.setattr(math_verify, "parse", parse)
    with pytest.raises(answers.GoldParseError, match="test/algebra/case.json"):
        answers.apply_banks(tmp_path)
    report = json.loads((tmp_path / "parser-summary.json").read_text())
    assert report["status"] == "gold_parse_failed"
    assert all(
        (tmp_path / f"{split}.jsonl.gz").read_bytes() == payload
        for split, payload in originals.items()
    )
