"""Parse the three raw MATH banks once, without changing their generation text.

Run with: uv run --extra math python tools/math500_answers.py --input-dir benchmarks/math500
A second application is an error. Gold parse failures leave all banks untouched and
are recorded in parser-summary.json with the dataset UID.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from branchpilot.artifacts import atomic_write_bytes, atomic_write_text
from branchpilot.schema import Rollout, read_jsonl_bytes

PARSER_NAME = "math-verify-0.8.0"
GOLD_SENTINEL = "math500-gold-unused"
SPLITS = ("train", "validation", "test")


class GoldParseError(ValueError):
    def __init__(self, rollout: Rollout) -> None:
        self.uid = rollout.uid
        self.dataset_uid = str(rollout.metadata.get("dataset_uid", rollout.uid))
        super().__init__(f"Cannot parse gold for source UID {self.dataset_uid} ({self.uid})")


def _require_raw(rollout: Rollout) -> None:
    if (
        "math500_parser" in rollout.metadata
        or "answer_correctness" in rollout.metadata
        or any(sample.answer is not None for sample in rollout.samples)
        or any((sample.parse_status or "").startswith("parsed") for sample in rollout.samples)
    ):
        raise ValueError(f"Already parsed or not a raw MATH rollout: {rollout.uid}")


def parse_rollout(rollout: Rollout) -> Rollout:
    """Assign sample-only labels in order, then score their representatives.

    Equivalence is symmetric, and only the first representative of each earlier
    class is compared. Neither gold nor later samples can change an earlier label.
    """
    from math_verify import LatexExtractionConfig, parse, verify

    _require_raw(rollout)
    representatives: list[Any] = []
    samples = []
    for sample in rollout.samples:
        if sample.finish_reason == "length" or sample.parse_status == "truncated":
            samples.append(replace(sample, answer=None, parse_status="truncated"))
            continue
        if sample.finish_reason != "stop" or sample.parse_status == "incomplete":
            samples.append(replace(sample, answer=None, parse_status="incomplete"))
            continue
        parsed = parse(
            sample.text,
            extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
            fallback_mode="no_fallback",
        )
        if not parsed:
            samples.append(replace(sample, answer=None, parse_status="unparsed"))
            continue
        label_index = len(representatives)
        for index, representative in enumerate(representatives):
            if verify(representative, parsed) and verify(parsed, representative):
                label_index = index
                break
        if label_index == len(representatives):
            representatives.append(parsed)
        samples.append(
            replace(sample, answer=f"math-answer-{label_index}", parse_status="parsed_explicit")
        )

    # Gold is deliberately inaccessible to the sample grouping above.
    gold_latex = rollout.metadata.get("gold_latex")
    if not isinstance(gold_latex, str) or not gold_latex.strip():
        raise GoldParseError(rollout)
    gold = parse(
        f"${gold_latex}$",
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
    )
    if not gold:
        raise GoldParseError(rollout)
    correctness = {
        f"math-answer-{index}": bool(verify(gold, representative))
        for index, representative in enumerate(representatives)
    }
    return replace(
        rollout,
        gold=GOLD_SENTINEL,
        samples=tuple(samples),
        metadata={
            **rollout.metadata,
            "answer_correctness": correctness,
            "math500_parser": PARSER_NAME,
        },
    )


def compressed_bank(rollouts: list[Rollout]) -> bytes:
    """Encode sorted JSONL with a filename-free gzip header and fixed timestamp."""
    payload = "".join(
        json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
        for record in rollouts
    ).encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        handle.write(payload)
    return output.getvalue()


def _counts(rollouts: list[Rollout]) -> dict[str, Any]:
    statuses: Counter[str] = Counter(
        sample.parse_status or "missing" for rollout in rollouts for sample in rollout.samples
    )
    classes = sum(len(rollout.metadata["answer_correctness"]) for rollout in rollouts)
    return {
        "problems": len(rollouts),
        "samples": sum(statuses.values()),
        "parse_status_counts": dict(sorted(statuses.items())),
        "parsed_samples": statuses["parsed_explicit"],
        "equivalence_classes": classes,
        "equivalent_samples_joined": statuses["parsed_explicit"] - classes,
        "correct_equivalence_classes": sum(
            sum(rollout.metadata["answer_correctness"].values()) for rollout in rollouts
        ),
        "gold_parsed": len(rollouts),
    }


def _package_versions() -> dict[str, str]:
    versions = {
        name: version(name)
        for name in ("math-verify", "antlr4-python3-runtime", "latex2sympy2-extended", "sympy")
    }
    for name, expected in (("math-verify", "0.8.0"), ("antlr4-python3-runtime", "4.13.2")):
        if versions[name] != expected:
            raise ValueError(f"The fixed parser requires {name}=={expected}, got {versions[name]}")
    return versions


def apply_banks(input_dir: Path) -> dict[str, Any]:
    """Preflight every raw bank, parse all gold, then replace the three files."""
    summary_path = input_dir / "parser-summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            raise ValueError("Already parsed: parser-summary.json records a complete application")
    packages = _package_versions()
    raw_hashes = {}
    raw_banks = {}
    for split in SPLITS:
        payload = (input_dir / f"{split}.jsonl.gz").read_bytes()
        raw_hashes[split] = hashlib.sha256(payload).hexdigest()
        raw_banks[split] = read_jsonl_bytes(gzip.decompress(payload), f"{split}.jsonl.gz")
        for rollout in raw_banks[split]:
            _require_raw(rollout)

    summary: dict[str, Any] = {
        "schema": 1,
        "parser": PARSER_NAME,
        "package_versions": packages,
        "prediction": "LatexExtractionConfig(boxed_match_priority=0); fallback_mode=no_fallback",
        "gold": "LatexExtractionConfig(); fallback_mode=no_fallback; $gold_latex$",
        "votes": "symmetric verification against first matching earlier representative",
        "raw_bank_sha256": raw_hashes,
    }
    parsed_banks = {}
    try:
        for split in SPLITS:
            parsed_banks[split] = [parse_rollout(rollout) for rollout in raw_banks[split]]
    except GoldParseError as exc:
        summary.update(
            status="gold_parse_failed",
            gold_parse_failure={"uid": exc.uid, "dataset_uid": exc.dataset_uid, "error": str(exc)},
        )
        atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        raise

    encoded = {split: compressed_bank(parsed_banks[split]) for split in SPLITS}
    summary.update(
        status="complete",
        parsed_bank_sha256={split: hashlib.sha256(encoded[split]).hexdigest() for split in SPLITS},
        parsed_bank_bytes={split: len(encoded[split]) for split in SPLITS},
        splits={split: _counts(parsed_banks[split]) for split in SPLITS},
        totals=_counts([rollout for split in SPLITS for rollout in parsed_banks[split]]),
    )
    for split in SPLITS:
        atomic_write_bytes(input_dir / f"{split}.jsonl.gz", encoded[split])
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = apply_banks(args.input_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"status": summary["status"], "totals": summary["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
