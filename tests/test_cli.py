import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from branchpilot.cli import (
    _assert_distinct_paths,
    build_parser,
    command_plan,
    command_quickstart,
    command_split,
)
from branchpilot.schema import write_jsonl
from branchpilot.synthetic import make_synthetic_rollouts


def test_every_cli_help_surface_renders() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    help_by_command = {
        name: command_parser.format_help() for name, command_parser in subparsers.choices.items()
    }

    assert set(help_by_command) == {
        "audit",
        "demo",
        "evaluate",
        "integrity",
        "plan",
        "quickstart",
        "report",
        "split",
        "synthetic",
        "train",
    }
    assert "95% upper bound" in help_by_command["plan"]
    assert "second dataset that must be disjoint" in help_by_command["integrity"]
    assert "path to the traffic log to audit" in help_by_command["audit"]
    assert "--include-quality-affecting" in help_by_command["audit"]
    top_level = parser.format_help()
    assert "validate trajectory integrity" in top_level
    assert "size the spend opportunity" in top_level


def test_distinct_path_guard_rejects_normalized_hardlink_and_symlink_aliases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("source", encoding="utf-8")
    hardlink = tmp_path / "hardlink.jsonl"
    hardlink.hardlink_to(source)
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(source)

    for alias in (tmp_path / "." / "source.jsonl", hardlink, symlink):
        with pytest.raises(ValueError, match="must be distinct"):
            _assert_distinct_paths(source=source, output=alias)


def test_split_rejects_source_output_alias_before_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, make_synthetic_rollouts(6, max_samples=2))
    original = source.read_bytes()
    args = Namespace(
        data=str(source),
        train_output=str(source),
        test_output=str(tmp_path / "test.jsonl"),
        train_size=3,
        test_size=3,
        seed=17,
        manifest=str(tmp_path / "split.json"),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        command_split(args)

    assert source.read_bytes() == original
    assert not (tmp_path / "test.jsonl").exists()
    assert not (tmp_path / "split.json").exists()


def test_quickstart_uses_a_cost_from_custom_grid(tmp_path: Path, capsys) -> None:
    args = Namespace(
        output_dir=str(tmp_path / "quickstart"),
        train_size=16,
        validation_size=6,
        test_size=8,
        max_samples=3,
        epochs=2,
        seed=17,
        costs=(0.1, 0.2),
    )

    command_quickstart(args)

    output = capsys.readouterr().out
    assert "λ=0.2" in output
    root = tmp_path / "quickstart"
    assert (root / "validation-report.html").is_file()
    assert len((root / "validation.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    assert len((root / "test.jsonl").read_text(encoding="utf-8").splitlines()) == 8
    benchmark = json.loads((root / "validation-benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["data"]["split"] == "validation"


def test_plan_command_exports_and_displays_selection_source(tmp_path: Path, capsys) -> None:
    payload = {
        "schema_version": 2,
        "data": {"split": "validation"},
        "max_samples": 4,
        "rows": [
            {
                "family": "heuristic",
                "policy": "confidence-0.75",
                "scoring_cost": 0.1,
                "accuracy": 0.9,
                "accuracy_interval": {"lower": 0.85, "upper": 0.95},
                "average_samples": 1.7,
                "average_samples_interval": {"lower": 1.6, "upper": 1.8},
                "average_tokens": 120.0,
                "average_tokens_interval": {"lower": 110.0, "upper": 130.0},
                "utility": 0.83,
            }
        ],
    }
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "plan.json"
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    args = Namespace(
        benchmark=str(benchmark),
        sample_budget=2.0,
        policy=None,
        point_estimate=False,
        family=["heuristic"],
        json_output=str(output),
    )

    command_plan(args)

    exported = json.loads(output.read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    assert exported["schema_version"] == 1
    assert exported["selection_source"] == {
        "benchmark_schema_version": 2,
        "payload_sha256": digest,
    }
    assert f"payload SHA-256 {digest[:12]}" in capsys.readouterr().out
