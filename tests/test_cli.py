import argparse
from argparse import Namespace
from pathlib import Path

import pytest

from branchpilot.cli import (
    _assert_distinct_paths,
    build_parser,
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
        "plan",
        "quickstart",
        "report",
        "split",
        "synthetic",
        "train",
    }
    assert "95% upper bound" in help_by_command["plan"]


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
        test_size=8,
        max_samples=3,
        epochs=2,
        seed=17,
        costs=(0.1, 0.2),
    )

    command_quickstart(args)

    output = capsys.readouterr().out
    assert "λ=0.2" in output
    assert (tmp_path / "quickstart" / "report.html").is_file()
