"""The `branchpilot audit` surface: terminal report, HTML report, and headline scoping.

Every test drives the real parser and the real handler, so the argument surface is under
test alongside the output. Logs are written per-test from a single record builder rather
than read from a shared fixture, because the properties under test -- an exact duplicate,
a multi-sample group, a model with no configured price -- are properties of one log and
should be visible in the test that depends on them.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from branchpilot.audit.detectors import MEASURED_BASIS, PROJECTION_BASIS
from branchpilot.audit.render import LIMITATIONS, NO_OPPORTUNITY
from branchpilot.audit.risk import HEADLINE_WARNING, VALIDATION_COMMANDS
from branchpilot.cli import build_parser
from branchpilot.pricing import SCHEMA_VERSION as PRICE_BOOK_SCHEMA_VERSION
from branchpilot.schema import write_jsonl
from branchpilot.synthetic import make_synthetic_rollouts

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"

AUTO_DETECTABLE = (
    "anthropic.jsonl",
    "helicone.jsonl",
    "litellm.jsonl",
    "openai.jsonl",
    "openrouter.jsonl",
)

GENERIC_MAPPING = {
    "id": "trace_id",
    "timestamp": "ts",
    "model": "llm.model",
    "provider": {"const": "openai"},
    "messages": "turns",
    "message_role_key": "speaker",
    "message_content_key": "text",
    "completion": "reply",
    "prompt_tokens": "counters.in",
    "cached_prompt_tokens": "counters.in_cached",
    "completion_tokens": "counters.out",
    "latency_ms": "took_ms",
    "status": "outcome",
    "group_key": "tenant",
}

TERMINAL_ORDER = (
    "Observed spend",
    "Spend by model",
    "Coverage",
    "Top opportunities",
    "Total addressable range",
    "Recommended next action",
)

EXTERNAL_REFERENCE = re.compile(r"src\s*=|href\s*=|url\s*\(|@import")
RANGE_FIGURE = re.compile(r'class="figure range[^"]*">([^<]+)<')


@dataclass(frozen=True, slots=True)
class Run:
    """One CLI invocation: its exit code, its flattened stdout, its operator message."""

    code: int
    out: str
    error: str


def _flat(text: str) -> str:
    """Collapse the console's line wrapping so assertions read the sentence, not the width."""
    return " ".join(text.split())


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> Run:
    code = 0
    error = ""
    try:
        args = build_parser().parse_args(list(argv))
        args.handler(args)
    except SystemExit as exit_signal:
        raw = exit_signal.code
        if raw is None:
            code = 0
        elif isinstance(raw, int):
            code = raw
        else:
            code = 1
            error = str(raw)
    captured = capsys.readouterr()
    return Run(code=code, out=_flat(captured.out), error=_flat(error or captured.err))


def _openai_record(
    identifier: str,
    *,
    user: str,
    created: int,
    system: str | None = None,
    group: str | None = None,
    prompt_tokens: int = 4000,
    completion_tokens: int = 800,
    cached_tokens: int | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """One `openai-jsonl` line. The only record builder in this file."""
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    request: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if group is not None:
        request["metadata"] = {"group": group}
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "request": request,
        "response": {
            "id": identifier,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Recorded."},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        },
        "status_code": 200,
        "latency_ms": 120.0,
    }


def _write_log(path: Path, records: list[dict[str, Any]]) -> str:
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return str(path)


def levered_log(tmp_path: Path) -> str:
    """A log carrying one IDENTICAL saving and one QUALITY_AFFECTING saving.

    ``dup-1`` repeats ``dup-0`` verbatim, so exact deduplication claims it. The three
    ``vote`` records share a group key and differ in content, so sampling claims the two
    after the first. No record carries a system prefix, which keeps the prefix cache out of
    the precedence chain and leaves both levers visible.
    """
    repeat = "Recompute the ledger total for warehouse nine."
    return _write_log(
        tmp_path / "levered.jsonl",
        [
            _openai_record("dup-0", user=repeat, created=1775030400),
            _openai_record("dup-1", user=repeat, created=1775030460),
            _openai_record("vote-0", user="Sum the first column.", created=1775030520, group="v"),
            _openai_record("vote-1", user="Sum the second column.", created=1775030580, group="v"),
            _openai_record("vote-2", user="Sum the third column.", created=1775030640, group="v"),
        ],
    )


def barren_log(tmp_path: Path) -> str:
    """Distinct requests, no shared prefix, no groups: nothing for a lever to claim."""
    return _write_log(
        tmp_path / "barren.jsonl",
        [
            _openai_record(
                f"solo-{index}",
                user=f"Question {index} about the ledger.",
                created=1775030400 + 60 * index,
            )
            for index in range(3)
        ],
    )


def cached_log(tmp_path: Path) -> str:
    """A shared system prefix the provider already served from cache."""
    prefix = "You are the ledger assistant. Answer in one sentence."
    return _write_log(
        tmp_path / "cached.jsonl",
        [
            _openai_record(
                f"warm-{index}",
                user=f"Reconcile row {index}.",
                created=1775030400 + 60 * index,
                system=prefix,
                cached_tokens=3000,
            )
            for index in range(3)
        ],
    )


def mapping_file(tmp_path: Path) -> str:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(GENERIC_MAPPING, sort_keys=True), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("name", AUTO_DETECTABLE)
def test_bare_audit_exits_zero_on_every_auto_detectable_format(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_cli(capsys, "audit", str(FIXTURES / name))

    assert run.code == 0, run.error
    for heading in TERMINAL_ORDER:
        assert heading in run.out


def test_terminal_sections_render_in_the_documented_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_cli(capsys, "audit", levered_log(tmp_path))

    positions = [run.out.index(heading) for heading in TERMINAL_ORDER]
    assert positions == sorted(positions)
    assert run.out.count("Recommended next action") == 1


def test_bare_audit_on_the_generic_format_names_the_mapping_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = run_cli(capsys, "audit", str(FIXTURES / "generic.jsonl"))

    assert run.code == 1
    assert "fix:" in run.error
    assert "--mapping" in run.error


def test_generic_format_with_an_explicit_mapping_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_cli(
        capsys,
        "audit",
        str(FIXTURES / "generic.jsonl"),
        "--format",
        "generic-jsonl",
        "--mapping",
        mapping_file(tmp_path),
    )

    assert run.code == 0, run.error
    assert "openai/gpt-4o-mini" in run.out


def test_generic_mapping_that_is_not_json_names_the_mapping_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    run = run_cli(
        capsys,
        "audit",
        str(FIXTURES / "generic.jsonl"),
        "--format",
        "generic-jsonl",
        "--mapping",
        str(broken),
    )

    assert run.code == 1
    assert "fix:" in run.error
    assert "--mapping" in run.error


def test_a_wholly_unpriced_log_reports_zero_coverage_and_still_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = run_cli(capsys, "audit", str(FIXTURES / "openrouter.jsonl"))

    assert run.code == 0, run.error
    assert "priced 0 of 6 record(s) (0.0%)" in run.out
    assert "deepinfra/meta-llama/llama-3.3-70b-instruct" in run.out
    assert "fix: add those (provider, model) pairs to a price book file" in run.out
    assert "Traceback" not in run.out


def test_a_log_with_no_opportunity_summarizes_instead_of_rendering_a_blank_page(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.html"
    run = run_cli(capsys, "audit", barren_log(tmp_path), "--html", str(report))

    assert run.code == 0, run.error
    assert NO_OPPORTUNITY in run.out
    document = report.read_text(encoding="utf-8")
    assert NO_OPPORTUNITY in document
    assert "Total addressable range" in document
    assert "Recommended next action" in document
    assert "Limitations" in document


def test_html_is_byte_identical_across_two_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = cached_log(tmp_path)
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"

    assert run_cli(capsys, "audit", source, "--html", str(first)).code == 0
    assert run_cli(capsys, "audit", source, "--html", str(second)).code == 0

    assert first.read_bytes() == second.read_bytes()


def test_html_references_no_external_resource(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.html"
    run_cli(capsys, "audit", levered_log(tmp_path), "--html", str(report))

    document = report.read_text(encoding="utf-8")
    assert EXTERNAL_REFERENCE.search(document) is None
    assert document.startswith("<!DOCTYPE html>")


def test_every_money_range_in_the_html_is_two_bounds_with_a_labelled_basis(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.html"
    run_cli(capsys, "audit", cached_log(tmp_path), "--html", str(report))

    document = report.read_text(encoding="utf-8")
    figures = RANGE_FIGURE.findall(document)
    assert figures
    for figure in figures:
        assert " to " in figure
    assert PROJECTION_BASIS in document
    assert MEASURED_BASIS in document


def test_html_embeds_the_evidence_needed_to_recompute_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = levered_log(tmp_path)
    report = tmp_path / "report.html"
    run_cli(capsys, "audit", source, "--html", str(report))

    document = report.read_text(encoding="utf-8")
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    assert digest in document
    assert "Records read" in document
    assert f"<dd>{PRICE_BOOK_SCHEMA_VERSION}</dd>" in document
    assert "2026-08-27" in document
    assert f"branchpilot audit {source} --format openai-jsonl --currency USD" in document
    for limitation in LIMITATIONS:
        assert html.escape(limitation, quote=True) in document


def test_default_headline_excludes_quality_affecting_levers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "audit.json"
    run = run_cli(capsys, "audit", levered_log(tmp_path), "--json", str(ledger))

    assert run.code == 0, run.error
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["headline"]["levers"] == ["exact_dedup"]
    assert payload["headline"]["risk_classes"] == ["IDENTICAL"]
    assert payload["headline"]["warning"] is None
    assert payload["include_quality_affecting"] is False
    assert "risk classes in scope: IDENTICAL" in run.out
    assert HEADLINE_WARNING not in run.out


def test_include_quality_affecting_widens_the_headline_and_adds_the_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = levered_log(tmp_path)
    narrow_ledger = tmp_path / "narrow.json"
    wide_ledger = tmp_path / "wide.json"

    run_cli(capsys, "audit", source, "--json", str(narrow_ledger))
    run = run_cli(
        capsys, "audit", source, "--include-quality-affecting", "--json", str(wide_ledger)
    )

    assert run.code == 0, run.error
    narrow = json.loads(narrow_ledger.read_text(encoding="utf-8"))["headline"]
    wide = json.loads(wide_ledger.read_text(encoding="utf-8"))["headline"]
    assert wide["levers"] == ["exact_dedup", "sampling"]
    assert wide["risk_classes"] == ["IDENTICAL", "QUALITY_AFFECTING"]
    assert wide["warning"] == HEADLINE_WARNING
    assert Decimal(wide["high"]) > Decimal(narrow["high"]) > 0
    assert _flat(HEADLINE_WARNING) in run.out


def test_quality_affecting_levers_carry_the_exact_validation_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.html"
    run = run_cli(capsys, "audit", levered_log(tmp_path), "--html", str(report))

    document = report.read_text(encoding="utf-8")
    assert "Quality-affecting levers" in run.out
    for lever, command in sorted(VALIDATION_COMMANDS.items()):
        assert lever in document
        assert command in document
        assert _flat(command) in run.out


def test_headline_html_warns_only_when_quality_affecting_levers_are_in_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = levered_log(tmp_path)
    narrow = tmp_path / "narrow.html"
    wide = tmp_path / "wide.html"

    run_cli(capsys, "audit", source, "--html", str(narrow))
    run_cli(capsys, "audit", source, "--include-quality-affecting", "--html", str(wide))

    assert HEADLINE_WARNING not in narrow.read_text(encoding="utf-8")
    assert HEADLINE_WARNING in wide.read_text(encoding="utf-8")


def test_window_narrows_the_deduplication_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = levered_log(tmp_path)
    whole_file = tmp_path / "whole.json"
    narrow = tmp_path / "narrow.json"

    run_cli(capsys, "audit", source, "--json", str(whole_file))
    run = run_cli(capsys, "audit", source, "--window", "30", "--json", str(narrow))

    assert run.code == 0, run.error
    assert json.loads(whole_file.read_text(encoding="utf-8"))["window_seconds"] is None
    assert json.loads(narrow.read_text(encoding="utf-8"))["window_seconds"] == 30
    assert json.loads(whole_file.read_text(encoding="utf-8"))["headline"]["levers"] == [
        "exact_dedup"
    ]
    assert json.loads(narrow.read_text(encoding="utf-8"))["headline"]["levers"] == []


def test_window_rejects_a_non_positive_value(capsys: pytest.CaptureFixture[str]) -> None:
    run = run_cli(capsys, "audit", str(FIXTURES / "openai.jsonl"), "--window", "0")

    assert run.code == 2
    assert "fix:" in run.error


def test_currency_that_the_price_book_does_not_quote_names_the_fix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = run_cli(capsys, "audit", str(FIXTURES / "openai.jsonl"), "--currency", "EUR")

    assert run.code == 1
    assert "fix:" in run.error
    assert "USD" in run.error


def test_a_missing_price_book_fails_with_a_fix_rather_than_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_cli(
        capsys,
        "audit",
        str(FIXTURES / "openai.jsonl"),
        "--price-book",
        str(tmp_path / "absent.json"),
    )

    assert run.code == 1
    assert "fix:" in run.error


def test_json_output_records_the_scope_and_the_single_next_action(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "audit.json"
    run = run_cli(capsys, "audit", levered_log(tmp_path), "--json", str(ledger))

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["format"] == "openai-jsonl"
    assert payload["currency"] == "USD"
    assert [item["lever"] for item in payload["opportunities"]] == [
        "exact_dedup",
        "prefix_cache",
        "batch_lane",
        "semantic_dedup",
        "tier_routing",
        "sampling",
    ]
    action = payload["recommended_next_action"]
    assert action.startswith("adopt exact_dedup")
    assert _flat(action) in run.out


def test_integrity_subcommand_profiles_trajectories_exactly_as_audit_used_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "train.jsonl"
    compare = tmp_path / "test.jsonl"
    record = tmp_path / "integrity.json"
    write_jsonl(data, make_synthetic_rollouts(6, max_samples=2, seed=1))
    write_jsonl(compare, make_synthetic_rollouts(6, max_samples=2, seed=2))

    run = run_cli(
        capsys,
        "integrity",
        "--data",
        str(data),
        "--compare",
        str(compare),
        "--json-output",
        str(record),
    )

    assert run.code == 0, run.error
    assert "Trajectory integrity audit" in run.out
    assert "dataset SHA-256" in run.out
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["comparison"]["disjoint"] is True
    assert payload["profile"]["record_count"] == 6


def test_audit_no_longer_accepts_the_trajectory_integrity_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "train.jsonl"
    write_jsonl(data, make_synthetic_rollouts(4, max_samples=2, seed=1))

    run = run_cli(capsys, "audit", "--data", str(data))

    assert run.code == 2
