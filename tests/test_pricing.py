from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from branchpilot.pricing import (
    PACKAGED_PRICE_BOOK,
    SCHEMA_VERSION,
    STALENESS_HORIZON_DAYS,
    PriceBook,
    PriceBookError,
    PriceEntry,
    StalenessWarning,
    UnknownModelError,
)

PRICING_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "branchpilot" / "pricing"

GOOD_ENTRY: dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "input": "3",
    "output": "15",
    "cached_input": "0.30",
    "batch_input": "1.50",
    "batch_output": "7.50",
    "currency": "USD",
    "effective_date": "2026-08-27",
    "source_url": "https://platform.claude.com/docs/en/about-claude/pricing",
}


def write_book(tmp_path: Path, entries: list[dict[str, Any]], **document: Any) -> Path:
    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "entries": entries}
    payload.update(document)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def entry_with(**overrides: Any) -> dict[str, Any]:
    entry = dict(GOOD_ENTRY)
    for key, value in overrides.items():
        if value is None:
            entry.pop(key)
        else:
            entry[key] = value
    return entry


def load_expecting_error(tmp_path: Path, entries: list[dict[str, Any]], **document: Any) -> str:
    path = write_book(tmp_path, entries, **document)
    with pytest.raises(PriceBookError) as caught:
        PriceBook.load(path)
    return str(caught.value)


# --- the packaged rate cards ------------------------------------------------


def test_packaged_book_loads_clean() -> None:
    book = PriceBook.load()

    assert book.entries
    assert book.providers == ("anthropic", "gemini", "openai")
    assert book == PriceBook.load(PACKAGED_PRICE_BOOK)


def test_packaged_entries_are_exact_sourced_and_ordered() -> None:
    book = PriceBook.load()

    for entry in book.entries:
        for name in ("input", "output", "cached_input", "batch_input", "batch_output"):
            rate = getattr(entry, name)
            assert isinstance(rate, Decimal)
            assert rate.is_finite()
            assert rate >= 0
        assert entry.cached_input <= entry.input
        assert entry.batch_input <= entry.input
        assert entry.batch_output <= entry.output
        assert entry.currency == "USD"
        assert isinstance(entry.effective_date, date)
        parts = urlsplit(entry.source_url)
        assert parts.scheme in {"http", "https"}
        assert parts.netloc


def test_packaged_rates_are_stored_as_json_strings() -> None:
    document = json.loads(PACKAGED_PRICE_BOOK.read_text(encoding="utf-8"))

    assert document["schema_version"] == SCHEMA_VERSION
    for raw in document["entries"]:
        for name in ("input", "output", "cached_input", "batch_input", "batch_output"):
            assert isinstance(raw[name], str), f"{raw['model']}.{name} must be a quoted rate"


def test_packaged_anthropic_seed_matches_published_rates() -> None:
    book = PriceBook.load()
    expected = {
        "claude-opus-4-6": ("5", "25", "0.50", "2.50", "12.50"),
        "claude-sonnet-4-6": ("3", "15", "0.30", "1.50", "7.50"),
        "claude-haiku-4-5": ("1", "5", "0.10", "0.50", "2.50"),
    }

    for model, rates in expected.items():
        entry = book.entry("anthropic", model)
        assert (
            entry.input,
            entry.output,
            entry.cached_input,
            entry.batch_input,
            entry.batch_output,
        ) == tuple(Decimal(value) for value in rates)
        assert entry.batch_input == entry.input / 2
        assert entry.batch_output == entry.output / 2
        assert entry.cached_input == entry.input / 10


def test_packaged_anthropic_multipliers_hold_across_the_seed() -> None:
    """Anthropic publishes batch at 50% of both rates and a cache read at 10% of input."""
    book = PriceBook.load()

    models = book.models("anthropic")
    assert len(models) == 9
    for model in models:
        entry = book.entry("anthropic", model)
        assert entry.batch_input == entry.input / 2, model
        assert entry.batch_output == entry.output / 2, model
        assert entry.cached_input == entry.input / 10, model


# --- arithmetic -------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # 1M input at $3 + 100k output at $15 = 3.00 + 1.50
        ({"tokens_in": 1_000_000, "tokens_out": 100_000}, "4.50"),
        # 2k fresh at $3 + 8k cached at $0.30 + 2k output at $15 = 0.006 + 0.0024 + 0.030
        (
            {"tokens_in": 10_000, "tokens_out": 2_000, "cached_in": 8_000},
            "0.0384",
        ),
        # batch lane: 1M input at $1.50 + 100k output at $7.50
        ({"tokens_in": 1_000_000, "tokens_out": 100_000, "batch": True}, "2.25"),
        # batch lane keeps the cache-read rate: 500k at $1.50 + 500k cached at $0.30 + 0 out
        (
            {"tokens_in": 1_000_000, "tokens_out": 0, "cached_in": 500_000, "batch": True},
            "0.90",
        ),
        # a fully cached prompt bills only the cache-read rate
        ({"tokens_in": 1_000, "tokens_out": 0, "cached_in": 1_000}, "0.0003"),
        # zero usage is free, not an error
        ({"tokens_in": 0, "tokens_out": 0}, "0"),
        # single tokens stay exact rather than collapsing to zero
        ({"tokens_in": 1, "tokens_out": 1}, "0.000018"),
    ],
)
def test_hand_computed_costs_match_exactly(kwargs: dict[str, Any], expected: str) -> None:
    book = PriceBook.load()

    result = book.price("anthropic", "claude-sonnet-4-6", **kwargs)

    assert isinstance(result, Decimal)
    assert result == Decimal(expected)


def test_batch_never_reuses_the_synchronous_rates() -> None:
    book = PriceBook.load()
    usage = {"tokens_in": 750_000, "tokens_out": 250_000}

    synchronous = book.price("openai", "gpt-4o", **usage)
    batched = book.price("openai", "gpt-4o", batch=True, **usage)

    assert synchronous == Decimal("4.375")
    assert batched == Decimal("2.1875")
    assert batched * 2 == synchronous


def test_cached_tokens_are_removed_from_the_fresh_input_charge() -> None:
    book = PriceBook.load()

    all_fresh = book.price("anthropic", "claude-opus-4-6", tokens_in=100_000, tokens_out=0)
    all_cached = book.price(
        "anthropic", "claude-opus-4-6", tokens_in=100_000, tokens_out=0, cached_in=100_000
    )

    assert all_fresh == Decimal("0.5")
    assert all_cached == Decimal("0.05")
    assert all_cached * 10 == all_fresh


# --- unknown models ---------------------------------------------------------


def test_unknown_model_raises_with_close_matches() -> None:
    book = PriceBook.load()

    with pytest.raises(UnknownModelError) as caught:
        book.price("anthropic", "claude-sonnet-4.6", tokens_in=10, tokens_out=10)

    message = str(caught.value)
    assert "claude-sonnet-4-6" in message
    assert "fix:" in message
    assert "--price-book" in message
    assert "No price is ever estimated" in message


def test_unknown_provider_lists_configured_providers() -> None:
    book = PriceBook.load()

    with pytest.raises(UnknownModelError) as caught:
        book.entry("vertex", "claude-sonnet-4-6")

    message = str(caught.value)
    assert "anthropic" in message
    assert "openai" in message
    assert "fix:" in message


def test_unknown_model_without_close_matches_still_lists_options() -> None:
    book = PriceBook.load()

    with pytest.raises(UnknownModelError) as caught:
        book.entry("gemini", "zzzzzzzz")

    message = str(caught.value)
    assert "gemini-3.5-flash" in message
    assert "fix:" in message


def test_unknown_model_error_is_a_price_book_error() -> None:
    assert issubclass(UnknownModelError, PriceBookError)


# --- token count guards -----------------------------------------------------


@pytest.mark.parametrize("name", ["tokens_in", "tokens_out", "cached_in"])
def test_non_integer_token_count_raises(name: str) -> None:
    book = PriceBook.load()
    usage: dict[str, Any] = {"tokens_in": 10, "tokens_out": 10, "cached_in": 0}
    usage[name] = 10.5

    with pytest.raises(PriceBookError) as caught:
        book.price("anthropic", "claude-sonnet-4-6", **usage)

    message = str(caught.value)
    assert name in message
    assert "fix:" in message


@pytest.mark.parametrize("value", [True, Decimal("10"), "10", None])
def test_non_int_token_types_raise(value: Any) -> None:
    book = PriceBook.load()

    with pytest.raises(PriceBookError) as caught:
        book.price("anthropic", "claude-sonnet-4-6", tokens_in=value, tokens_out=0)

    assert "fix:" in str(caught.value)


def test_negative_token_count_raises() -> None:
    book = PriceBook.load()

    with pytest.raises(PriceBookError) as caught:
        book.price("anthropic", "claude-sonnet-4-6", tokens_in=-1, tokens_out=0)

    message = str(caught.value)
    assert "negative" in message
    assert "fix:" in message


def test_cached_in_above_tokens_in_raises() -> None:
    book = PriceBook.load()

    with pytest.raises(PriceBookError) as caught:
        book.price("anthropic", "claude-sonnet-4-6", tokens_in=100, tokens_out=0, cached_in=101)

    message = str(caught.value)
    assert "cached_in 101 exceeds tokens_in 100" in message
    assert "fix:" in message


def test_non_bool_batch_flag_raises() -> None:
    book = PriceBook.load()

    with pytest.raises(PriceBookError) as caught:
        book.price("anthropic", "claude-sonnet-4-6", tokens_in=1, tokens_out=1, batch="yes")

    assert "fix:" in str(caught.value)


# --- load-time validation, one failing fixture per rule ---------------------


@pytest.mark.parametrize(
    ("entries", "fragment"),
    [
        pytest.param([entry_with(input=3.0)], "quoted JSON string", id="unquoted-rate"),
        pytest.param([entry_with(output=15)], "quoted JSON string", id="integer-rate"),
        pytest.param([entry_with(input="NaN")], "not finite", id="nan-rate"),
        pytest.param([entry_with(output="Infinity")], "not finite", id="infinite-rate"),
        pytest.param([entry_with(input="-3")], "is negative", id="negative-rate"),
        pytest.param([entry_with(input="not-a-rate")], "not a decimal rate", id="unparsable-rate"),
        pytest.param(
            [entry_with(cached_input="3.01")],
            "cached_input 3.01 above input 3",
            id="cached-above-input",
        ),
        pytest.param(
            [entry_with(batch_input="3.5")],
            "batch_input 3.5 above input 3",
            id="batch-input-above-input",
        ),
        pytest.param(
            [entry_with(batch_output="15.5")],
            "batch_output 15.5 above output 15",
            id="batch-output-above-output",
        ),
        pytest.param([entry_with(currency="usd")], "3-letter uppercase", id="lowercase-currency"),
        pytest.param([entry_with(currency="US")], "3-letter uppercase", id="short-currency"),
        pytest.param([entry_with(currency="DOLLAR")], "3-letter uppercase", id="long-currency"),
        pytest.param(
            [entry_with(effective_date="27-08-2026")],
            "ISO-8601 day",
            id="non-iso-date",
        ),
        pytest.param(
            [entry_with(effective_date="2026-8-27")],
            "ISO-8601 day",
            id="unpadded-date",
        ),
        pytest.param(
            [entry_with(effective_date="2026-02-30")],
            "not a real calendar day",
            id="impossible-date",
        ),
        pytest.param(
            [entry_with(source_url="/docs/pricing")],
            "absolute http(s) URL",
            id="relative-url",
        ),
        pytest.param(
            [entry_with(source_url="ftp://example.com/pricing")],
            "absolute http(s) URL",
            id="non-http-url",
        ),
        pytest.param([entry_with(source_url="")], "no source_url", id="empty-url"),
        pytest.param([entry_with(provider="")], "non-empty string", id="empty-provider"),
        pytest.param([entry_with(model=" gpt-5 ")], "surrounding whitespace", id="padded-model"),
        pytest.param([GOOD_ENTRY, dict(GOOD_ENTRY)], "duplicate entries", id="duplicate-pair"),
        pytest.param([entry_with(currency=None)], "missing required field", id="missing-field"),
        pytest.param([entry_with(region="eu")], "unknown field", id="unknown-field"),
        pytest.param([], "no entries", id="empty-entries"),
    ],
)
def test_each_validation_rule_has_a_fix_clause(
    tmp_path: Path, entries: list[dict[str, Any]], fragment: str
) -> None:
    message = load_expecting_error(tmp_path, entries)

    assert fragment in message
    assert "fix:" in message


def test_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    message = load_expecting_error(tmp_path, [GOOD_ENTRY], schema_version=SCHEMA_VERSION + 1)

    assert f"declares schema_version {SCHEMA_VERSION + 1}" in message
    assert "fix:" in message


def test_non_integer_schema_version_is_rejected(tmp_path: Path) -> None:
    message = load_expecting_error(tmp_path, [GOOD_ENTRY], schema_version="1")

    assert "non-integer schema_version" in message
    assert "fix:" in message


def test_unknown_document_field_is_rejected(tmp_path: Path) -> None:
    message = load_expecting_error(tmp_path, [GOOD_ENTRY], vendor="acme")

    assert "unknown field(s) vendor" in message
    assert "fix:" in message


def test_entries_must_be_an_array(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "entries": {"a": 1}}), encoding="utf-8"
    )

    with pytest.raises(PriceBookError) as caught:
        PriceBook.load(path)

    assert "must be a JSON array" in str(caught.value)
    assert "fix:" in str(caught.value)


def test_entry_must_be_an_object(tmp_path: Path) -> None:
    message = load_expecting_error(tmp_path, ["claude-sonnet-4-6"])

    assert "is not a JSON object" in message
    assert "fix:" in message


def test_malformed_json_names_the_position(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text('{"schema_version": 1, "entries": [', encoding="utf-8")

    with pytest.raises(PriceBookError) as caught:
        PriceBook.load(path)

    message = str(caught.value)
    assert "not valid JSON" in message
    assert "line 1" in message
    assert "fix:" in message


def test_non_object_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(PriceBookError) as caught:
        PriceBook.load(path)

    assert "must be a JSON object" in str(caught.value)
    assert "fix:" in str(caught.value)


def test_missing_price_book_file_names_the_flag(tmp_path: Path) -> None:
    with pytest.raises(PriceBookError) as caught:
        PriceBook.load(tmp_path / "absent.json")

    message = str(caught.value)
    assert "does not exist" in message
    assert "--price-book" in message
    assert "fix:" in message


def test_override_book_is_used_instead_of_the_packaged_one(tmp_path: Path) -> None:
    path = write_book(tmp_path, [entry_with(input="99", output="99")])

    book = PriceBook.load(path)

    assert book.price("anthropic", "claude-sonnet-4-6", tokens_in=1_000_000, tokens_out=0) == (
        Decimal("99")
    )
    assert book.providers == ("anthropic",)


# --- direct construction is guarded too -------------------------------------


def test_price_entry_rejects_inexact_rates() -> None:
    with pytest.raises(PriceBookError) as caught:
        PriceEntry(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input=3.0,
            output=Decimal("15"),
            cached_input=Decimal("0.30"),
            batch_input=Decimal("1.50"),
            batch_output=Decimal("7.50"),
            currency="USD",
            effective_date=date(2026, 8, 27),
            source_url=GOOD_ENTRY["source_url"],
        )

    message = str(caught.value)
    assert "not exact" in message
    assert "fix:" in message


def test_price_entry_rejects_datetime_as_effective_date() -> None:
    with pytest.raises(PriceBookError) as caught:
        PriceEntry(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input=Decimal("3"),
            output=Decimal("15"),
            cached_input=Decimal("0.30"),
            batch_input=Decimal("1.50"),
            batch_output=Decimal("7.50"),
            currency="USD",
            effective_date=datetime(2026, 8, 27),
            source_url=GOOD_ENTRY["source_url"],
        )

    assert "fix:" in str(caught.value)


def test_price_book_entries_are_immutable() -> None:
    book = PriceBook.load()

    assert isinstance(book.entries, tuple)
    with pytest.raises(FrozenInstanceError):
        book.entries[0].input = Decimal("1")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        book.entries = ()  # type: ignore[misc]


def test_models_for_unknown_provider_is_empty() -> None:
    book = PriceBook.load()

    assert book.models("vertex") == ()
    assert "claude-sonnet-4-6" in book.models("anthropic")


# --- staleness --------------------------------------------------------------


def test_staleness_flags_a_deliberately_old_entry(tmp_path: Path) -> None:
    today = date(2026, 8, 27)
    stale_day = today - timedelta(days=STALENESS_HORIZON_DAYS + 100)
    path = write_book(
        tmp_path,
        [
            entry_with(effective_date=today.isoformat()),
            entry_with(model="claude-haiku-4-5", effective_date=stale_day.isoformat()),
        ],
    )
    book = PriceBook.load(path)

    warnings = book.staleness_warnings(today=today)

    assert len(warnings) == 1
    warning = warnings[0]
    assert isinstance(warning, StalenessWarning)
    assert warning.entry.model == "claude-haiku-4-5"
    assert warning.age_days == STALENESS_HORIZON_DAYS + 100
    assert warning.horizon_days == STALENESS_HORIZON_DAYS
    assert "fix:" in warning.message()
    assert warning.entry.source_url in warning.message()


@pytest.mark.parametrize(
    ("age_days", "flagged"),
    [
        (0, False),
        (STALENESS_HORIZON_DAYS - 1, False),
        (STALENESS_HORIZON_DAYS, False),
        (STALENESS_HORIZON_DAYS + 1, True),
    ],
)
def test_staleness_horizon_boundary(tmp_path: Path, age_days: int, flagged: bool) -> None:
    today = date(2026, 8, 27)
    path = write_book(
        tmp_path, [entry_with(effective_date=(today - timedelta(days=age_days)).isoformat())]
    )

    warnings = PriceBook.load(path).staleness_warnings(today=today)

    assert bool(warnings) is flagged


def test_staleness_orders_oldest_first(tmp_path: Path) -> None:
    today = date(2026, 8, 27)
    path = write_book(
        tmp_path,
        [
            entry_with(model="a", effective_date=(today - timedelta(days=200)).isoformat()),
            entry_with(model="b", effective_date=(today - timedelta(days=400)).isoformat()),
        ],
    )

    warnings = PriceBook.load(path).staleness_warnings(today=today)

    assert [warning.entry.model for warning in warnings] == ["b", "a"]


def test_staleness_defaults_to_the_current_day() -> None:
    assert isinstance(PriceBook.load().staleness_warnings(), tuple)


def test_staleness_rejects_a_non_date_today() -> None:
    with pytest.raises(PriceBookError) as caught:
        PriceBook.load().staleness_warnings(today="2026-08-27")

    assert "fix:" in str(caught.value)


def test_staleness_warning_refuses_to_describe_a_fresh_entry() -> None:
    book = PriceBook.load()

    with pytest.raises(ValueError):
        StalenessWarning(entry=book.entries[0], age_days=1, horizon_days=STALENESS_HORIZON_DAYS)


# --- the cost path carries no inexact arithmetic ----------------------------


def test_pricing_package_has_no_inexact_arithmetic() -> None:
    sources = sorted(PRICING_PACKAGE.rglob("*.py"))

    assert sources, f"expected python sources under {PRICING_PACKAGE}"
    offenders: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "float(" in line or re.search(r"(?<![\w.]) float (?![\w])", line):
                offenders.append(f"{source.name}:{lineno}: {line.strip()}")

    assert offenders == []


def test_every_public_error_message_names_a_fix(tmp_path: Path) -> None:
    book = PriceBook.load()
    messages: list[str] = []

    for call in (
        lambda: book.entry("vertex", "x"),
        lambda: book.entry("anthropic", "nope"),
        lambda: book.price("anthropic", "claude-sonnet-4-6", tokens_in=-1, tokens_out=0),
        lambda: book.price("anthropic", "claude-sonnet-4-6", tokens_in=1.5, tokens_out=0),
        lambda: book.price(
            "anthropic", "claude-sonnet-4-6", tokens_in=1, tokens_out=0, cached_in=2
        ),
        lambda: book.staleness_warnings(today=0),
        lambda: PriceBook.load(tmp_path / "absent.json"),
    ):
        with pytest.raises(PriceBookError) as caught:
            call()
        messages.append(str(caught.value))

    assert messages
    for message in messages:
        assert "fix:" in message, message
