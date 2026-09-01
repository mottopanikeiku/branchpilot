"""Risk classification and the headline range rule.

There are exactly two risk classes, they are fixed by what a lever does, and the default
headline sums only the response-identical ones. Folding the quality-affecting levers in is
possible but never silent: it widens the range and attaches the warning.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from branchpilot.audit import (
    HEADLINE_WARNING,
    IDENTICAL,
    LEVER_RISK,
    LEVERS,
    PRECEDENCE,
    QUALITY_AFFECTING,
    RISK_CLASSES,
    VALIDATION_COMMANDS,
    VALIDATION_REQUIREMENTS,
    Headline,
    Opportunity,
    Range,
    detect_opportunities,
    headline,
    price_records,
    quality_affecting_levers,
)
from branchpilot.ingest.formats import EMPTY_TEXT_HASH, RequestRecord
from branchpilot.pricing import PriceBook

PRICE_BOOK = Path(__file__).resolve().parent / "fixtures" / "audit" / "prices.json"

BASE = 1_700_000_000.0
HASH_ONE = "a1" * 16
HASH_TWO = "b2" * 16
HASH_THREE = "c3" * 16

LARGE_SPEND = Decimal("0.013")


@pytest.fixture(scope="module")
def book() -> PriceBook:
    return PriceBook.load(PRICE_BOOK)


def _record(**overrides: Any) -> RequestRecord:
    """The one place this file builds a RequestRecord."""
    fields: dict[str, Any] = {
        "id": "req",
        "timestamp": BASE,
        "model": "acme-large",
        "provider": "acme",
        "messages_hash": HASH_ONE,
        "system_prefix_hash": EMPTY_TEXT_HASH,
        "prompt_tokens": 1000,
        "cached_prompt_tokens": None,
        "completion_tokens": 100,
        "latency_ms": None,
        "status": "ok",
        "group_key": None,
        "raw_index": 0,
        "system_prefix_chars": 0,
    }
    fields.update(overrides)
    return RequestRecord(**fields)


def _ready(lever: str, low: str, high: str | None = None) -> Opportunity:
    span = Range(low=Decimal(low), high=Decimal(high if high is not None else low))
    return Opportunity(
        lever=lever,
        risk_class=LEVER_RISK[lever],
        eligible_requests=1,
        eligible_spend=span.high,
        projected_saving=span.low,
        confidence_interval=span,
        assumptions=("projection from observed tokens times configured prices",),
        required_changes=("adopt the lever",),
        status="ready",
    )


def _blocked(lever: str, status: str) -> Opportunity:
    return Opportunity(
        lever=lever,
        risk_class=LEVER_RISK[lever],
        eligible_requests=1,
        eligible_spend=LARGE_SPEND,
        projected_saving=None,
        confidence_interval=None,
        assumptions=(),
        required_changes=(f"fix: supply what {status} names",),
        status=status,
    )


# --- the two classes --------------------------------------------------------


def test_there_are_exactly_two_risk_classes() -> None:
    assert RISK_CLASSES == (IDENTICAL, QUALITY_AFFECTING)
    assert (IDENTICAL, QUALITY_AFFECTING) == ("IDENTICAL", "QUALITY_AFFECTING")


def test_every_lever_has_a_fixed_class() -> None:
    assert set(LEVER_RISK) == set(LEVERS) == set(PRECEDENCE)
    assert LEVER_RISK == {
        "exact_dedup": IDENTICAL,
        "prefix_cache": IDENTICAL,
        "batch_lane": IDENTICAL,
        "semantic_dedup": QUALITY_AFFECTING,
        "tier_routing": QUALITY_AFFECTING,
        "sampling": QUALITY_AFFECTING,
    }


def test_precedence_is_the_documented_order() -> None:
    assert PRECEDENCE == (
        "exact_dedup",
        "prefix_cache",
        "batch_lane",
        "semantic_dedup",
        "tier_routing",
        "sampling",
    )


def test_quality_affecting_levers_are_listed_in_precedence_order() -> None:
    assert quality_affecting_levers() == ("semantic_dedup", "tier_routing", "sampling")


def test_an_opportunity_cannot_claim_the_wrong_class() -> None:
    with pytest.raises(ValueError, match="is IDENTICAL, not QUALITY_AFFECTING"):
        Opportunity(
            lever="exact_dedup",
            risk_class=QUALITY_AFFECTING,
            eligible_requests=1,
            eligible_spend=LARGE_SPEND,
            projected_saving=LARGE_SPEND,
            confidence_interval=Range(low=LARGE_SPEND, high=LARGE_SPEND),
            assumptions=("a",),
            required_changes=("b",),
            status="ready",
        )


def test_every_quality_affecting_lever_names_its_validation() -> None:
    assert set(VALIDATION_COMMANDS) == set(quality_affecting_levers())
    assert set(VALIDATION_REQUIREMENTS) == set(quality_affecting_levers())
    for lever in quality_affecting_levers():
        assert VALIDATION_COMMANDS[lever].startswith("branchpilot audit ")
        assert "--include-quality-affecting" in VALIDATION_COMMANDS[lever]
        assert VALIDATION_REQUIREMENTS[lever]


def test_the_headline_warning_names_a_fix() -> None:
    assert "fix:" in HEADLINE_WARNING
    assert "QUALITY_AFFECTING" in HEADLINE_WARNING


# --- the headline range -----------------------------------------------------


def test_the_default_headline_excludes_quality_affecting_levers() -> None:
    opportunities = (_ready("exact_dedup", "0.013"), _ready("sampling", "0.026"))

    found = headline(opportunities)

    assert found.low == Decimal("0.013")
    assert found.high == Decimal("0.013")
    assert found.levers == ("exact_dedup",)
    assert found.include_quality_affecting is False
    assert found.warning is None


def test_the_flag_widens_the_range_and_adds_the_warning() -> None:
    opportunities = (_ready("exact_dedup", "0.013"), _ready("sampling", "0.026"))

    found = headline(opportunities, include_quality_affecting=True)

    assert found.low == Decimal("0.039")
    assert found.high == Decimal("0.039")
    assert found.levers == ("exact_dedup", "sampling")
    assert found.warning == HEADLINE_WARNING


def test_the_headline_keeps_both_bounds_apart() -> None:
    found = headline((_ready("exact_dedup", "0.013", "0.026"),))

    assert found.low == Decimal("0.013")
    assert found.high == Decimal("0.026")
    assert not found.empty


def test_the_headline_sums_in_precedence_order() -> None:
    opportunities = (
        _ready("batch_lane", "0.001"),
        _ready("sampling", "0.002"),
        _ready("exact_dedup", "0.004"),
    )

    found = headline(opportunities, include_quality_affecting=True)

    assert found.levers == ("exact_dedup", "batch_lane", "sampling")
    assert found.low == Decimal("0.007")


def test_realized_savings_are_never_folded_into_the_headline() -> None:
    realized = Opportunity(
        lever="prefix_cache",
        risk_class=IDENTICAL,
        eligible_requests=2,
        eligible_spend=Decimal("0.0188"),
        projected_saving=Decimal("0.0072"),
        confidence_interval=Range(low=Decimal("0.0072"), high=Decimal("0.0072")),
        assumptions=("measured from the log",),
        required_changes=("keep the prefix stable",),
        status="realized",
    )

    for flag in (False, True):
        found = headline((realized,), include_quality_affecting=flag)

        assert found.levers == ()
        assert found.low == Decimal(0)
        assert found.empty


def test_blocked_levers_contribute_nothing() -> None:
    opportunities = (
        _blocked("batch_lane", "needs_eligibility_rule"),
        _blocked("semantic_dedup", "needs_embeddings"),
        _blocked("tier_routing", "needs_evaluation"),
    )

    found = headline(opportunities, include_quality_affecting=True)

    assert found.empty
    assert found.levers == ()
    assert found.low == Decimal(0)
    assert found.high == Decimal(0)


def test_the_headline_flag_must_be_a_boolean() -> None:
    with pytest.raises(TypeError, match="include_quality_affecting must be True or False"):
        headline((), include_quality_affecting="yes")  # type: ignore[arg-type]


def test_a_headline_cannot_omit_the_warning_it_owes() -> None:
    with pytest.raises(ValueError, match="must carry the warning"):
        Headline(
            low=Decimal(0),
            high=Decimal("0.013"),
            levers=("sampling",),
            include_quality_affecting=True,
            warning=None,
        )


def test_a_headline_cannot_warn_about_levers_it_excluded() -> None:
    with pytest.raises(ValueError, match="must carry the warning"):
        Headline(
            low=Decimal(0),
            high=Decimal("0.013"),
            levers=("exact_dedup",),
            include_quality_affecting=False,
            warning=HEADLINE_WARNING,
        )


def test_headline_money_stays_decimal() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        Headline(
            low=0.0,  # type: ignore[arg-type]
            high=0.0,  # type: ignore[arg-type]
            levers=(),
            include_quality_affecting=False,
            warning=None,
        )


# --- against a real log -----------------------------------------------------


def test_a_real_log_quarantines_its_quality_affecting_saving(book: PriceBook) -> None:
    records = (
        _record(id="a", raw_index=0, timestamp=BASE),
        _record(id="b", raw_index=1, timestamp=BASE + 1),
        _record(id="c", raw_index=2, timestamp=BASE + 2, messages_hash=HASH_TWO, group_key="t"),
        _record(id="d", raw_index=3, timestamp=BASE + 3, messages_hash=HASH_THREE, group_key="t"),
    )
    items, unpriced = price_records(records, book)
    assert unpriced == ()

    opportunities, _ = detect_opportunities(items, book=book)
    default = headline(opportunities)
    widened = headline(opportunities, include_quality_affecting=True)

    assert default.levers == ("exact_dedup",)
    assert default.low == LARGE_SPEND
    assert default.warning is None
    assert widened.levers == ("exact_dedup", "sampling")
    assert widened.low == LARGE_SPEND * 2
    assert widened.high > default.high
    assert widened.warning == HEADLINE_WARNING
