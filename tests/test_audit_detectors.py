"""Detector arithmetic, precedence, and the refusal to project without evidence.

Every expectation below is hand-computed against ``tests/fixtures/audit/prices.json``, whose
rates are chosen so the arithmetic is checkable by eye. For ``acme/acme-large`` at 1,000
prompt tokens and 100 completion tokens:

* synchronous:  1000 x $10/1M + 100 x $30/1M = $0.010 + $0.003 = **$0.013**
* batch lane:   1000 x  $4/1M + 100 x  $6/1M = $0.004 + $0.0006 = **$0.0046**
* 400 cached:   600 x $10/1M + 400 x $1/1M + 100 x $30/1M = **$0.0094**

so the batch saving is $0.013 - $0.0046 = **$0.0084** (not half of spend, which is $0.0065)
and a 400-token cache read has already saved $0.013 - $0.0094 = **$0.0036**.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from branchpilot.audit import (
    BLOCKED_STATUSES,
    IDENTICAL,
    LEVER_RISK,
    MEASURED_BASIS,
    PRECEDENCE,
    PROJECTION_BASIS,
    QUALITY_AFFECTING,
    STATUSES,
    AuditResult,
    Opportunity,
    Overlap,
    PriceBookProvenance,
    Priced,
    Range,
    detect_batch_lane,
    detect_exact_dedup,
    detect_opportunities,
    detect_prefix_cache,
    detect_sampling,
    detect_semantic_dedup,
    detect_tier_routing,
    price_records,
    profile_workload,
    spend_by_model,
    sum_money,
)
from branchpilot.ingest.formats import EMPTY_TEXT_HASH, IngestReport, RequestRecord
from branchpilot.pricing import PriceBook, UnknownModelError

PRICE_BOOK = Path(__file__).resolve().parent / "fixtures" / "audit" / "prices.json"

BASE = 1_700_000_000.0
HASH_ONE = "a1" * 16
HASH_TWO = "b2" * 16
PREFIX_ONE = "c3" * 16
PREFIX_TWO = "d4" * 16

LARGE_SPEND = Decimal("0.013")
LARGE_CACHED_SPEND = Decimal("0.0094")
LARGE_CACHED_SAVING = Decimal("0.0036")
LARGE_BATCH_SAVING = Decimal("0.0084")
SMALL_SPEND = Decimal("0.0013")
SOLO_SPEND = Decimal("0.0024")


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


def _priced(book: PriceBook, *records: RequestRecord) -> tuple[Priced, ...]:
    items, unpriced = price_records(records, book)
    assert unpriced == ()
    return items


def _repeats(book: PriceBook, count: int, **overrides: Any) -> tuple[Priced, ...]:
    """``count`` identical requests, one second apart."""
    return _priced(
        book,
        *(
            _record(id=f"req-{index}", raw_index=index, timestamp=BASE + index, **overrides)
            for index in range(count)
        ),
    )


def _by_lever(opportunities: tuple[Opportunity, ...]) -> dict[str, Opportunity]:
    return {item.lever: item for item in opportunities}


# --- exact_dedup ------------------------------------------------------------


def test_exact_dedup_projects_the_whole_duplicate_spend(book: PriceBook) -> None:
    found = detect_exact_dedup(_repeats(book, 3))

    assert found.lever == "exact_dedup"
    assert found.risk_class == IDENTICAL
    assert found.status == "ready"
    assert found.eligible_requests == 2
    assert found.eligible_spend == LARGE_SPEND * 2
    assert found.projected_saving == Decimal("0.026")
    assert found.confidence_interval == Range(low=Decimal("0.026"), high=Decimal("0.026"))
    assert PROJECTION_BASIS in found.assumptions


def test_exact_dedup_ignores_distinct_requests(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0),
        _record(id="b", raw_index=1, timestamp=BASE + 1, messages_hash=HASH_TWO),
    )

    found = detect_exact_dedup(items)

    assert found.status == "no_opportunity"
    assert found.eligible_requests == 0
    assert found.projected_saving == Decimal(0)


def test_exact_dedup_window_excludes_a_late_repeat(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0),
        _record(id="b", raw_index=1, timestamp=BASE + 120),
    )

    assert detect_exact_dedup(items, window_seconds=60).projected_saving == Decimal(0)
    assert detect_exact_dedup(items, window_seconds=180).projected_saving == LARGE_SPEND


def test_exact_dedup_window_appears_in_the_assumptions(book: PriceBook) -> None:
    found = detect_exact_dedup(_repeats(book, 2), window_seconds=60)

    assert any("within a 60-second window" in phrase for phrase in found.assumptions)


def test_exact_dedup_high_bound_adds_a_failed_repeat(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0),
        _record(id="b", raw_index=1, timestamp=BASE + 1),
        _record(id="c", raw_index=2, timestamp=BASE + 2, status="error"),
    )

    found = detect_exact_dedup(items)

    assert found.eligible_requests == 2
    assert found.confidence_interval == Range(low=LARGE_SPEND, high=Decimal("0.026"))
    assert found.projected_saving == LARGE_SPEND
    assert not found.confidence_interval.degenerate


# --- prefix_cache -----------------------------------------------------------


def test_prefix_cache_reports_realized_savings(book: PriceBook) -> None:
    items = _repeats(
        book, 2, system_prefix_hash=PREFIX_ONE, system_prefix_chars=400, cached_prompt_tokens=400
    )

    found = detect_prefix_cache(items)

    assert found.risk_class == IDENTICAL
    assert found.status == "realized"
    assert found.eligible_requests == 2
    assert found.eligible_spend == LARGE_CACHED_SPEND * 2
    assert found.projected_saving == Decimal("0.0072")
    assert found.confidence_interval == Range(low=Decimal("0.0072"), high=Decimal("0.0072"))
    assert MEASURED_BASIS in found.assumptions
    assert PROJECTION_BASIS not in found.assumptions
    assert any("800 of 2000 prompt tokens (40.0%)" in phrase for phrase in found.assumptions)


def test_prefix_cache_measures_the_share_over_reporting_records_only(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, system_prefix_hash=PREFIX_ONE, cached_prompt_tokens=400),
        _record(
            id="b",
            raw_index=1,
            timestamp=BASE + 1,
            messages_hash=HASH_TWO,
            system_prefix_hash=PREFIX_ONE,
        ),
    )

    found = detect_prefix_cache(items)

    assert found.status == "realized"
    assert found.eligible_requests == 2
    assert found.eligible_spend == LARGE_CACHED_SPEND + LARGE_SPEND
    assert found.projected_saving == LARGE_CACHED_SAVING
    assert any("400 of 1000 prompt tokens (40.0%)" in phrase for phrase in found.assumptions)
    assert any("1 of 2 eligible" in phrase for phrase in found.assumptions)


def test_prefix_cache_without_cached_counts_projects_nothing(book: PriceBook) -> None:
    found = detect_prefix_cache(_repeats(book, 2, system_prefix_hash=PREFIX_ONE))

    assert found.status == "needs_token_counts"
    assert found.projected_saving is None
    assert found.confidence_interval is None
    assert found.assumptions == ()
    assert found.eligible_requests == 2
    assert found.eligible_spend == LARGE_SPEND * 2
    assert any(phrase.startswith("fix:") for phrase in found.required_changes)


def test_prefix_cache_with_zero_cache_reads_projects_nothing(book: PriceBook) -> None:
    found = detect_prefix_cache(
        _repeats(book, 2, system_prefix_hash=PREFIX_ONE, cached_prompt_tokens=0)
    )

    assert found.status == "needs_token_counts"
    assert found.projected_saving is None
    assert any("reports no cache reads" in phrase for phrase in found.required_changes)


def test_prefix_cache_ignores_the_empty_prefix(book: PriceBook) -> None:
    found = detect_prefix_cache(_repeats(book, 3, cached_prompt_tokens=400))

    assert found.status == "no_opportunity"
    assert found.eligible_requests == 0
    assert found.projected_saving == Decimal(0)


def test_prefix_cache_needs_a_prefix_shared_by_two_requests(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, system_prefix_hash=PREFIX_ONE, cached_prompt_tokens=400),
        _record(
            id="b",
            raw_index=1,
            timestamp=BASE + 1,
            system_prefix_hash=PREFIX_TWO,
            cached_prompt_tokens=400,
        ),
    )

    assert detect_prefix_cache(items).status == "no_opportunity"


# --- batch_lane -------------------------------------------------------------


def test_batch_lane_without_a_predicate_projects_nothing(book: PriceBook) -> None:
    found = detect_batch_lane(_repeats(book, 2))

    assert found.risk_class == IDENTICAL
    assert found.status == "needs_eligibility_rule"
    assert found.projected_saving is None
    assert found.confidence_interval is None
    assert found.eligible_requests == 0
    assert found.eligible_spend == Decimal(0)
    assert any(phrase.startswith("fix:") for phrase in found.required_changes)


def test_batch_lane_prices_the_saving_from_the_batch_rates(book: PriceBook) -> None:
    items = _repeats(book, 2)

    found = detect_batch_lane(items, predicate=lambda record: True)

    assert found.status == "ready"
    assert found.eligible_requests == 2
    assert found.eligible_spend == Decimal("0.026")
    assert found.projected_saving == LARGE_BATCH_SAVING * 2
    assert found.projected_saving == Decimal("0.0168")
    assert found.projected_saving != found.eligible_spend / 2


def test_batch_lane_counts_exactly_the_predicate_matches(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, group_key="offline-eval"),
        _record(id="b", raw_index=1, timestamp=BASE + 1, group_key="interactive"),
    )

    found = detect_batch_lane(items, predicate=lambda record: record.group_key == "offline-eval")

    assert found.eligible_requests == 1
    assert found.eligible_spend == LARGE_SPEND
    assert found.projected_saving == LARGE_BATCH_SAVING


def test_batch_lane_reports_zero_when_the_predicate_matches_nothing(book: PriceBook) -> None:
    found = detect_batch_lane(_repeats(book, 2), predicate=lambda record: False)

    assert found.status == "no_opportunity"
    assert found.projected_saving == Decimal(0)
    assert any("matched no request" in phrase for phrase in found.assumptions)


def test_batch_lane_assumes_no_discount(book: PriceBook) -> None:
    items = _priced(book, _record(provider="zenith", model="zenith-solo"))

    found = detect_batch_lane(items, predicate=lambda record: True)

    assert found.status == "ready"
    assert found.eligible_spend == SOLO_SPEND
    assert found.projected_saving == Decimal(0)


# --- semantic_dedup ---------------------------------------------------------


def test_semantic_dedup_always_needs_embeddings(book: PriceBook) -> None:
    found = detect_semantic_dedup(_repeats(book, 3))

    assert found.risk_class == QUALITY_AFFECTING
    assert found.status == "needs_embeddings"
    assert found.projected_saving is None
    assert found.confidence_interval is None
    assert found.eligible_requests == 0
    assert found.eligible_spend == Decimal(0)
    assert any(phrase.startswith("fix:") for phrase in found.required_changes)


# --- tier_routing -----------------------------------------------------------


def test_tier_routing_reports_addressable_spend_only(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0),
        _record(id="b", raw_index=1, timestamp=BASE + 1),
        _record(id="c", raw_index=2, timestamp=BASE + 2, model="acme-small"),
    )

    found = detect_tier_routing(items, book=book)

    assert found.risk_class == QUALITY_AFFECTING
    assert found.status == "needs_evaluation"
    assert found.projected_saving is None
    assert found.confidence_interval is None
    assert found.eligible_requests == 2
    assert found.eligible_spend == Decimal("0.026")
    assert any("acme/acme-large -> acme-small" in phrase for phrase in found.required_changes)


def test_tier_routing_reports_zero_for_the_cheapest_model(book: PriceBook) -> None:
    found = detect_tier_routing(_priced(book, _record(model="acme-small")), book=book)

    assert found.status == "no_opportunity"
    assert found.projected_saving == Decimal(0)


def test_tier_routing_reports_zero_for_a_single_model_provider(book: PriceBook) -> None:
    items = _priced(book, _record(provider="zenith", model="zenith-solo"))

    assert detect_tier_routing(items, book=book).status == "no_opportunity"


# --- sampling ---------------------------------------------------------------


def test_sampling_projects_every_redundant_sample(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, group_key="task-1"),
        _record(id="b", raw_index=1, timestamp=BASE + 1, group_key="task-1"),
        _record(id="c", raw_index=2, timestamp=BASE + 2, group_key="task-1"),
        _record(id="d", raw_index=3, timestamp=BASE + 3, group_key="task-2"),
    )

    found = detect_sampling(items)

    assert found.risk_class == QUALITY_AFFECTING
    assert found.status == "ready"
    assert found.eligible_requests == 2
    assert found.eligible_spend == Decimal("0.026")
    assert found.projected_saving == Decimal("0.026")
    assert any("1 group_key value(s)" in phrase for phrase in found.assumptions)


def test_sampling_returns_zero_on_a_single_sample_workload(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, group_key="task-1"),
        _record(id="b", raw_index=1, timestamp=BASE + 1, group_key="task-2"),
    )

    found = detect_sampling(items)

    assert found.status == "no_opportunity"
    assert found.eligible_requests == 0
    assert found.projected_saving == Decimal(0)
    assert found.confidence_interval == Range(low=Decimal(0), high=Decimal(0))


def test_sampling_ignores_ungrouped_requests(book: PriceBook) -> None:
    found = detect_sampling(_repeats(book, 3))

    assert found.status == "no_opportunity"
    assert found.projected_saving == Decimal(0)


# --- unpriced models --------------------------------------------------------


def test_an_unknown_pair_is_partitioned_and_never_raises(book: PriceBook) -> None:
    records = (
        _record(id="a", raw_index=0),
        _record(id="b", raw_index=1, timestamp=BASE + 1, model="ghost-model"),
    )

    items, unpriced = price_records(records, book)

    with pytest.raises(UnknownModelError):
        book.entry("acme", "ghost-model")
    assert len(items) == 1
    assert [record.model for record in unpriced] == ["ghost-model"]

    opportunities, overlaps = detect_opportunities(items, book=book)

    assert [item.lever for item in opportunities] == list(PRECEDENCE)
    assert overlaps == ()


def test_an_entirely_unpriced_log_still_reports_every_lever(book: PriceBook) -> None:
    items, unpriced = price_records((_record(model="ghost-model"),), book)

    assert items == ()
    assert len(unpriced) == 1

    opportunities, overlaps = detect_opportunities(items, book=book)

    assert [item.lever for item in opportunities] == list(PRECEDENCE)
    assert overlaps == ()
    for item in opportunities:
        assert item.eligible_requests == 0
        assert item.eligible_spend == Decimal(0)


# --- precedence and overlap -------------------------------------------------


def test_the_pipeline_sizes_a_lever_from_the_subset_it_assigned(book: PriceBook) -> None:
    items = _repeats(book, 3)

    opportunities, overlaps = detect_opportunities(items, book=book)
    dedup = _by_lever(opportunities)["exact_dedup"]

    assert dedup == detect_exact_dedup(items)
    assert dedup.eligible_requests == 2
    assert dedup.projected_saving == Decimal("0.026")
    assert overlaps == ()


def test_a_request_eligible_for_two_levers_is_counted_once(book: PriceBook) -> None:
    items = _repeats(
        book, 2, system_prefix_hash=PREFIX_ONE, system_prefix_chars=400, cached_prompt_tokens=400
    )

    opportunities, overlaps = detect_opportunities(items, book=book)
    levers = _by_lever(opportunities)

    assert levers["exact_dedup"].status == "ready"
    assert levers["exact_dedup"].eligible_requests == 1
    assert levers["exact_dedup"].eligible_spend == LARGE_CACHED_SPEND
    assert levers["exact_dedup"].projected_saving == LARGE_CACHED_SPEND
    assert levers["prefix_cache"].status == "realized"
    assert levers["prefix_cache"].eligible_requests == 1
    assert levers["prefix_cache"].eligible_spend == LARGE_CACHED_SPEND
    assert levers["prefix_cache"].projected_saving == LARGE_CACHED_SAVING
    assert overlaps == (
        Overlap(
            lever="prefix_cache",
            claimed_by="exact_dedup",
            requests=1,
            spend=LARGE_CACHED_SPEND,
        ),
    )
    assert overlaps[0].message().startswith("1 request(s) worth 0.0094")


def test_a_fully_displaced_lever_says_which_lever_took_its_traffic(book: PriceBook) -> None:
    items = _repeats(book, 2, system_prefix_hash=PREFIX_ONE, cached_prompt_tokens=400)

    opportunities, overlaps = detect_opportunities(
        items, book=book, batch_predicate=lambda record: True
    )
    batch = _by_lever(opportunities)["batch_lane"]

    assert batch.status == "no_opportunity"
    assert batch.projected_saving == Decimal(0)
    assert any("higher-precedence lever" in phrase for phrase in batch.assumptions)
    assert not any("matched no request" in phrase for phrase in batch.assumptions)
    assert [(item.lever, item.claimed_by, item.requests) for item in overlaps] == [
        ("prefix_cache", "exact_dedup", 1),
        ("batch_lane", "exact_dedup", 1),
        ("batch_lane", "prefix_cache", 1),
    ]


def test_precedence_leaves_a_lower_lever_its_own_traffic(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, group_key="task-1"),
        _record(id="b", raw_index=1, timestamp=BASE + 1, group_key="task-1"),
        _record(
            id="c",
            raw_index=2,
            timestamp=BASE + 2,
            messages_hash=HASH_TWO,
            group_key="task-2",
        ),
        _record(
            id="d",
            raw_index=3,
            timestamp=BASE + 3,
            messages_hash=HASH_TWO,
            group_key="task-2",
        ),
    )

    opportunities, overlaps = detect_opportunities(items, book=book)
    levers = _by_lever(opportunities)

    assert levers["exact_dedup"].eligible_requests == 2
    assert levers["exact_dedup"].projected_saving == Decimal("0.026")
    assert levers["sampling"].status == "no_opportunity"
    assert [(item.lever, item.claimed_by, item.requests) for item in overlaps] == [
        ("sampling", "exact_dedup", 2)
    ]


def test_the_window_reaches_the_pipeline(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0),
        _record(id="b", raw_index=1, timestamp=BASE + 600),
    )

    opportunities, _ = detect_opportunities(items, book=book, window_seconds=60)

    assert _by_lever(opportunities)["exact_dedup"].projected_saving == Decimal(0)


# --- contract invariants ----------------------------------------------------


def _mixed_log(book: PriceBook) -> tuple[Priced, ...]:
    return _priced(
        book,
        _record(id="a", raw_index=0, system_prefix_hash=PREFIX_ONE),
        _record(
            id="b",
            raw_index=1,
            timestamp=BASE + 1,
            messages_hash=HASH_TWO,
            system_prefix_hash=PREFIX_ONE,
        ),
        _record(id="c", raw_index=2, timestamp=BASE + 2, model="acme-small", group_key="task-1"),
        _record(id="d", raw_index=3, timestamp=BASE + 3, model="acme-small", group_key="task-1"),
    )


def test_no_figure_is_reported_without_evidence(book: PriceBook) -> None:
    opportunities, _ = detect_opportunities(_mixed_log(book), book=book)

    assert [item.lever for item in opportunities] == list(PRECEDENCE)
    for item in opportunities:
        assert item.status in STATUSES
        assert item.risk_class == LEVER_RISK[item.lever]
        if item.projected_saving is None:
            assert item.status in BLOCKED_STATUSES
            assert item.confidence_interval is None
            assert item.required_changes
        else:
            assert isinstance(item.projected_saving, Decimal)
            assert item.assumptions
            assert item.confidence_interval is not None
            assert item.confidence_interval.contains(item.projected_saving)


def test_every_blocked_status_is_reachable(book: PriceBook) -> None:
    opportunities, _ = detect_opportunities(_mixed_log(book), book=book)
    blocked = {item.status for item in opportunities if item.projected_saving is None}

    assert blocked == set(BLOCKED_STATUSES)


# --- workload description ---------------------------------------------------


def test_profile_workload_reports_coverage_and_unpriced_pairs(book: PriceBook) -> None:
    records = (
        _record(id="a", raw_index=0, system_prefix_hash=PREFIX_ONE, cached_prompt_tokens=400),
        _record(id="b", raw_index=1, timestamp=BASE + 1, system_prefix_hash=PREFIX_ONE),
        _record(id="c", raw_index=2, timestamp=BASE + 2, model="ghost-model"),
    )
    items, unpriced = price_records(records, book)

    profile = profile_workload(
        items, unpriced, IngestReport(parsed=3, skipped=1, reasons={"blank_line": 1})
    )

    assert profile.records == 3
    assert profile.priced_records == 2
    assert profile.unpriced_records == 1
    assert profile.unpriced_pairs == (("acme", "ghost-model"),)
    assert profile.coverage_percent == Decimal("66.7")
    assert profile.prompt_tokens == 3000
    assert profile.cached_prompt_tokens == 400
    assert profile.records_with_cached_counts == 1
    assert profile.distinct_message_hashes == 1
    assert profile.distinct_prefix_hashes == 1
    assert profile.duration_seconds == 2.0
    assert profile.to_dict()["skip_reasons"] == {"blank_line": 1}


def test_spend_by_model_orders_by_observed_spend(book: PriceBook) -> None:
    items = _priced(
        book,
        _record(id="a", raw_index=0, model="acme-small"),
        _record(id="b", raw_index=1, timestamp=BASE + 1),
        _record(id="c", raw_index=2, timestamp=BASE + 2, provider="zenith", model="zenith-solo"),
    )

    rows = spend_by_model(items)

    assert [(row.provider, row.model, row.spend) for row in rows] == [
        ("acme", "acme-large", LARGE_SPEND),
        ("zenith", "zenith-solo", SOLO_SPEND),
        ("acme", "acme-small", SMALL_SPEND),
    ]
    assert sum_money([row.spend for row in rows]) == Decimal("0.0167")


# --- result types -----------------------------------------------------------


def _opportunity(**overrides: Any) -> Opportunity:
    fields: dict[str, Any] = {
        "lever": "exact_dedup",
        "risk_class": IDENTICAL,
        "eligible_requests": 1,
        "eligible_spend": Decimal("0.013"),
        "projected_saving": Decimal("0.013"),
        "confidence_interval": Range(low=Decimal("0.013"), high=Decimal("0.013")),
        "assumptions": (PROJECTION_BASIS,),
        "required_changes": ("enable the cache",),
        "status": "ready",
    }
    fields.update(overrides)
    return Opportunity(**fields)


def test_a_projection_without_assumptions_is_rejected() -> None:
    with pytest.raises(ValueError, match="assumptions"):
        _opportunity(assumptions=())


def test_a_blocked_status_cannot_carry_a_figure() -> None:
    with pytest.raises(ValueError, match="carries a "):
        _opportunity(status="needs_token_counts")


def test_a_blocked_status_must_name_what_is_missing() -> None:
    with pytest.raises(ValueError, match="without naming what is missing"):
        _opportunity(
            status="needs_token_counts",
            projected_saving=None,
            confidence_interval=None,
            required_changes=(),
        )


def test_a_lever_cannot_save_more_than_it_addresses() -> None:
    with pytest.raises(ValueError, match="of eligible spend"):
        _opportunity(eligible_spend=Decimal("0.001"))


def test_float_money_is_rejected() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        Range(low=0.0, high=1.0)  # type: ignore[arg-type]


def test_an_audit_result_must_report_every_lever(book: PriceBook) -> None:
    opportunities, overlaps = detect_opportunities(_mixed_log(book), book=book)

    with pytest.raises(ValueError, match="exactly one opportunity per lever"):
        _audit_result(book, opportunities[:-1], overlaps)


def test_an_audit_result_ranks_identical_levers_first(book: PriceBook) -> None:
    items = _repeats(book, 3, group_key="task-1")
    opportunities, overlaps = detect_opportunities(items, book=book)

    result = _audit_result(book, opportunities, overlaps)

    assert [item.lever for item in result.ranked()] == ["exact_dedup"]
    assert [item.lever for item in result.ranked(include_quality_affecting=True)] == ["exact_dedup"]
    assert [item.lever for item in result.blocked] == [
        "batch_lane",
        "semantic_dedup",
        "tier_routing",
    ]
    assert result.opportunity("sampling").status == "no_opportunity"
    assert result.to_dict()["opportunities"][0]["lever"] == "exact_dedup"


def _audit_result(
    book: PriceBook,
    opportunities: tuple[Opportunity, ...],
    overlaps: tuple[Overlap, ...],
) -> AuditResult:
    return AuditResult(
        source="traffic.jsonl",
        source_sha256="0" * 64,
        format_id="generic-jsonl",
        currency="USD",
        observed_spend=Decimal("0.039"),
        spend_by_model=(),
        workload=profile_workload((), (), IngestReport(parsed=0, skipped=0, reasons={})),
        opportunities=opportunities,
        overlaps=overlaps,
        price_book=PriceBookProvenance(
            schema_version=1,
            path=str(PRICE_BOOK),
            currency="USD",
            entries=tuple(
                (entry.provider, entry.model, entry.effective_date.isoformat())
                for entry in book.entries
            ),
        ),
        window_seconds=None,
        reproduction_command="branchpilot audit traffic.jsonl",
        fixes=(),
    )
