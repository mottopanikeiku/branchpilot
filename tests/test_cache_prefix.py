"""Prefix-cache analysis: clustering, realized behavior, projections, and volatility.

The 2,000-character / 50-request acceptance logs are written into ``tmp_path`` from the
constants below rather than committed: the input is 100 KB of repeated prefix padding, and
generating it keeps the fixture readable while still driving the real ingest readers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from branchpilot.cache import (
    PLAN_STATUSES,
    PROVIDER_CACHE_DOCS,
    TOKEN_SOURCES,
    VOLATILE_MIN_REQUESTS,
    CachePlan,
    ModelPrefixUsage,
    PrefixAnalysis,
    PrefixAnalysisError,
    PrefixCluster,
    VolatilePrefix,
    analyze_prefixes,
    cache_write_multiplier,
    max_cache_breakpoints,
    minimum_cacheable_tokens,
    plan_cache,
)
from branchpilot.ingest import read_requests
from branchpilot.ingest.formats import RequestRecord
from branchpilot.pricing import SCHEMA_VERSION, PriceBook

PROVIDER = "anthropic"
MODEL = "claude-sonnet-5"
REQUESTS = 50
PREFIX_CHARS = 2000
PREFIX_TOKENS = 2000
PROMPT_TOKENS = 5000
FIRST_TIMESTAMP = 1_775_030_400
STEP_SECONDS = 60

# Exactly PREFIX_CHARS characters of stable system instructions.
STABLE_PREFIX = ("Sort the kumquat manifest for warehouse nine. " * 50)[:PREFIX_CHARS]

# claude-sonnet-5 is $2.00/M input and $0.20/M cached input in the packaged price book.
FRESH_COST = Decimal("0.004")  # 2000 tokens at $2.00/M
READ_SAVING = Decimal("0.0036")  # 2000 tokens at ($2.00 - $0.20)/M
WRITE_PREMIUM = Decimal("0.001")  # 0.25 x FRESH_COST for the 5-minute Anthropic write
HOURLY_PREMIUM = Decimal("0.004")  # 1.00 x FRESH_COST for the 1-hour write multiplier


def _digest(label: str) -> str:
    """A 32-hex prefix digest, built the way the ingest readers build one."""
    return hashlib.sha256(label.encode("utf-8")).digest()[:16].hex()


def _record(**overrides: Any) -> RequestRecord:
    fields: dict[str, Any] = {
        "id": "msg_0000",
        "timestamp": float(FIRST_TIMESTAMP),
        "model": MODEL,
        "provider": PROVIDER,
        "messages_hash": _digest("messages"),
        "system_prefix_hash": _digest("stable"),
        "prompt_tokens": PROMPT_TOKENS,
        "cached_prompt_tokens": None,
        "completion_tokens": 96,
        "latency_ms": 812.5,
        "status": "ok",
        "group_key": None,
        "raw_index": 0,
        "system_prefix_chars": PREFIX_CHARS,
    }
    fields.update(overrides)
    return RequestRecord(**fields)


def _stream(count: int, **overrides: Any) -> list[RequestRecord]:
    """``count`` records that differ only in identity, index, and timestamp."""
    return [
        _record(
            id=f"msg_{index:04d}",
            raw_index=index,
            timestamp=float(FIRST_TIMESTAMP + index * STEP_SECONDS),
            **overrides,
        )
        for index in range(count)
    ]


def _uuid_prefix(index: int) -> str:
    """A prefix of constant length whose leading uuid changes every request."""
    injected = f"{index:08x}-1111-4222-8333-{index:012x}"
    assert len(injected) == 36
    return injected + STABLE_PREFIX[36:]


def _write_log(path: Path, systems: Iterable[str], *, model: str = MODEL) -> Path:
    lines = []
    for index, system in enumerate(systems):
        lines.append(
            json.dumps(
                {
                    "timestamp": FIRST_TIMESTAMP + index * STEP_SECONDS,
                    "request": {
                        "model": model,
                        "system": system,
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "manifest"}]}
                        ],
                    },
                    "response": {
                        "id": f"msg_{index:04d}",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [{"type": "text", "text": "sorted"}],
                        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": 96},
                    },
                    "latency_ms": 812.5,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _analyze_log(path: Path) -> PrefixAnalysis:
    return analyze_prefixes(read_requests(path, format="anthropic-jsonl"))


def _plan(analysis: PrefixAnalysis, book: PriceBook, **kwargs: Any) -> CachePlan:
    plans = analysis.plans(book, **kwargs)
    assert len(plans) == 1
    return plans[0]


def _slice(**overrides: Any) -> ModelPrefixUsage:
    fields: dict[str, Any] = {
        "provider": PROVIDER,
        "model": MODEL,
        "requests": 2,
        "billed_requests": 2,
        "prompt_tokens": 2 * PROMPT_TOKENS,
        "measured_requests": 0,
        "measured_prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "hit_requests": 0,
        "min_positive_cached_tokens": None,
    }
    fields.update(overrides)
    return ModelPrefixUsage(**fields)


@pytest.fixture(scope="module")
def book() -> PriceBook:
    return PriceBook.load()


@pytest.fixture
def stable_log(tmp_path: Path) -> Path:
    return _write_log(tmp_path / "stable.jsonl", [STABLE_PREFIX] * REQUESTS)


@pytest.fixture
def churning_log(tmp_path: Path) -> Path:
    return _write_log(tmp_path / "churning.jsonl", [_uuid_prefix(i) for i in range(REQUESTS)])


# --- clustering -------------------------------------------------------------


def test_stable_prefix_log_yields_one_cluster(stable_log: Path) -> None:
    assert len(STABLE_PREFIX) == PREFIX_CHARS

    stream = read_requests(stable_log, format="anthropic-jsonl")
    analysis = analyze_prefixes(stream)

    assert stream.report().parsed == REQUESTS
    assert analysis.records == REQUESTS
    assert len(analysis.clusters) == 1
    cluster = analysis.clusters[0]
    assert cluster.prefix_chars == PREFIX_CHARS
    assert cluster.requests == REQUESTS
    assert cluster.billed_requests == REQUESTS
    assert cluster.distinct_models == 1
    assert cluster.models == ((PROVIDER, MODEL),)
    assert cluster.total_prefix_chars == PREFIX_CHARS * REQUESTS
    assert cluster.repeated_prefix_chars == PREFIX_CHARS * (REQUESTS - 1)
    assert cluster.prompt_tokens == PROMPT_TOKENS * REQUESTS
    assert cluster.first_seen == FIRST_TIMESTAMP
    assert cluster.last_seen == FIRST_TIMESTAMP + (REQUESTS - 1) * STEP_SECONDS
    assert cluster.has_prefix is True
    assert analysis.cacheable_clusters == (cluster,)
    assert analysis.cluster(cluster.system_prefix_hash) is cluster


def test_cluster_reports_no_realized_behavior_without_cached_counts(stable_log: Path) -> None:
    cluster = _analyze_log(stable_log).clusters[0]

    assert cluster.measured_requests == 0
    assert cluster.measured_prompt_tokens == 0
    assert cluster.cached_prompt_tokens == 0
    assert cluster.hit_requests == 0
    assert cluster.realized_cached_share is None
    assert cluster.realized_hit_rate is None
    assert cluster.usage[0].measured_prefix_tokens is None


def test_realized_cache_behavior_comes_from_the_log() -> None:
    # 49 of 50 requests read 1,900 prompt tokens from cache; the first was a cold write.
    records = [_record(id="msg_0000", raw_index=0, cached_prompt_tokens=0)]
    records += _stream(REQUESTS - 1, cached_prompt_tokens=1900)
    analysis = analyze_prefixes(records)

    cluster = analysis.cluster(_digest("stable"))
    usage = cluster.slice_for(PROVIDER, MODEL)
    assert usage.measured_requests == REQUESTS
    assert usage.measured_prompt_tokens == PROMPT_TOKENS * REQUESTS
    assert usage.cached_prompt_tokens == 1900 * (REQUESTS - 1)
    assert usage.hit_requests == REQUESTS - 1
    assert usage.min_positive_cached_tokens == 1900
    assert usage.measured_prefix_tokens == 1900
    assert usage.unrealized_hits == 0
    # 93,100 of 250,000 measured prompt tokens were served from cache.
    assert cluster.realized_cached_share == Decimal("0.372400")
    assert cluster.realized_hit_rate == Decimal("0.980000")


def test_multi_model_cluster_slices_per_model(book: PriceBook) -> None:
    records = _stream(4) + [
        _record(id=f"gpt_{i}", raw_index=10 + i, provider="openai", model="gpt-5") for i in range(4)
    ]
    analysis = analyze_prefixes(records)

    cluster = analysis.clusters[0]
    assert cluster.requests == 8
    assert cluster.distinct_models == 2
    assert cluster.models == (("anthropic", MODEL), ("openai", "gpt-5"))
    assert [slice_.requests for slice_ in cluster.usage] == [4, 4]

    # 2,500 tokens clears both minimums: 1,024 on claude-sonnet-5 and 2,048 on gpt-5.
    plans = analysis.plans(book, prefix_tokens={cluster.system_prefix_hash: 2500})
    assert [(plan.provider, plan.model, plan.status) for plan in plans] == [
        ("anthropic", MODEL, "ok"),
        ("openai", "gpt-5", "ok"),
    ]
    # 3 uncached repeats at 2,500 tokens: $1.80/M saved less a $0.00125 write premium.
    assert plans[0].projected_saving == Decimal("0.01225")
    # OpenAI caches implicitly, so there is no marker to place and no write premium.
    assert plans[1].breakpoints == 0
    assert plans[1].write_premium == Decimal(0)
    assert plans[1].break_even_hits == 0
    assert plans[1].min_cacheable_tokens == 2048
    # 3 uncached repeats at 2,500 tokens: $1.125/M saved with no premium to recover.
    assert plans[1].projected_saving == Decimal("0.0084375")


def test_traffic_without_a_system_prefix_is_never_cacheable(book: PriceBook) -> None:
    analysis = analyze_prefixes(_stream(5, system_prefix_chars=0, system_prefix_hash=_digest("")))

    cluster = analysis.clusters[0]
    assert cluster.has_prefix is False
    assert cluster.total_prefix_chars == 0
    assert cluster.repeated_prefix_chars == 0
    assert analysis.cacheable_clusters == ()
    plan = _plan(analysis, book)
    assert plan.status == "no_prefix"
    assert plan.projected_saving is None
    assert plan.fix is not None


# --- projections ------------------------------------------------------------


def test_break_even_matches_hand_computation(stable_log: Path, book: PriceBook) -> None:
    analysis = _analyze_log(stable_log)
    prefix_hash = analysis.clusters[0].system_prefix_hash

    plan = _plan(analysis, book, prefix_tokens={prefix_hash: PREFIX_TOKENS})

    assert plan.status == "ok"
    assert plan.actionable is True
    assert plan.fix is None
    assert plan.token_source == "supplied"
    assert plan.prefix_tokens == PREFIX_TOKENS
    assert plan.requests == REQUESTS
    assert plan.billed_requests == REQUESTS
    assert plan.unrealized_hits == REQUESTS - 1
    assert plan.breakpoints == 1
    assert plan.min_cacheable_tokens == 1024
    assert plan.read_saving == READ_SAVING
    assert plan.write_premium == WRITE_PREMIUM
    # ceil($0.001 write premium / $0.0036 per read) == 1
    assert plan.break_even_hits == 1
    assert plan.projected_saving == READ_SAVING * (REQUESTS - 1) - WRITE_PREMIUM
    assert plan.projected_saving == Decimal("0.1754")


def test_break_even_scales_with_the_write_multiplier(stable_log: Path, book: PriceBook) -> None:
    analysis = _analyze_log(stable_log)
    prefix_hash = analysis.clusters[0].system_prefix_hash

    plan = _plan(
        analysis,
        book,
        prefix_tokens={prefix_hash: PREFIX_TOKENS},
        write_multipliers={PROVIDER: Decimal("2")},
    )

    assert plan.write_premium == HOURLY_PREMIUM
    # ceil($0.004 write premium / $0.0036 per read) == 2
    assert plan.break_even_hits == 2
    assert plan.projected_saving == Decimal("0.1724")


def test_missing_token_counts_refuses_to_project(stable_log: Path, book: PriceBook) -> None:
    plan = _plan(_analyze_log(stable_log), book)

    assert plan.status == "needs_token_counts"
    assert plan.projected_saving is None
    assert plan.read_saving is None
    assert plan.write_premium is None
    assert plan.break_even_hits is None
    assert plan.prefix_tokens is None
    assert plan.token_source == "unavailable"
    assert plan.actionable is False
    assert plan.fix is not None
    assert "prompt_tokens" in plan.fix


def test_measured_cache_reads_supply_the_token_count(book: PriceBook) -> None:
    records = _stream(REQUESTS - 1) + [
        _record(id="msg_last", raw_index=99, cached_prompt_tokens=PREFIX_TOKENS)
    ]
    analysis = analyze_prefixes(records)

    plan = _plan(analysis, book)

    assert plan.status == "ok"
    assert plan.token_source == "measured"
    assert plan.prefix_tokens == PREFIX_TOKENS
    assert plan.read_saving == READ_SAVING
    # One request already hit, so the write premium is already paid.
    assert plan.write_premium == WRITE_PREMIUM
    assert plan.unrealized_hits == REQUESTS - 2
    assert plan.projected_saving == READ_SAVING * (REQUESTS - 2)


def test_fully_cached_traffic_leaves_nothing_on_the_table(book: PriceBook) -> None:
    records = [_record(id="msg_0000", raw_index=0, cached_prompt_tokens=0)]
    records += _stream(REQUESTS - 1, cached_prompt_tokens=PREFIX_TOKENS)

    plan = _plan(analyze_prefixes(records), book)

    assert plan.status == "already_cached"
    assert plan.unrealized_hits == 0
    assert plan.projected_saving == Decimal(0)
    assert plan.actionable is False
    assert plan.fix is not None


def test_unbilled_requests_never_project_a_saving(book: PriceBook) -> None:
    refused = analyze_prefixes(_stream(REQUESTS, status="rate_limited", prompt_tokens=0))
    cluster = refused.clusters[0]
    assert cluster.requests == REQUESTS
    assert cluster.billed_requests == 0

    plan = _plan(refused, book, prefix_tokens={cluster.system_prefix_hash: PREFIX_TOKENS})
    assert plan.status == "no_repeats"
    assert plan.unrealized_hits == 0
    assert plan.projected_saving is None
    assert plan.fix is not None

    unbilled = [_record(id=f"nope_{i}", raw_index=50 + i, prompt_tokens=0) for i in range(40)]
    mixed = analyze_prefixes(_stream(10) + unbilled)
    hash_ = mixed.clusters[0].system_prefix_hash
    assert mixed.clusters[0].billed_requests == 10
    partial = _plan(mixed, book, prefix_tokens={hash_: PREFIX_TOKENS})
    assert partial.requests == 50
    assert partial.billed_requests == 10
    assert partial.unrealized_hits == 9
    assert partial.projected_saving == READ_SAVING * 9 - WRITE_PREMIUM


def test_unpriced_model_reports_instead_of_raising(book: PriceBook) -> None:
    analysis = analyze_prefixes(_stream(5, model="claude-mystery-9"))
    hash_ = analysis.clusters[0].system_prefix_hash

    plan = _plan(analysis, book, prefix_tokens={hash_: PREFIX_TOKENS})

    assert plan.status == "unpriced_model"
    assert plan.projected_saving is None
    assert plan.read_saving is None
    assert plan.prefix_tokens == PREFIX_TOKENS
    assert plan.token_source == "supplied"
    # The provider fallback minimum still applies to an unnamed model.
    assert plan.min_cacheable_tokens == 4096
    assert plan.fix is not None
    assert "--price-book" in plan.fix


def test_prefix_below_the_provider_minimum_is_not_cacheable(book: PriceBook) -> None:
    analysis = analyze_prefixes(_stream(5, model="claude-opus-4-6"))
    hash_ = analysis.clusters[0].system_prefix_hash

    plan = _plan(analysis, book, prefix_tokens={hash_: 1000})

    assert plan.status == "below_provider_minimum"
    assert plan.min_cacheable_tokens == 4096
    assert plan.projected_saving == Decimal(0)
    assert plan.read_saving is not None
    assert plan.fix is not None
    assert PROVIDER_CACHE_DOCS[PROVIDER] in plan.fix


def test_too_few_repeats_do_not_recover_the_write_premium(book: PriceBook) -> None:
    analysis = analyze_prefixes(_stream(2))
    hash_ = analysis.clusters[0].system_prefix_hash

    plan = _plan(
        analysis,
        book,
        prefix_tokens={hash_: PREFIX_TOKENS},
        write_multipliers={PROVIDER: Decimal("5")},
    )

    assert plan.status == "below_break_even"
    assert plan.unrealized_hits == 1
    # ceil($0.016 write premium / $0.0036 per read) == 5
    assert plan.break_even_hits == 5
    assert plan.projected_saving == Decimal(0)
    assert plan.fix is not None
    assert "at least 5 times" in plan.fix


def test_a_cache_read_that_saves_nothing_has_no_break_even(tmp_path: Path) -> None:
    flat = tmp_path / "prices.json"
    flat.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "entries": [
                    {
                        "provider": PROVIDER,
                        "model": MODEL,
                        "input": "2",
                        "output": "10",
                        "cached_input": "2",
                        "batch_input": "1",
                        "batch_output": "5",
                        "currency": "USD",
                        "effective_date": "2026-08-27",
                        "source_url": "https://platform.claude.com/docs/en/about-claude/pricing",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    analysis = analyze_prefixes(_stream(5))
    hash_ = analysis.clusters[0].system_prefix_hash

    plan = _plan(analysis, PriceBook.load(flat), prefix_tokens={hash_: PREFIX_TOKENS})

    assert plan.status == "below_break_even"
    assert plan.read_saving == Decimal(0)
    assert plan.break_even_hits is None
    assert plan.projected_saving == Decimal(0)
    assert plan.fix is not None
    assert "None" not in plan.message()


def test_every_refusal_carries_exactly_one_fix_clause(stable_log: Path, book: PriceBook) -> None:
    analysis = _analyze_log(stable_log)

    for plan in analysis.plans(book):
        assert plan.status in PLAN_STATUSES
        assert plan.fix is not None
        assert plan.message().count("fix:") == 1
        assert plan.status in plan.message()


# --- volatility -------------------------------------------------------------


def test_uuid_in_the_system_prefix_is_flagged_volatile(churning_log: Path) -> None:
    analysis = _analyze_log(churning_log)

    assert analysis.records == REQUESTS
    assert len(analysis.clusters) == REQUESTS
    assert len(analysis.volatile) == 1
    signal = analysis.volatile[0]
    assert (signal.provider, signal.model) == (PROVIDER, MODEL)
    assert signal.prefix_chars == PREFIX_CHARS
    assert signal.requests == REQUESTS
    assert signal.distinct_hashes == REQUESTS
    assert signal.churn_rate == Decimal("1.000000")
    assert signal.first_seen == FIRST_TIMESTAMP
    assert signal.last_seen == FIRST_TIMESTAMP + (REQUESTS - 1) * STEP_SECONDS
    message = signal.message()
    assert "uuid" in message
    assert "fix:" in message
    assert PROVIDER_CACHE_DOCS[PROVIDER] in message


def test_a_stable_prefix_is_not_flagged_volatile(stable_log: Path) -> None:
    analysis = _analyze_log(stable_log)

    assert analysis.volatile == ()
    lengths = analyze_prefixes(_stream(REQUESTS))
    assert lengths.volatile == ()


def test_one_prefix_revision_is_not_churn() -> None:
    records = _stream(30) + [
        _record(id=f"v2_{i}", raw_index=100 + i, system_prefix_hash=_digest("stable-v2"))
        for i in range(20)
    ]

    analysis = analyze_prefixes(records)

    assert len(analysis.clusters) == 2
    assert analysis.volatile == ()


def test_churn_needs_enough_requests_to_mean_anything() -> None:
    def churn(count: int) -> tuple[VolatilePrefix, ...]:
        records = [
            _record(id=f"m{i}", raw_index=i, system_prefix_hash=_digest(f"uuid-{i}"))
            for i in range(count)
        ]
        return analyze_prefixes(records).volatile

    assert churn(VOLATILE_MIN_REQUESTS - 1) == ()
    flagged = churn(VOLATILE_MIN_REQUESTS)
    assert len(flagged) == 1
    assert flagged[0].churn_rate == Decimal("1.000000")


def test_churn_is_reported_per_model() -> None:
    records = [
        _record(id=f"a{i}", raw_index=i, system_prefix_hash=_digest(f"uuid-{i}")) for i in range(6)
    ]
    records += [_record(id=f"b{i}", raw_index=10 + i, model="claude-haiku-4-5") for i in range(6)]

    analysis = analyze_prefixes(records)

    assert [(v.model, v.requests) for v in analysis.volatile] == [(MODEL, 6)]


def test_a_prefix_with_no_characters_never_churns() -> None:
    records = [
        _record(
            id=f"m{i}",
            raw_index=i,
            system_prefix_chars=0,
            system_prefix_hash=_digest(f"uuid-{i}"),
        )
        for i in range(REQUESTS)
    ]

    assert analyze_prefixes(records).volatile == ()


def test_churn_threshold_is_configurable() -> None:
    # Ten requests over five prefixes: churn is 4/9, under the default threshold.
    records = [
        _record(id=f"m{i}", raw_index=i, system_prefix_hash=_digest(f"uuid-{i // 2}"))
        for i in range(10)
    ]

    assert analyze_prefixes(records).volatile == ()
    lenient = analyze_prefixes(records, churn_threshold=Decimal("0.4"))
    assert len(lenient.volatile) == 1
    assert lenient.volatile[0].churn_rate == Decimal("0.444444")


# --- determinism ------------------------------------------------------------


def _permutations(records: Sequence[RequestRecord]) -> list[list[RequestRecord]]:
    reversed_ = list(reversed(records))
    interleaved = [*records[1::2], *records[0::2]]
    rotated = [*records[7:], *records[:7]]
    return [reversed_, interleaved, rotated]


def test_permuting_input_order_changes_nothing(book: PriceBook) -> None:
    records = _stream(20, cached_prompt_tokens=PREFIX_TOKENS)
    records += [
        _record(id=f"u{i}", raw_index=200 + i, system_prefix_hash=_digest(f"uuid-{i}"))
        for i in range(10)
    ]
    records += [
        _record(id=f"o{i}", raw_index=300 + i, provider="openai", model="gpt-5") for i in range(6)
    ]
    baseline = analyze_prefixes(records)
    tokens = {cluster.system_prefix_hash: PREFIX_TOKENS for cluster in baseline.clusters}
    baseline_plans = baseline.plans(book, prefix_tokens=tokens)

    for permutation in _permutations(records):
        analysis = analyze_prefixes(permutation)
        assert analysis == baseline
        assert analysis.plans(book, prefix_tokens=tokens) == baseline_plans


def test_clusters_are_ordered_by_weight_then_hash() -> None:
    records = _stream(3, system_prefix_hash=_digest("small"))
    records += [
        _record(id=f"big{i}", raw_index=50 + i, system_prefix_hash=_digest("big")) for i in range(9)
    ]

    analysis = analyze_prefixes(records)

    assert [cluster.requests for cluster in analysis.clusters] == [9, 3]


def test_an_empty_stream_analyzes_to_nothing() -> None:
    analysis = analyze_prefixes([])

    assert analysis == PrefixAnalysis(records=0, clusters=(), volatile=())
    assert analysis.cacheable_clusters == ()


# --- provider facts ---------------------------------------------------------


def test_provider_minimums_prefer_an_exact_model_match() -> None:
    assert minimum_cacheable_tokens(PROVIDER, MODEL) == 1024
    assert minimum_cacheable_tokens(PROVIDER, "claude-opus-5") == 512
    assert minimum_cacheable_tokens("openai", "gpt-5") == 2048
    assert minimum_cacheable_tokens("gemini", "gemini-2.5-pro") == 2048
    # An unnamed model falls back to the provider's largest documented minimum.
    assert minimum_cacheable_tokens(PROVIDER, "claude-unreleased") == 4096
    # An undocumented provider is never guessed at.
    assert minimum_cacheable_tokens("mistral", "mistral-large") is None


def test_cache_write_premiums_and_breakpoints_are_provider_facts() -> None:
    assert cache_write_multiplier(PROVIDER) == Decimal("1.25")
    assert cache_write_multiplier("openai") == Decimal(1)
    assert cache_write_multiplier("gemini") == Decimal(1)
    assert cache_write_multiplier("mistral") == Decimal(1)
    assert max_cache_breakpoints(PROVIDER) == 4
    assert max_cache_breakpoints("openai") == 0
    assert max_cache_breakpoints("gemini") == 0
    assert max_cache_breakpoints("mistral") == 0


def test_the_package_exports_the_analyzer_surface() -> None:
    import branchpilot.cache as package
    from branchpilot.cache import prefix

    assert package.__all__ == prefix.__all__
    for name in package.__all__:
        assert getattr(package, name) is getattr(prefix, name)
    assert {"analyze_prefixes", "PrefixCluster", "CachePlan"} <= set(package.__all__)
    assert sorted(TOKEN_SOURCES) == ["measured", "supplied", "unavailable"]


# --- guards -----------------------------------------------------------------


def test_analyze_prefixes_rejects_unusable_arguments() -> None:
    with pytest.raises(PrefixAnalysisError, match="min_requests must be at least 2"):
        analyze_prefixes([], min_requests=1)
    with pytest.raises(PrefixAnalysisError, match="min_prefix_chars must be at least 1"):
        analyze_prefixes([], min_prefix_chars=0)
    with pytest.raises(PrefixAnalysisError, match="churn_threshold must be between 0 and 1"):
        analyze_prefixes([], churn_threshold=Decimal("2"))
    with pytest.raises(PrefixAnalysisError, match="finite decimal.Decimal"):
        analyze_prefixes([], churn_threshold=0.5)  # type: ignore[arg-type]
    with pytest.raises(PrefixAnalysisError, match="takes RequestRecord values, not dict"):
        analyze_prefixes([{"id": "msg_0000"}])  # type: ignore[list-item]


def test_plan_cache_rejects_mismatched_inputs(book: PriceBook) -> None:
    cluster = analyze_prefixes(_stream(3)).clusters[0]
    usage = cluster.usage[0]

    with pytest.raises(PrefixAnalysisError, match="needs a PrefixCluster"):
        plan_cache(usage, usage, book)  # type: ignore[arg-type]
    with pytest.raises(PrefixAnalysisError, match="needs a PriceBook"):
        plan_cache(cluster, usage, "prices.json")  # type: ignore[arg-type]
    with pytest.raises(PrefixAnalysisError, match="does not belong to cluster"):
        plan_cache(cluster, _slice(), book)
    with pytest.raises(PrefixAnalysisError, match="prefix_tokens cannot be zero"):
        plan_cache(cluster, usage, book, prefix_tokens=0)
    with pytest.raises(PrefixAnalysisError, match="write_multiplier .* is below 1"):
        plan_cache(cluster, usage, book, write_multiplier=Decimal("0.5"))


def test_plans_rejects_unusable_overrides(book: PriceBook) -> None:
    analysis = analyze_prefixes(_stream(3))

    with pytest.raises(PrefixAnalysisError, match="needs a PriceBook"):
        analysis.plans("prices.json")  # type: ignore[arg-type]
    with pytest.raises(PrefixAnalysisError, match="prefix_tokens must be a mapping"):
        analysis.plans(book, prefix_tokens=[2000])  # type: ignore[arg-type]
    with pytest.raises(PrefixAnalysisError, match="cannot be zero"):
        analysis.plans(book, prefix_tokens={_digest("stable"): 0})
    with pytest.raises(PrefixAnalysisError, match="write_multipliers must be a mapping"):
        analysis.plans(book, write_multipliers=[Decimal(2)])  # type: ignore[arg-type]
    with pytest.raises(PrefixAnalysisError, match="never costs less than"):
        analysis.plans(book, write_multipliers={PROVIDER: Decimal("0.9")})


def test_usage_slices_refuse_impossible_counts() -> None:
    with pytest.raises(PrefixAnalysisError, match="billed_requests .* exceeds requests"):
        _slice(requests=2, billed_requests=3)
    with pytest.raises(PrefixAnalysisError, match="hit_requests .* exceeds billed_requests"):
        _slice(billed_requests=1, hit_requests=2, measured_requests=2)
    with pytest.raises(PrefixAnalysisError, match="measured_requests .* exceeds requests"):
        _slice(measured_requests=3)
    with pytest.raises(PrefixAnalysisError, match="requests must be at least 1"):
        _slice(requests=0, billed_requests=0, prompt_tokens=0)
    with pytest.raises(PrefixAnalysisError, match="min_positive_cached_tokens cannot be zero"):
        _slice(min_positive_cached_tokens=0)
    with pytest.raises(PrefixAnalysisError, match="cached_prompt_tokens .* exceeds measured"):
        _slice(measured_requests=1, measured_prompt_tokens=10, cached_prompt_tokens=11)


def test_clusters_refuse_inconsistent_slices() -> None:
    first = _slice(provider="openai", model="gpt-5")
    second = _slice()
    with pytest.raises(PrefixAnalysisError, match="must be sorted"):
        PrefixCluster(
            system_prefix_hash=_digest("stable"),
            prefix_chars=PREFIX_CHARS,
            requests=4,
            usage=(first, second),
            first_seen=1.0,
            last_seen=2.0,
        )
    with pytest.raises(PrefixAnalysisError, match="sum to 4 requests"):
        PrefixCluster(
            system_prefix_hash=_digest("stable"),
            prefix_chars=PREFIX_CHARS,
            requests=9,
            usage=(second, first),
            first_seen=1.0,
            last_seen=2.0,
        )
    with pytest.raises(PrefixAnalysisError, match="last_seen precedes first_seen"):
        PrefixCluster(
            system_prefix_hash=_digest("stable"),
            prefix_chars=PREFIX_CHARS,
            requests=2,
            usage=(second,),
            first_seen=9.0,
            last_seen=2.0,
        )


def test_cluster_lookups_name_what_is_available() -> None:
    analysis = analyze_prefixes(_stream(3))
    cluster = analysis.clusters[0]

    with pytest.raises(PrefixAnalysisError, match="no cluster for prefix hash"):
        analysis.cluster(_digest("absent"))
    with pytest.raises(PrefixAnalysisError, match="has no slice for openai/gpt-5"):
        cluster.slice_for("openai", "gpt-5")


def _plan_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "system_prefix_hash": _digest("stable"),
        "provider": PROVIDER,
        "model": MODEL,
        "status": "ok",
        "requests": 2,
        "billed_requests": 2,
        "unrealized_hits": 1,
        "breakpoints": 1,
        "min_cacheable_tokens": 1024,
        "prefix_tokens": PREFIX_TOKENS,
        "token_source": "supplied",
        "write_premium": WRITE_PREMIUM,
        "read_saving": READ_SAVING,
        "projected_saving": Decimal("0.0026"),
        "break_even_hits": 1,
        "fix": None,
    }
    fields.update(overrides)
    return fields


def test_cache_plan_refuses_self_contradiction() -> None:
    assert CachePlan(**_plan_fields()).actionable is True

    with pytest.raises(PrefixAnalysisError, match="is not a known status"):
        CachePlan(**_plan_fields(status="maybe", fix="x; fix: y"))
    with pytest.raises(PrefixAnalysisError, match="is not a known source"):
        CachePlan(**_plan_fields(token_source="guessed"))
    with pytest.raises(PrefixAnalysisError, match="prefix_tokens and token_source disagree"):
        CachePlan(**_plan_fields(prefix_tokens=None))
    with pytest.raises(PrefixAnalysisError, match="but a saving was projected"):
        CachePlan(
            **_plan_fields(
                status="needs_token_counts",
                prefix_tokens=None,
                token_source="unavailable",
                fix="unknown; fix: re-export the log",
            )
        )
    with pytest.raises(PrefixAnalysisError, match="and fix disagree"):
        CachePlan(**_plan_fields(fix="stray; fix: nothing"))
    with pytest.raises(PrefixAnalysisError, match="must name the operator's next step"):
        CachePlan(**_plan_fields(status="no_repeats", fix="sent once"))
    with pytest.raises(PrefixAnalysisError, match="must be a decimal.Decimal"):
        CachePlan(**_plan_fields(projected_saving=0.0026))
    with pytest.raises(PrefixAnalysisError, match="billed_requests .* exceeds requests"):
        CachePlan(**_plan_fields(billed_requests=3))


def test_analysis_refuses_to_lose_or_duplicate_records() -> None:
    cluster = analyze_prefixes(_stream(3)).clusters[0]

    with pytest.raises(PrefixAnalysisError, match="clusters hold 3 requests"):
        PrefixAnalysis(records=4, clusters=(cluster,), volatile=())
    with pytest.raises(PrefixAnalysisError, match="two clusters for one system_prefix_hash"):
        PrefixAnalysis(records=6, clusters=(cluster, cluster), volatile=())


def test_volatile_prefix_refuses_impossible_churn() -> None:
    with pytest.raises(PrefixAnalysisError, match="needs at least 2 requests"):
        VolatilePrefix(
            provider=PROVIDER,
            model=MODEL,
            prefix_chars=PREFIX_CHARS,
            requests=1,
            distinct_hashes=1,
            churn_rate=Decimal(0),
            first_seen=1.0,
            last_seen=1.0,
        )
    with pytest.raises(PrefixAnalysisError, match="must be between 1 and requests"):
        VolatilePrefix(
            provider=PROVIDER,
            model=MODEL,
            prefix_chars=PREFIX_CHARS,
            requests=2,
            distinct_hashes=3,
            churn_rate=Decimal(1),
            first_seen=1.0,
            last_seen=1.0,
        )
