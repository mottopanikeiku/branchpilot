"""The six opportunity detectors, and the precedence pipeline that keeps them additive.

Each detector answers one question about one log and refuses to answer any other. The
shared discipline is that a figure is either arithmetic over tokens the log actually
recorded, or it is ``None`` with a status naming the missing input. Nothing here estimates,
extrapolates, or applies an industry rule of thumb.

What each detector can and cannot know
--------------------------------------
``exact_dedup``
    Fully determined by the log. Identical ``messages_hash`` inside the window means a
    cache would have returned the recorded response, so the saving is the repeat's whole
    billed cost.
``prefix_cache``
    Clusters by ``system_prefix_hash``. :class:`~branchpilot.ingest.formats.RequestRecord`
    carries no prefix *token* count, so the size of an unrealized prefix-cache opportunity
    is not derivable -- inferring it from ``prompt_tokens`` would be inventing the number.
    Where ``cached_prompt_tokens`` is present the detector instead reports what caching is
    *already* saving, measured from the log. Where it is absent the answer is
    ``needs_token_counts``.
``batch_lane``
    Eligibility is a property of the operator's application (is this request allowed to
    take 24 hours?), not of the log. Without a predicate the answer is
    ``needs_eligibility_rule``. With one, the saving comes from the price book's
    ``batch_input``/``batch_output`` rates -- never a hardcoded discount.
``semantic_dedup``
    Needs an embedding backend to name a single eligible request. Without one it reports
    ``needs_embeddings`` and no figure.
``tier_routing``
    Can identify spend that a cheaper same-provider model could address, but whether the
    cheaper model is good enough is an evaluation question. Reports addressable spend with
    ``needs_evaluation``.
``sampling``
    Multi-sample groups named by the log's ``group_key``. The ceiling -- collapsing every
    group to its first sample -- is exact arithmetic, so it is reported as such, with the
    quality caveat carried by its ``QUALITY_AFFECTING`` class. Single-sample workloads
    return zero rather than erroring.

Precedence
----------
``exact_dedup > prefix_cache > batch_lane > semantic_dedup > tier_routing > sampling``.
Only levers that can project a saving claim requests, because precedence exists to stop
the same dollar being counted twice in a total. ``semantic_dedup`` and ``tier_routing``
never project, so they neither claim nor are claimed; their figures are an addressable
surface and are explicitly not additive with anything. Every displaced request is reported
as an :class:`~branchpilot.audit.result.Overlap`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from branchpilot.audit.result import ModelSpend, Opportunity, Overlap, Range, WorkloadProfile
from branchpilot.audit.risk import LEVER_RISK, PRECEDENCE
from branchpilot.ingest.formats import EMPTY_TEXT_HASH, IngestReport, RequestRecord
from branchpilot.pricing import PriceBook, PriceEntry, UnknownModelError

__all__ = [
    "BOUND_RULE",
    "CLAIMING_LEVERS",
    "PROJECTION_BASIS",
    "Priced",
    "detect_batch_lane",
    "detect_exact_dedup",
    "detect_opportunities",
    "detect_prefix_cache",
    "detect_sampling",
    "detect_semantic_dedup",
    "detect_tier_routing",
    "price_records",
    "profile_workload",
    "spend_by_model",
]

PROJECTION_BASIS = "projection from observed tokens times configured prices"
"""The label every projected figure carries. Not a measurement of realized savings."""

MEASURED_BASIS = "measured from the log"
"""The label the one realized figure carries."""

BOUND_RULE = (
    "the low bound counts only requests the log records with status 'ok'; the high bound adds "
    "requests logged with any other status, because a provider may or may not have billed a "
    "call that failed"
)

CLAIMING_LEVERS = ("exact_dedup", "prefix_cache", "batch_lane", "sampling")
"""Levers that can project a saving, and therefore participate in overlap precedence."""

_DISPLACED_NOTE = (
    "every request this lever was eligible for is counted under a higher-precedence lever, so "
    "counting it here as well would count the same dollar twice; see the reported overlaps"
)

_OK = "ok"


@dataclass(frozen=True, slots=True)
class Priced:
    """One record with every lane's exact cost precomputed once."""

    record: RequestRecord
    entry: PriceEntry
    spend: Decimal
    """What the log says this request cost, cache reads included."""
    uncached_spend: Decimal
    """What the same tokens would have cost with no cache read at all."""
    batch_spend: Decimal
    """What the same tokens would have cost in the provider's batch lane."""

    @property
    def ok(self) -> bool:
        return self.record.status == _OK

    @property
    def realized_cache_saving(self) -> Decimal:
        """What the recorded cache reads already saved against standard rates."""
        return self.uncached_spend - self.spend

    @property
    def batch_saving(self) -> Decimal:
        return self.spend - self.batch_spend


def price_records(
    records: Iterable[RequestRecord], book: PriceBook
) -> tuple[tuple[Priced, ...], tuple[RequestRecord, ...]]:
    """Partition records into priced and unpriced.

    A real operator's first log always contains a model the price book has never heard of,
    so an unknown pair is a reporting fact, not a crash. Opportunities are computed on the
    priced side only, and the unpriced volume is reported alongside the coverage figure.
    """
    priced: list[Priced] = []
    unpriced: list[RequestRecord] = []
    for record in records:
        try:
            entry = book.entry(record.provider, record.model)
        except UnknownModelError:
            unpriced.append(record)
            continue
        cached = record.cached_prompt_tokens or 0
        priced.append(
            Priced(
                record=record,
                entry=entry,
                spend=book.price(
                    record.provider,
                    record.model,
                    tokens_in=record.prompt_tokens,
                    tokens_out=record.completion_tokens,
                    cached_in=cached,
                ),
                uncached_spend=book.price(
                    record.provider,
                    record.model,
                    tokens_in=record.prompt_tokens,
                    tokens_out=record.completion_tokens,
                ),
                batch_spend=book.price(
                    record.provider,
                    record.model,
                    tokens_in=record.prompt_tokens,
                    tokens_out=record.completion_tokens,
                    cached_in=cached,
                    batch=True,
                ),
            )
        )
    return tuple(priced), tuple(unpriced)


def _total(items: Iterable[Priced], amount: Callable[[Priced], Decimal]) -> Decimal:
    total = Decimal(0)
    for item in items:
        total += amount(item)
    return total


def _bounded(items: Sequence[Priced], amount: Callable[[Priced], Decimal]) -> Range:
    """Build the reported range from the log's own status field."""
    high = _total(items, amount)
    low = _total((item for item in items if item.ok), amount)
    return Range(low=low, high=high)


def _chronological(items: Sequence[Priced]) -> tuple[int, ...]:
    """Indices in a stable time order. ``raw_index`` breaks timestamp ties deterministically."""
    return tuple(
        sorted(
            range(len(items)),
            key=lambda index: (items[index].record.timestamp, items[index].record.raw_index),
        )
    )


def _select(items: Sequence[Priced], indices: Sequence[int]) -> tuple[Priced, ...]:
    return tuple(items[index] for index in indices)


def _zero(lever: str, *, note: str, required_changes: Sequence[str]) -> Opportunity:
    return Opportunity(
        lever=lever,
        risk_class=LEVER_RISK[lever],
        eligible_requests=0,
        eligible_spend=Decimal(0),
        projected_saving=Decimal(0),
        confidence_interval=Range(low=Decimal(0), high=Decimal(0)),
        assumptions=(note,),
        required_changes=tuple(required_changes),
        status="no_opportunity",
    )


# --------------------------------------------------------------------------------------
# D3 exact_dedup
# --------------------------------------------------------------------------------------


def eligible_exact_dedup(
    items: Sequence[Priced], *, window_seconds: int | None = None
) -> tuple[int, ...]:
    """Indices of repeat requests: an identical ``messages_hash`` seen inside the window."""
    last_seen: dict[str, float] = {}
    duplicates: list[int] = []
    for index in _chronological(items):
        record = items[index].record
        previous = last_seen.get(record.messages_hash)
        if previous is not None and (
            window_seconds is None or record.timestamp - previous <= window_seconds
        ):
            duplicates.append(index)
        last_seen[record.messages_hash] = record.timestamp
    return tuple(sorted(duplicates))


def detect_exact_dedup(
    items: Sequence[Priced], *, window_seconds: int | None = None
) -> Opportunity:
    """Identical requests inside the window. Response-identical, so the saving is the whole
    repeat."""
    return _build_exact_dedup(
        _select(items, eligible_exact_dedup(items, window_seconds=window_seconds)),
        window_seconds=window_seconds,
    )


def _build_exact_dedup(
    eligible: Sequence[Priced], *, window_seconds: int | None, displaced: int = 0
) -> Opportunity:
    """Size a repeat set that has already been selected.

    ``eligible`` is taken as given and is never re-filtered here. The precedence pipeline
    hands over the subset it assigned to this lever, and re-deriving eligibility from that
    subset would silently drop every repeat whose first occurrence is not part of it.
    """
    scope = (
        "over the whole file"
        if window_seconds is None
        else f"within a {window_seconds}-second window"
    )
    if not eligible:
        return _zero(
            "exact_dedup",
            note=(
                _DISPLACED_NOTE
                if displaced
                else f"no request hash repeats {scope}, so there is nothing to deduplicate"
            ),
            required_changes=(),
        )
    interval = _bounded(eligible, lambda item: item.spend)
    return Opportunity(
        lever="exact_dedup",
        risk_class=LEVER_RISK["exact_dedup"],
        eligible_requests=len(eligible),
        eligible_spend=_total(eligible, lambda item: item.spend),
        projected_saving=interval.low,
        confidence_interval=interval,
        assumptions=(
            PROJECTION_BASIS,
            f"requests are counted as repeats when an identical request hash was already seen "
            f"{scope}",
            "a cache serving a repeat returns the response the log already recorded, so the "
            "saving is that request's entire billed cost",
            BOUND_RULE,
        ),
        required_changes=(
            "enable the exact-response cache, keyed on the request hash the audit already computes",
            f"give the cache a time-to-live of at least the dedup window ({scope})",
            "scope the cache key by principal so one tenant's response is never served to another",
        ),
        status="ready",
    )


# --------------------------------------------------------------------------------------
# D1 prefix_cache
# --------------------------------------------------------------------------------------


def eligible_prefix_cache(items: Sequence[Priced]) -> tuple[int, ...]:
    """Indices of requests sharing a non-empty system prefix with at least one other."""
    counts = Counter(
        item.record.system_prefix_hash
        for item in items
        if item.record.system_prefix_hash != EMPTY_TEXT_HASH
    )
    return tuple(
        index
        for index, item in enumerate(items)
        if counts.get(item.record.system_prefix_hash, 0) > 1
    )


def detect_prefix_cache(items: Sequence[Priced]) -> Opportunity:
    """Repeated system prefixes.

    Reports realized cache behaviour where the log carries ``cached_prompt_tokens``, and
    refuses to size the opportunity where it does not. The prefix's token count is simply
    not in the record, and deriving it from ``prompt_tokens`` would be a guess dressed as
    arithmetic.
    """
    return _build_prefix_cache(_select(items, eligible_prefix_cache(items)))


def _build_prefix_cache(eligible: Sequence[Priced], *, displaced: int = 0) -> Opportunity:
    """Size an already-selected cluster set. ``eligible`` is never re-filtered here, because
    a cluster whose other members a higher lever claimed would stop looking like a cluster."""
    if not eligible:
        return _zero(
            "prefix_cache",
            note=(
                _DISPLACED_NOTE
                if displaced
                else "no system prefix is shared by two or more requests, so there is no "
                "prefix for a cache to hold"
            ),
            required_changes=(),
        )
    eligible_spend = _total(eligible, lambda item: item.spend)
    covered = tuple(item for item in eligible if item.record.cached_prompt_tokens is not None)
    cached_tokens = sum(item.record.cached_prompt_tokens or 0 for item in covered)
    clusters = len({item.record.system_prefix_hash for item in eligible})
    missing = len(eligible) - len(covered)
    if cached_tokens == 0:
        return Opportunity(
            lever="prefix_cache",
            risk_class=LEVER_RISK["prefix_cache"],
            eligible_requests=len(eligible),
            eligible_spend=eligible_spend,
            projected_saving=None,
            confidence_interval=None,
            assumptions=(),
            required_changes=(
                f"{len(eligible)} request(s) across {clusters} repeated system prefix(es) are "
                f"eligible, but this log reports no cache reads"
                + (f" and omits cached token counts on {missing} of them" if missing else ""),
                "fix: log cached_prompt_tokens per request, or route this traffic through the "
                "branchpilot gateway, which records it -- the saving is repeated prefix tokens "
                "times the input-minus-cached-input rate, and the prefix token count is not "
                "present in the log",
                "no figure is projected here: the record carries prompt_tokens but not prefix "
                "tokens, and the prefix length is never inferred from the prompt length",
            ),
            status="needs_token_counts",
        )
    measured_tokens = sum(item.record.prompt_tokens for item in covered)
    interval = _bounded(covered, lambda item: item.realized_cache_saving)
    share = (Decimal(cached_tokens) * 100 / Decimal(measured_tokens)).quantize(Decimal("0.1"))
    assumptions = [
        MEASURED_BASIS,
        f"{cached_tokens} of {measured_tokens} prompt tokens ({share}%) across {clusters} "
        f"repeated system prefix(es) were served from the provider's prefix cache",
        "the figure is what those recorded cache reads already saved against standard "
        "input rates -- it is money already captured, not money still available, so it is "
        "excluded from the addressable total",
        BOUND_RULE,
    ]
    required_changes = [
        "keep the shared prefix byte-identical and first in the request; any per-request "
        "preamble ahead of it silently disables the cache",
        f"the remaining {measured_tokens - cached_tokens} uncached prompt token(s) cannot be "
        "sized from this log, because the record carries no prefix token count",
    ]
    if missing:
        assumptions.insert(
            1,
            f"the share is measured over the {len(covered)} of {len(eligible)} eligible "
            "request(s) that report cached token counts; the rest are counted as eligible "
            "spend only",
        )
        required_changes.append(
            f"fix: log cached_prompt_tokens on the other {missing} eligible request(s), whose "
            "cache behaviour this log does not record"
        )
    return Opportunity(
        lever="prefix_cache",
        risk_class=LEVER_RISK["prefix_cache"],
        eligible_requests=len(eligible),
        eligible_spend=eligible_spend,
        projected_saving=interval.low,
        confidence_interval=interval,
        assumptions=tuple(assumptions),
        required_changes=tuple(required_changes),
        status="realized",
    )


# --------------------------------------------------------------------------------------
# D2 batch_lane
# --------------------------------------------------------------------------------------

BatchPredicate = Callable[[RequestRecord], bool]


def eligible_batch_lane(
    items: Sequence[Priced], predicate: BatchPredicate | None
) -> tuple[int, ...]:
    """Indices the caller's predicate accepts. No predicate means nothing is known eligible."""
    if predicate is None:
        return ()
    return tuple(index for index, item in enumerate(items) if predicate(item.record))


def detect_batch_lane(
    items: Sequence[Priced], *, predicate: BatchPredicate | None = None
) -> Opportunity:
    """The provider's asynchronous lane, priced from the price book's own batch rates."""
    return _build_batch_lane(
        _select(items, eligible_batch_lane(items, predicate)), predicate=predicate
    )


def _build_batch_lane(
    eligible: Sequence[Priced], *, predicate: BatchPredicate | None, displaced: int = 0
) -> Opportunity:
    """Size an already-selected eligible set. ``eligible`` is never re-filtered here."""
    if predicate is None:
        return Opportunity(
            lever="batch_lane",
            risk_class=LEVER_RISK["batch_lane"],
            eligible_requests=0,
            eligible_spend=Decimal(0),
            projected_saving=None,
            confidence_interval=None,
            assumptions=(),
            required_changes=(
                "no request is counted as batch-eligible without a rule, because tolerating a "
                "24-hour turnaround is a property of your application and not of the log",
                "fix: supply an eligibility predicate over RequestRecord -- for example every "
                "request whose group_key names an offline job -- and re-run; the saving is then "
                "computed from the price book's batch_input and batch_output rates, never from "
                "an assumed discount",
            ),
            status="needs_eligibility_rule",
        )
    if not eligible:
        return _zero(
            "batch_lane",
            note=(
                _DISPLACED_NOTE
                if displaced
                else "the supplied eligibility predicate matched no request in this log"
            ),
            required_changes=(),
        )
    interval = _bounded(eligible, lambda item: item.batch_saving)
    models = sorted({f"{item.entry.provider}/{item.entry.model}" for item in eligible})
    return Opportunity(
        lever="batch_lane",
        risk_class=LEVER_RISK["batch_lane"],
        eligible_requests=len(eligible),
        eligible_spend=_total(eligible, lambda item: item.spend),
        projected_saving=interval.low,
        confidence_interval=interval,
        assumptions=(
            PROJECTION_BASIS,
            "eligibility is exactly the requests your predicate accepted; the audit adds none "
            "of its own",
            "the saving is the difference between the configured synchronous and batch rates "
            f"for {', '.join(models)} -- no discount is assumed",
            "the batch lane returns the same model's output, so responses are unchanged; only "
            "latency is",
            BOUND_RULE,
        ),
        required_changes=(
            "move the predicate's traffic onto the provider's batch endpoint and accept its "
            "turnaround, which providers publish as up to 24 hours",
            "persist in-flight batch work so a restart cannot lose a job the provider has "
            "already been paid for",
        ),
        status="ready",
    )


# --------------------------------------------------------------------------------------
# D4 semantic_dedup
# --------------------------------------------------------------------------------------


def detect_semantic_dedup(items: Sequence[Priced]) -> Opportunity:
    """Near-duplicate requests. Undecidable without embeddings, so nothing is projected."""
    return Opportunity(
        lever="semantic_dedup",
        risk_class=LEVER_RISK["semantic_dedup"],
        eligible_requests=0,
        eligible_spend=Decimal(0),
        projected_saving=None,
        confidence_interval=None,
        assumptions=(),
        required_changes=(
            f"the audit hashes and discards prompt text on read, so none of the "
            f"{len(items)} priced request(s) can be compared for similarity after the fact",
            "fix: configure an embedding backend and a similarity threshold, then re-run; "
            "until then no eligible request can be named and no figure is projected",
            "this lever can change the response your caller receives, so it must be validated "
            "on your own traffic before adoption",
        ),
        status="needs_embeddings",
    )


# --------------------------------------------------------------------------------------
# D5 tier_routing
# --------------------------------------------------------------------------------------


def _cheaper_models(entry: PriceEntry, book: PriceBook) -> tuple[str, ...]:
    return tuple(
        model
        for model in book.models(entry.provider)
        if (candidate := book.entry(entry.provider, model)).input < entry.input
        and candidate.output < entry.output
    )


def eligible_tier_routing(items: Sequence[Priced], book: PriceBook) -> tuple[int, ...]:
    """Indices whose provider offers a strictly cheaper model on both input and output."""
    cache: dict[tuple[str, str], bool] = {}
    eligible: list[int] = []
    for index, item in enumerate(items):
        key = (item.entry.provider, item.entry.model)
        if key not in cache:
            cache[key] = bool(_cheaper_models(item.entry, book))
        if cache[key]:
            eligible.append(index)
    return tuple(eligible)


def detect_tier_routing(items: Sequence[Priced], *, book: PriceBook) -> Opportunity:
    """Spend a cheaper same-provider model could address. Whether it should is an evaluation."""
    eligible = _select(items, eligible_tier_routing(items, book))
    if not eligible:
        return _zero(
            "tier_routing",
            note=(
                "every model in this log is already the cheapest configured model for its "
                "provider on both input and output"
            ),
            required_changes=(),
        )
    alternatives = sorted(
        {
            f"{item.entry.provider}/{item.entry.model} -> {', '.join(cheaper)}"
            for item in eligible
            if (cheaper := _cheaper_models(item.entry, book))
        }
    )
    return Opportunity(
        lever="tier_routing",
        risk_class=LEVER_RISK["tier_routing"],
        eligible_requests=len(eligible),
        eligible_spend=_total(eligible, lambda item: item.spend),
        projected_saving=None,
        confidence_interval=None,
        assumptions=(),
        required_changes=(
            "the figure above is addressable spend, not a saving: a cheaper model exists for "
            f"{len(eligible)} request(s) ({'; '.join(alternatives)})",
            "fix: score the cheaper tier against the current model on a labelled slice of your "
            "own traffic; no saving is projected because whether the cheaper model is good "
            "enough is a property of your workload, not of the price book",
            "this lever can change the response your caller receives, so it must be validated "
            "on your own traffic before adoption",
        ),
        status="needs_evaluation",
    )


# --------------------------------------------------------------------------------------
# D6 sampling
# --------------------------------------------------------------------------------------


def eligible_sampling(items: Sequence[Priced]) -> tuple[int, ...]:
    """Indices of every sample after the first in each multi-sample group."""
    counts = Counter(item.record.group_key for item in items if item.record.group_key is not None)
    seen: set[str] = set()
    redundant: list[int] = []
    for index in _chronological(items):
        key = items[index].record.group_key
        if key is None or counts[key] < 2:
            continue
        if key in seen:
            redundant.append(index)
        else:
            seen.add(key)
    return tuple(sorted(redundant))


def detect_sampling(items: Sequence[Priced]) -> Opportunity:
    """Multi-sample groups. Returns zero on a single-sample workload rather than erroring."""
    return _build_sampling(_select(items, eligible_sampling(items)))


def _build_sampling(eligible: Sequence[Priced], *, displaced: int = 0) -> Opportunity:
    """Size an already-selected redundant-sample set. ``eligible`` is never re-filtered here,
    because every member of it is by construction a group's second or later sample."""
    if not eligible:
        return _zero(
            "sampling",
            note=(
                _DISPLACED_NOTE
                if displaced
                else "no group_key in this log names more than one request, so the workload "
                "draws a single sample per call and there is nothing to collapse"
            ),
            required_changes=(),
        )
    interval = _bounded(eligible, lambda item: item.spend)
    groups = len({item.record.group_key for item in eligible})
    return Opportunity(
        lever="sampling",
        risk_class=LEVER_RISK["sampling"],
        eligible_requests=len(eligible),
        eligible_spend=_total(eligible, lambda item: item.spend),
        projected_saving=interval.low,
        confidence_interval=interval,
        assumptions=(
            PROJECTION_BASIS,
            f"multi-sample groups are the {groups} group_key value(s) this log attaches to more "
            "than one request; the audit does not infer grouping from anything else",
            "the figure is the ceiling: it assumes every group collapses to its first sample. "
            "The realized figure depends on how much accuracy the extra samples were buying, "
            "which only an evaluation on your own traffic can say",
            BOUND_RULE,
        ),
        required_changes=(
            "measure accuracy at one sample against accuracy at the observed sample count on a "
            "labelled slice of your traffic before collapsing any group",
            "adopt a per-request sample budget rather than a fixed count, so easy requests stop "
            "at one sample and hard ones do not",
        ),
        status="ready",
    )


# --------------------------------------------------------------------------------------
# precedence pipeline
# --------------------------------------------------------------------------------------


def detect_opportunities(
    items: Sequence[Priced],
    *,
    book: PriceBook,
    window_seconds: int | None = None,
    batch_predicate: BatchPredicate | None = None,
) -> tuple[tuple[Opportunity, ...], tuple[Overlap, ...]]:
    """Run every detector, resolve overlaps by precedence, and report what was displaced.

    Eligibility is computed once per lever over the whole log. Precedence then assigns each
    request to the first lever that wanted it, and each lever is sized from exactly the
    subset it was assigned -- so a request eligible for two levers is counted once, under
    the higher-precedence one, and appears in an :class:`~branchpilot.audit.result.Overlap`
    under the lower one.
    """
    raw = {
        "exact_dedup": eligible_exact_dedup(items, window_seconds=window_seconds),
        "prefix_cache": eligible_prefix_cache(items),
        "batch_lane": eligible_batch_lane(items, batch_predicate),
        "sampling": eligible_sampling(items),
    }
    claimed: set[int] = set()
    assigned: dict[str, tuple[int, ...]] = {}
    displaced: dict[str, int] = {}
    for lever in CLAIMING_LEVERS:
        keep = tuple(index for index in raw[lever] if index not in claimed)
        assigned[lever] = keep
        displaced[lever] = len(raw[lever]) - len(keep)
        claimed.update(keep)

    detected = {
        "exact_dedup": _build_exact_dedup(
            _select(items, assigned["exact_dedup"]),
            window_seconds=window_seconds,
            displaced=displaced["exact_dedup"],
        ),
        "prefix_cache": _build_prefix_cache(
            _select(items, assigned["prefix_cache"]), displaced=displaced["prefix_cache"]
        ),
        "batch_lane": _build_batch_lane(
            _select(items, assigned["batch_lane"]),
            predicate=batch_predicate,
            displaced=displaced["batch_lane"],
        ),
        "semantic_dedup": detect_semantic_dedup(items),
        "tier_routing": detect_tier_routing(items, book=book),
        "sampling": _build_sampling(
            _select(items, assigned["sampling"]), displaced=displaced["sampling"]
        ),
    }

    overlaps: list[Overlap] = []
    for position, lever in enumerate(CLAIMING_LEVERS):
        for higher in CLAIMING_LEVERS[:position]:
            shared = sorted(set(raw[lever]) & set(assigned[higher]))
            if not shared:
                continue
            overlaps.append(
                Overlap(
                    lever=lever,
                    claimed_by=higher,
                    requests=len(shared),
                    spend=_total(_select(items, shared), lambda item: item.spend),
                )
            )
    order = {lever: index for index, lever in enumerate(PRECEDENCE)}
    overlaps.sort(key=lambda item: (order[item.lever], order[item.claimed_by]))
    return tuple(detected[lever] for lever in PRECEDENCE), tuple(overlaps)


# --------------------------------------------------------------------------------------
# workload description
# --------------------------------------------------------------------------------------


def spend_by_model(items: Sequence[Priced]) -> tuple[ModelSpend, ...]:
    """Observed spend per (provider, model), largest first."""
    groups: dict[tuple[str, str], list[Priced]] = {}
    for item in items:
        groups.setdefault((item.record.provider, item.record.model), []).append(item)
    rows = [
        ModelSpend(
            provider=provider,
            model=model,
            requests=len(group),
            prompt_tokens=sum(item.record.prompt_tokens for item in group),
            cached_prompt_tokens=sum(item.record.cached_prompt_tokens or 0 for item in group),
            completion_tokens=sum(item.record.completion_tokens for item in group),
            spend=_total(group, lambda item: item.spend),
        )
        for (provider, model), group in groups.items()
    ]
    rows.sort(key=lambda row: (-row.spend, row.provider, row.model))
    return tuple(rows)


def profile_workload(
    items: Sequence[Priced],
    unpriced: Sequence[RequestRecord],
    report: IngestReport,
) -> WorkloadProfile:
    """Describe the log itself, independent of any lever."""
    records = [item.record for item in items] + list(unpriced)
    timestamps = sorted(record.timestamp for record in records)
    return WorkloadProfile(
        records=len(records),
        priced_records=len(items),
        unpriced_records=len(unpriced),
        ok_records=sum(1 for record in records if record.status == _OK),
        unpriced_pairs=tuple(sorted({(record.provider, record.model) for record in unpriced})),
        prompt_tokens=sum(record.prompt_tokens for record in records),
        cached_prompt_tokens=sum(record.cached_prompt_tokens or 0 for record in records),
        completion_tokens=sum(record.completion_tokens for record in records),
        records_with_cached_counts=sum(
            1 for record in records if record.cached_prompt_tokens is not None
        ),
        distinct_message_hashes=len({record.messages_hash for record in records}),
        distinct_prefix_hashes=len(
            {
                record.system_prefix_hash
                for record in records
                if record.system_prefix_hash != EMPTY_TEXT_HASH
            }
        ),
        parsed=report.parsed,
        skipped=report.skipped,
        skip_reasons=dict(report.reasons),
        first_timestamp=timestamps[0] if timestamps else None,
        last_timestamp=timestamps[-1] if timestamps else None,
    )
