"""The spend audit: what a log's own numbers can prove about cheaper ways to run it.

Every figure this package produces is exact arithmetic over tokens the log actually
recorded, times rates the price book actually configures. There is no float in any cost
path and no estimate anywhere: a lever the log cannot size reports ``projected_saving=None``
with a ``needs_*`` status naming the one missing input, never a plausible-looking guess.

Running an audit
----------------
:func:`~branchpilot.audit.detectors.price_records` partitions records into a priced side and
an unpriced one -- an unknown ``(provider, model)`` pair is a coverage fact, not a crash --
and :func:`~branchpilot.audit.detectors.detect_opportunities` runs every lever over the
priced side::

    from branchpilot.audit import detect_opportunities, price_records, profile_workload
    from branchpilot.ingest import read_requests
    from branchpilot.pricing import PriceBook

    book = PriceBook.load()
    stream = read_requests("traffic.jsonl")
    priced, unpriced = price_records(stream, book)
    opportunities, overlaps = detect_opportunities(priced, book=book)
    workload = profile_workload(priced, unpriced, stream.report())

The six levers are ``exact_dedup``, ``prefix_cache``, ``batch_lane``, ``semantic_dedup``,
``tier_routing``, and ``sampling``. Eligibility is computed once per lever over the whole
log and then assigned by the precedence
``exact_dedup > prefix_cache > batch_lane > semantic_dedup > tier_routing > sampling``, so a
request eligible for two levers is counted once -- under the higher-precedence one -- and
reported under the lower one as an :class:`~branchpilot.audit.result.Overlap`.

Risk and the headline
---------------------
Each lever's risk class is fixed by what the lever does, never configurable:
:data:`~branchpilot.audit.risk.IDENTICAL` levers return the same response, while
:data:`~branchpilot.audit.risk.QUALITY_AFFECTING` levers can change it.
:func:`~branchpilot.audit.risk.headline` therefore sums only ``IDENTICAL`` levers by
default; ``include_quality_affecting=True`` widens the range and attaches
:data:`~branchpilot.audit.risk.HEADLINE_WARNING`.
"""

from __future__ import annotations

from branchpilot.audit.detectors import (
    BOUND_RULE,
    CLAIMING_LEVERS,
    MEASURED_BASIS,
    PROJECTION_BASIS,
    BatchPredicate,
    Priced,
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
)
from branchpilot.audit.result import (
    BLOCKED_STATUSES,
    PROJECTING_STATUSES,
    STATUSES,
    AuditResult,
    ModelSpend,
    Opportunity,
    Overlap,
    PriceBookProvenance,
    Range,
    WorkloadProfile,
    sum_money,
)
from branchpilot.audit.risk import (
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
    headline,
    quality_affecting_levers,
)

__all__ = [
    "BLOCKED_STATUSES",
    "BOUND_RULE",
    "CLAIMING_LEVERS",
    "HEADLINE_WARNING",
    "IDENTICAL",
    "LEVERS",
    "LEVER_RISK",
    "MEASURED_BASIS",
    "PRECEDENCE",
    "PROJECTING_STATUSES",
    "PROJECTION_BASIS",
    "QUALITY_AFFECTING",
    "RISK_CLASSES",
    "STATUSES",
    "VALIDATION_COMMANDS",
    "VALIDATION_REQUIREMENTS",
    "AuditResult",
    "BatchPredicate",
    "Headline",
    "ModelSpend",
    "Opportunity",
    "Overlap",
    "PriceBookProvenance",
    "Priced",
    "Range",
    "WorkloadProfile",
    "detect_batch_lane",
    "detect_exact_dedup",
    "detect_opportunities",
    "detect_prefix_cache",
    "detect_sampling",
    "detect_semantic_dedup",
    "detect_tier_routing",
    "headline",
    "price_records",
    "profile_workload",
    "quality_affecting_levers",
    "spend_by_model",
    "sum_money",
]
