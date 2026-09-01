"""Risk classification of spend levers, and the headline range rule.

Two classes, and only two, because an operator's first question about a savings figure is
"will this change my outputs?":

``IDENTICAL``
    The response the caller receives is byte-identical, or the provider guarantees
    equivalence. Exact deduplication returns the same response; a prefix cache read
    returns the same completion the uncached call would have; the batch lane is the same
    model at a published discount.
``QUALITY_AFFECTING``
    The response can differ. A semantic cache returns a *near* neighbour's answer; tier
    routing sends work to a different model; dropping samples changes which candidate
    wins.

The default headline range therefore includes only ``IDENTICAL`` levers.
``QUALITY_AFFECTING`` levers are reported in their own section with the evidence required
before adoption, and are folded into the headline only under
``--include-quality-affecting``, which adds a warning block.

This module deliberately imports nothing from :mod:`branchpilot.audit.result`; the
dependency runs the other way, so ``result`` can validate a lever's class at construction.
:func:`headline` reads ``lever``, ``risk_class``, ``status``, and ``confidence_interval``
off whatever opportunities it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from collections.abc import Sequence

    from branchpilot.audit.result import Opportunity

__all__ = [
    "HEADLINE_WARNING",
    "IDENTICAL",
    "LEVER_RISK",
    "LEVERS",
    "PRECEDENCE",
    "QUALITY_AFFECTING",
    "RISK_CLASSES",
    "VALIDATION_COMMANDS",
    "VALIDATION_REQUIREMENTS",
    "Headline",
    "headline",
    "quality_affecting_levers",
]

IDENTICAL = "IDENTICAL"
"""Response-identical or provider-guaranteed. Safe to include in the headline."""

QUALITY_AFFECTING = "QUALITY_AFFECTING"
"""Can change outputs. Quarantined from the headline until the operator validates it."""

RISK_CLASSES = (IDENTICAL, QUALITY_AFFECTING)

LEVERS = (
    "prefix_cache",
    "batch_lane",
    "exact_dedup",
    "semantic_dedup",
    "tier_routing",
    "sampling",
)
"""Every lever the audit reports on."""

PRECEDENCE = (
    "exact_dedup",
    "prefix_cache",
    "batch_lane",
    "semantic_dedup",
    "tier_routing",
    "sampling",
)
"""Overlap precedence. A request eligible for two levers is counted under the earlier one."""

LEVER_RISK = {
    "exact_dedup": IDENTICAL,
    "prefix_cache": IDENTICAL,
    "batch_lane": IDENTICAL,
    "semantic_dedup": QUALITY_AFFECTING,
    "tier_routing": QUALITY_AFFECTING,
    "sampling": QUALITY_AFFECTING,
}
"""The class of each lever. Fixed by what the lever does, never configurable."""

VALIDATION_COMMANDS = {
    "semantic_dedup": (
        "branchpilot audit LOGS --include-quality-affecting --json quality-affecting.json"
    ),
    "tier_routing": (
        "branchpilot audit LOGS --include-quality-affecting --json quality-affecting.json"
    ),
    "sampling": (
        "branchpilot audit LOGS --include-quality-affecting --json quality-affecting.json"
    ),
}
"""The exact command that reports a quality-affecting lever's figures for review."""

VALIDATION_REQUIREMENTS = {
    "semantic_dedup": (
        "compare the cached neighbour's response against a fresh response on a held-out slice "
        "of your own traffic and accept a similarity threshold only after reviewing the "
        "disagreements; a shadow lane that does this for you is not part of this version, so "
        "this lever must not be adopted from this report alone"
    ),
    "tier_routing": (
        "score the cheaper tier against the current model on a labelled slice of your own "
        "traffic and keep the escalation predicate observed-only until the scores agree; a "
        "shadow lane that does this for you is not part of this version, so this lever must "
        "not be adopted from this report alone"
    ),
    "sampling": (
        "measure accuracy at one sample against accuracy at the observed sample count on a "
        "labelled slice of your own traffic before collapsing any group; a shadow lane that "
        "does this for you is not part of this version, so this lever must not be adopted "
        "from this report alone"
    ),
}
"""What evidence an operator must gather before adopting a quality-affecting lever."""

HEADLINE_WARNING = (
    "WARNING: this headline includes QUALITY_AFFECTING levers. Semantic deduplication, tier "
    "routing, and sampling can change the responses your callers receive, so these figures "
    "are not adoptable as they stand; fix: validate each quality-affecting lever on your own "
    "traffic first, or re-run without --include-quality-affecting for the "
    "response-identical-only range"
)


def quality_affecting_levers() -> tuple[str, ...]:
    """Levers whose adoption can change outputs, in precedence order."""
    return tuple(lever for lever in PRECEDENCE if LEVER_RISK[lever] == QUALITY_AFFECTING)


@dataclass(frozen=True, slots=True)
class Headline:
    """The one range the report leads with, plus exactly which levers are inside it."""

    low: Decimal
    high: Decimal
    levers: tuple[str, ...]
    include_quality_affecting: bool
    warning: str | None

    def __post_init__(self) -> None:
        for name in ("low", "high"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise TypeError(
                    f"headline {name} must be a Decimal, not {type(value).__name__}; "
                    "fix: keep money in Decimal end to end -- never float"
                )
            if value < 0:
                raise ValueError(
                    f"headline {name} cannot be negative; fix: a lever that cannot "
                    "substantiate a saving reports None, not a negative number"
                )
        if self.low > self.high:
            raise ValueError(
                f"headline low {self.low} exceeds high {self.high}; "
                "fix: build the range as (conservative bound, optimistic bound)"
            )
        if not isinstance(self.include_quality_affecting, bool):
            raise TypeError(
                "headline include_quality_affecting must be True or False; "
                "fix: pass the parsed --include-quality-affecting flag"
            )
        expected = HEADLINE_WARNING if self.include_quality_affecting else None
        if self.warning != expected:
            raise ValueError(
                "a headline that includes quality-affecting levers must carry the warning, and "
                "one that does not must carry none; fix: build headlines with "
                "branchpilot.audit.risk.headline()"
            )

    @property
    def empty(self) -> bool:
        """True when no lever in scope substantiated any money."""
        return self.high == 0


def headline(
    opportunities: Sequence[Opportunity],
    *,
    include_quality_affecting: bool = False,
) -> Headline:
    """Sum the addressable ranges of the levers that are in scope.

    Only opportunities with status ``ready`` contribute. ``realized`` figures are money the
    log shows is *already* being saved, so folding them in would credit the operator twice
    for a win they already hold; they are reported separately and never summed here.
    """
    if not isinstance(include_quality_affecting, bool):
        raise TypeError(
            "include_quality_affecting must be True or False; "
            "fix: pass the parsed --include-quality-affecting flag"
        )
    allowed = set(RISK_CLASSES) if include_quality_affecting else {IDENTICAL}
    low = Decimal(0)
    high = Decimal(0)
    levers: list[str] = []
    order = {lever: index for index, lever in enumerate(PRECEDENCE)}
    for opportunity in sorted(opportunities, key=lambda item: order[item.lever]):
        if opportunity.status != "ready" or opportunity.risk_class not in allowed:
            continue
        interval = opportunity.confidence_interval
        low += interval.low
        high += interval.high
        levers.append(opportunity.lever)
    return Headline(
        low=low,
        high=high,
        levers=tuple(levers),
        include_quality_affecting=include_quality_affecting,
        warning=HEADLINE_WARNING if include_quality_affecting else None,
    )
