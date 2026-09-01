"""Value types the audit produces: opportunities, overlaps, and the whole result.

Every money field is a :class:`~decimal.Decimal`. There is no float anywhere in a cost
path, because a report whose figures move with the platform's rounding cannot be defended.

The types enforce the two rules that make the report honest, at construction time rather
than in the renderer:

*No projection without evidence.* An :class:`Opportunity` that carries a
``projected_saving`` must also carry a confidence interval containing it and a non-empty
assumptions list. A detector that cannot substantiate a figure sets ``projected_saving`` to
``None`` and says why in ``status`` -- never a guess.

*Realized is not addressable.* Where a log records ``cached_prompt_tokens`` the audit can
measure what prefix caching is *already* saving. That figure is real but it is not money
the operator can go and get, so it carries status ``realized`` and is excluded from every
addressable total. It is the only figure in the report labelled "measured from the log";
everything else is a projection from observed tokens times configured prices.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from branchpilot.audit.risk import LEVER_RISK, PRECEDENCE, RISK_CLASSES

__all__ = [
    "BLOCKED_STATUSES",
    "PROJECTING_STATUSES",
    "STATUSES",
    "AuditResult",
    "ModelSpend",
    "Opportunity",
    "Overlap",
    "PriceBookProvenance",
    "Range",
    "WorkloadProfile",
    "sum_money",
]

STATUSES = (
    "ready",
    "realized",
    "no_opportunity",
    "needs_eligibility_rule",
    "needs_token_counts",
    "needs_embeddings",
    "needs_evaluation",
)
"""Closed vocabulary. ``ready`` is addressable, ``realized`` is already captured, the
``needs_*`` statuses each name the one input that is missing."""

PROJECTING_STATUSES = ("ready", "realized", "no_opportunity")
"""Statuses that must carry a figure."""

BLOCKED_STATUSES = tuple(status for status in STATUSES if status not in PROJECTING_STATUSES)
"""Statuses that must not carry a figure."""


def _require_money(value: Any, label: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(
            f"{label} must be a Decimal, not {type(value).__name__}; "
            "fix: keep money in Decimal end to end -- never float"
        )
    if not value.is_finite():
        raise ValueError(
            f"{label} must be a finite amount; fix: drop records whose token counts do not "
            "price exactly instead of forwarding a non-finite total"
        )
    if value < 0:
        raise ValueError(
            f"{label} cannot be negative; fix: report a lever that saves nothing as zero, and "
            "one that cannot be sized as None with a status"
        )


def _require_count(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer; fix: pass an int count for {label}")
    if value < 0:
        raise ValueError(f"{label} cannot be negative; fix: pass a non-negative {label}")


def _require_phrases(values: Any, label: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise TypeError(
            f"{label} must be a sequence of strings, not {type(values).__name__}; "
            f"fix: pass a list or tuple of {label}"
        )
    phrases = tuple(values)
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError(
                f"{label} entries must be non-empty strings; "
                f"fix: drop empty entries instead of rendering a blank {label} line"
            )
    return phrases


@dataclass(frozen=True, slots=True)
class Range:
    """A money range. Every projected figure in the report is one of these, never a point.

    The bounds are not a statistical interval and are not modelled: ``low`` is the figure
    computed over requests the log records as ``status == "ok"``, and ``high`` adds the
    requests whose logged status is something else, because a provider may or may not have
    billed a failed call. Both bounds are exact arithmetic over observed tokens.
    """

    low: Decimal
    high: Decimal

    def __post_init__(self) -> None:
        _require_money(self.low, "range low")
        _require_money(self.high, "range high")
        if self.low > self.high:
            raise ValueError(
                f"range low {self.low} exceeds high {self.high}; "
                "fix: build the range as (conservative bound, optimistic bound)"
            )

    def contains(self, amount: Decimal) -> bool:
        return self.low <= amount <= self.high

    @property
    def degenerate(self) -> bool:
        """True when both bounds coincide, which happens when every request succeeded."""
        return self.low == self.high


@dataclass(frozen=True, slots=True)
class Opportunity:
    """One lever's finding for one log."""

    lever: str
    risk_class: str
    eligible_requests: int
    eligible_spend: Decimal
    projected_saving: Decimal | None
    confidence_interval: Range | None
    assumptions: tuple[str, ...]
    required_changes: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if self.lever not in LEVER_RISK:
            known = ", ".join(PRECEDENCE)
            raise ValueError(
                f"unknown lever {self.lever!r}; fix: use one of the six audited levers: {known}"
            )
        if self.risk_class not in RISK_CLASSES:
            raise ValueError(
                f"unknown risk class {self.risk_class!r}; "
                f"fix: use one of: {', '.join(RISK_CLASSES)}"
            )
        expected = LEVER_RISK[self.lever]
        if self.risk_class != expected:
            raise ValueError(
                f"lever {self.lever!r} is {expected}, not {self.risk_class}; fix: a lever's risk "
                "class is fixed by what the lever does -- read it from "
                "branchpilot.audit.risk.LEVER_RISK instead of passing one"
            )
        if self.status not in STATUSES:
            raise ValueError(
                f"unknown status {self.status!r}; fix: use one of: {', '.join(STATUSES)}"
            )
        _require_count(self.eligible_requests, "eligible_requests")
        _require_money(self.eligible_spend, "eligible_spend")
        object.__setattr__(self, "assumptions", _require_phrases(self.assumptions, "assumptions"))
        object.__setattr__(
            self, "required_changes", _require_phrases(self.required_changes, "required_changes")
        )
        if self.status in BLOCKED_STATUSES:
            self._check_blocked()
        else:
            self._check_projected()

    def _check_blocked(self) -> None:
        if self.projected_saving is not None or self.confidence_interval is not None:
            raise ValueError(
                f"lever {self.lever!r} reports status {self.status!r} but still carries a "
                "figure; fix: a detector that cannot substantiate a number returns "
                "projected_saving=None and confidence_interval=None"
            )
        if not self.required_changes:
            raise ValueError(
                f"lever {self.lever!r} reports status {self.status!r} without naming what is "
                "missing; fix: add the required change that would unblock the lever"
            )

    def _check_projected(self) -> None:
        if self.projected_saving is None or self.confidence_interval is None:
            raise ValueError(
                f"lever {self.lever!r} reports status {self.status!r} but carries no figure; "
                "fix: pass both projected_saving and confidence_interval, or use a needs_* "
                f"status: {', '.join(BLOCKED_STATUSES)}"
            )
        _require_money(self.projected_saving, "projected_saving")
        if not isinstance(self.confidence_interval, Range):
            raise TypeError(
                "confidence_interval must be a branchpilot.audit.result.Range; "
                "fix: wrap the bounds in Range(low=..., high=...)"
            )
        if not self.confidence_interval.contains(self.projected_saving):
            raise ValueError(
                f"lever {self.lever!r} projects {self.projected_saving} outside its range "
                f"{self.confidence_interval.low}..{self.confidence_interval.high}; "
                "fix: report the conservative bound as the projection"
            )
        if not self.assumptions:
            raise ValueError(
                f"lever {self.lever!r} projects {self.projected_saving} with no assumptions; "
                "fix: every projection carries the assumptions the report renders beside it -- "
                "a bare number cannot be defended"
            )
        if self.status == "ready" and self.confidence_interval.high > self.eligible_spend:
            raise ValueError(
                f"lever {self.lever!r} projects up to {self.confidence_interval.high} against "
                f"{self.eligible_spend} of eligible spend; fix: a lever cannot save more than "
                "the traffic it addresses actually cost"
            )
        if self.status == "no_opportunity" and self.projected_saving != 0:
            raise ValueError(
                f"lever {self.lever!r} reports no opportunity but projects "
                f"{self.projected_saving}; fix: use status 'ready' for a non-zero projection"
            )

    @property
    def addressable(self) -> bool:
        """True when this lever contributes money to the headline range."""
        return self.status == "ready"

    @property
    def measured(self) -> bool:
        """True when the figure was measured from the log rather than projected."""
        return self.status == "realized"

    def to_dict(self) -> dict[str, Any]:
        interval = self.confidence_interval
        saving = self.projected_saving
        return {
            "lever": self.lever,
            "risk_class": self.risk_class,
            "status": self.status,
            "eligible_requests": self.eligible_requests,
            "eligible_spend": str(self.eligible_spend),
            "projected_saving": None if saving is None else str(saving),
            "confidence_interval": (
                None if interval is None else {"low": str(interval.low), "high": str(interval.high)}
            ),
            "assumptions": list(self.assumptions),
            "required_changes": list(self.required_changes),
        }


@dataclass(frozen=True, slots=True)
class Overlap:
    """Requests a lever was eligible for that a higher-precedence lever counted instead."""

    lever: str
    claimed_by: str
    requests: int
    spend: Decimal

    def __post_init__(self) -> None:
        order = {lever: index for index, lever in enumerate(PRECEDENCE)}
        for name in ("lever", "claimed_by"):
            value = getattr(self, name)
            if value not in order:
                raise ValueError(
                    f"unknown lever {value!r} in overlap {name}; "
                    f"fix: use one of: {', '.join(PRECEDENCE)}"
                )
        if order[self.claimed_by] >= order[self.lever]:
            raise ValueError(
                f"{self.claimed_by!r} does not outrank {self.lever!r}; fix: overlaps are always "
                f"reported against the higher-precedence lever, in the order "
                f"{' > '.join(PRECEDENCE)}"
            )
        _require_count(self.requests, "overlap requests")
        _require_money(self.spend, "overlap spend")
        if self.requests == 0:
            raise ValueError(
                "an overlap with no requests is not an overlap; "
                "fix: omit empty overlaps instead of reporting a zero row"
            )

    def message(self) -> str:
        return (
            f"{self.requests} request(s) worth {self.spend} were eligible for "
            f"{self.lever} but are counted under {self.claimed_by}, which takes precedence"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lever": self.lever,
            "claimed_by": self.claimed_by,
            "requests": self.requests,
            "spend": str(self.spend),
        }


@dataclass(frozen=True, slots=True)
class ModelSpend:
    """Observed spend for one (provider, model) pair."""

    provider: str
    model: str
    requests: int
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    spend: Decimal

    def __post_init__(self) -> None:
        for name in ("provider", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"model spend {name} must be a non-empty string")
        for name in ("requests", "prompt_tokens", "cached_prompt_tokens", "completion_tokens"):
            _require_count(getattr(self, name), f"model spend {name}")
        _require_money(self.spend, "model spend")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "spend": str(self.spend),
        }


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    """What the log contains, before any lever is considered."""

    records: int
    priced_records: int
    unpriced_records: int
    ok_records: int
    unpriced_pairs: tuple[tuple[str, str], ...]
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    records_with_cached_counts: int
    distinct_message_hashes: int
    distinct_prefix_hashes: int
    parsed: int
    skipped: int
    skip_reasons: Mapping[str, int]
    first_timestamp: float | None
    last_timestamp: float | None

    def __post_init__(self) -> None:
        for name in (
            "records",
            "priced_records",
            "unpriced_records",
            "ok_records",
            "prompt_tokens",
            "cached_prompt_tokens",
            "completion_tokens",
            "records_with_cached_counts",
            "distinct_message_hashes",
            "distinct_prefix_hashes",
            "parsed",
            "skipped",
        ):
            _require_count(getattr(self, name), f"workload {name}")
        if self.priced_records + self.unpriced_records != self.records:
            raise ValueError(
                f"priced {self.priced_records} plus unpriced {self.unpriced_records} does not "
                f"equal {self.records} records; fix: partition every record into exactly one of "
                "the two buckets"
            )
        object.__setattr__(self, "unpriced_pairs", tuple(self.unpriced_pairs))

    @property
    def coverage_percent(self) -> Decimal:
        """Share of records the price book could price, as a percentage to one decimal."""
        if self.records == 0:
            return Decimal("0.0")
        return (Decimal(self.priced_records) * 100 / Decimal(self.records)).quantize(Decimal("0.1"))

    @property
    def duration_seconds(self) -> float | None:
        if self.first_timestamp is None or self.last_timestamp is None:
            return None
        return self.last_timestamp - self.first_timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "priced_records": self.priced_records,
            "unpriced_records": self.unpriced_records,
            "ok_records": self.ok_records,
            "coverage_percent": str(self.coverage_percent),
            "unpriced_pairs": [list(pair) for pair in self.unpriced_pairs],
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "records_with_cached_counts": self.records_with_cached_counts,
            "distinct_message_hashes": self.distinct_message_hashes,
            "distinct_prefix_hashes": self.distinct_prefix_hashes,
            "parsed": self.parsed,
            "skipped": self.skipped,
            "skip_reasons": dict(sorted(self.skip_reasons.items())),
        }


@dataclass(frozen=True, slots=True)
class PriceBookProvenance:
    """Which rate cards priced this log, so any figure can be recomputed by hand."""

    schema_version: int
    path: str
    currency: str
    entries: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        _require_count(self.schema_version, "price book schema_version")
        object.__setattr__(self, "entries", tuple(self.entries))

    @property
    def effective_dates(self) -> tuple[str, ...]:
        return tuple(sorted({entry[2] for entry in self.entries}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "currency": self.currency,
            "entries": [
                {"provider": provider, "model": model, "effective_date": day}
                for provider, model, day in self.entries
            ],
            "effective_dates": list(self.effective_dates),
        }


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Everything one audit produced. Renderers read this and add nothing of their own."""

    source: str
    source_sha256: str
    format_id: str
    currency: str
    observed_spend: Decimal
    spend_by_model: tuple[ModelSpend, ...]
    workload: WorkloadProfile
    opportunities: tuple[Opportunity, ...]
    overlaps: tuple[Overlap, ...]
    price_book: PriceBookProvenance
    window_seconds: int | None
    reproduction_command: str
    fixes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_money(self.observed_spend, "observed spend")
        object.__setattr__(self, "spend_by_model", tuple(self.spend_by_model))
        object.__setattr__(self, "overlaps", tuple(self.overlaps))
        object.__setattr__(self, "fixes", tuple(self.fixes))
        order = {lever: index for index, lever in enumerate(PRECEDENCE)}
        opportunities = tuple(sorted(self.opportunities, key=lambda item: order[item.lever]))
        levers = [item.lever for item in opportunities]
        if levers != list(PRECEDENCE):
            raise ValueError(
                f"an audit result must carry exactly one opportunity per lever, got: "
                f"{', '.join(levers) or 'none'}; fix: report every lever, using a needs_* status "
                "for the ones this log cannot substantiate"
            )
        object.__setattr__(self, "opportunities", opportunities)
        if self.window_seconds is not None:
            _require_count(self.window_seconds, "window_seconds")

    def opportunity(self, lever: str) -> Opportunity:
        for item in self.opportunities:
            if item.lever == lever:
                return item
        raise KeyError(f"no opportunity for lever {lever!r}")

    def ranked(self, *, include_quality_affecting: bool = False) -> tuple[Opportunity, ...]:
        """Addressable opportunities, largest conservative bound first."""
        from branchpilot.audit.risk import IDENTICAL, RISK_CLASSES

        allowed = set(RISK_CLASSES) if include_quality_affecting else {IDENTICAL}
        order = {lever: index for index, lever in enumerate(PRECEDENCE)}
        candidates = [
            item
            for item in self.opportunities
            if item.addressable and item.risk_class in allowed and item.projected_saving > 0
        ]
        candidates.sort(
            key=lambda item: (
                -item.confidence_interval.low,
                -item.confidence_interval.high,
                order[item.lever],
            )
        )
        return tuple(candidates)

    @property
    def realized(self) -> tuple[Opportunity, ...]:
        """Figures measured from the log rather than projected."""
        return tuple(item for item in self.opportunities if item.measured)

    @property
    def blocked(self) -> tuple[Opportunity, ...]:
        """Levers this log cannot size, in precedence order."""
        return tuple(item for item in self.opportunities if item.status in BLOCKED_STATUSES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "format": self.format_id,
            "currency": self.currency,
            "window_seconds": self.window_seconds,
            "observed_spend": str(self.observed_spend),
            "spend_by_model": [item.to_dict() for item in self.spend_by_model],
            "workload": self.workload.to_dict(),
            "opportunities": [item.to_dict() for item in self.opportunities],
            "overlaps": [item.to_dict() for item in self.overlaps],
            "price_book": self.price_book.to_dict(),
            "reproduction_command": self.reproduction_command,
            "fixes": list(self.fixes),
        }


def sum_money(amounts: Sequence[Decimal]) -> Decimal:
    """Exact sum with an explicit zero, so an empty selection is still a Decimal."""
    total = Decimal(0)
    for amount in amounts:
        total += amount
    return total
