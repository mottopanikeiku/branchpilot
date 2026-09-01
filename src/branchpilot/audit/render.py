"""Standalone HTML for one spend audit, plus the summary wording the terminal shares.

Three properties are load-bearing, and each is enforced by a test rather than by care:

*Standalone.* The document references no external resource. There is no ``src``, no
``href``, no CSS ``url()`` and no ``@import`` anywhere in the output, so the file renders
identically on an air-gapped laptop and cannot phone home when an operator opens a report
about their own traffic.

*Deterministic.* The same log and the same price book produce byte-identical HTML. Nothing
here reads the clock, the environment, or a hash seed, and every mapping is iterated in a
sorted order, so two runs can be diffed and a report can be committed next to the log it
describes.

*Defensible.* Every saving is rendered as a range, never a point, and every range carries
the basis it was computed on. Projected figures are labelled
:data:`~branchpilot.audit.detectors.PROJECTION_BASIS` -- arithmetic over the tokens the log
recorded and the prices that were configured, not a measurement of money anybody saved. The
one exception is realized prefix-cache behaviour, which the provider already applied and the
log already recorded; it is labelled
:data:`~branchpilot.audit.detectors.MEASURED_BASIS` and is excluded from the addressable
total so that money already captured is never counted as money still available.

The headline obeys the same scoping rule as the terminal: only ``IDENTICAL`` levers are in
it by default. ``QUALITY_AFFECTING`` levers get their own section carrying the exact command
that reports their figures, and they join the headline only when the caller passes
``include_quality_affecting=True``, which also renders the warning block.
"""

from __future__ import annotations

import html
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import TYPE_CHECKING

from branchpilot.audit.detectors import BOUND_RULE, MEASURED_BASIS, PROJECTION_BASIS
from branchpilot.audit.risk import (
    IDENTICAL,
    QUALITY_AFFECTING,
    RISK_CLASSES,
    VALIDATION_COMMANDS,
    VALIDATION_REQUIREMENTS,
    headline,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from collections.abc import Iterable, Sequence

    from branchpilot.audit.result import AuditResult, ModelSpend, Opportunity, Range

__all__ = [
    "LIMITATIONS",
    "NO_OPPORTUNITY",
    "TOP_OPPORTUNITIES",
    "format_amount",
    "format_money",
    "format_range",
    "next_action",
    "render_html",
    "scoped_levers",
]

TOP_OPPORTUNITIES = 3
"""How many opportunities the headline ranks. Three is what an operator can act on."""

NO_OPPORTUNITY = (
    "no material opportunity found: no lever in scope substantiated a saving from this log"
)
"""Rendered instead of a ranking when nothing in scope carries money. Never a blank page."""

LIMITATIONS = (
    "Every saving here is a projection from the tokens this log recorded times the prices "
    "that were configured. It is not a measurement of money anybody saved, and it is not an "
    "invoice reconciliation.",
    BOUND_RULE + ", so the two bounds are the same arithmetic over two request sets, not a "
    "statistical confidence interval",
    "Records whose (provider, model) pair has no configured price are excluded from every "
    "figure on this page. The coverage section states how many.",
    "Prefix-cache savings are never estimated. Where the log carries cached prompt token "
    "counts the report states what the provider's cache already saved; where it does not, no "
    "figure is projected, because the prefix token count is not in the record and inferring "
    "it from the prompt length would be a guess dressed as arithmetic.",
    "QUALITY_AFFECTING levers can change the responses your callers receive. They are "
    "excluded from the headline unless the report was rendered with "
    "--include-quality-affecting, and no figure here substitutes for validating them on your "
    "own traffic.",
    "The audit reads token counts, hashes and status codes. No prompt or completion text is "
    "read, stored, or rendered, so nothing on this page can leak content.",
    "Configured rate cards are list prices. Committed-use discounts, negotiated rates and "
    "free tiers are not modelled; pass --price-book FILE with your own rates to reflect them.",
)
"""What this report cannot tell you. Rendered verbatim, never summarized away."""

_MONEY_PRECISION = 60
_MONEY_PLACES = Decimal("0.000001")


def format_amount(amount: Decimal) -> str:
    """One money amount, fixed to six places so two runs render identical characters."""
    with localcontext() as ctx:
        ctx.prec = _MONEY_PRECISION
        quantized = amount.quantize(_MONEY_PLACES, rounding=ROUND_HALF_UP)
    return f"{quantized:f}"


def format_money(amount: Decimal, currency: str) -> str:
    return f"{format_amount(amount)} {currency}"


def format_range(interval: Range, currency: str) -> str:
    """A range, always written as two bounds. A bare point is never rendered for a saving."""
    return f"{format_amount(interval.low)} to {format_amount(interval.high)} {currency}"


def scoped_levers(*, include_quality_affecting: bool) -> tuple[str, ...]:
    """The risk classes the headline is allowed to sum under this scope."""
    return RISK_CLASSES if include_quality_affecting else (IDENTICAL,)


def next_action(result: AuditResult, *, include_quality_affecting: bool = False) -> str:
    """The one thing to do next. Exactly one sentence, chosen by what the log can support."""
    workload = result.workload
    if workload.records and workload.priced_records == 0:
        pairs = ", ".join(f"{provider}/{model}" for provider, model in workload.unpriced_pairs)
        return (
            f"price this workload before acting on it: none of {workload.records} record(s) "
            f"matched a configured rate card ({pairs}); fix: add those (provider, model) pairs "
            "to a price book file and re-run with --price-book FILE"
        )
    ranked = result.ranked(include_quality_affecting=include_quality_affecting)
    if ranked:
        top = ranked[0]
        interval = format_range(top.confidence_interval, result.currency)
        return (
            f"adopt {top.lever}: {top.eligible_requests} request(s) in this log are eligible and "
            f"the projected saving is {interval}; first step: {top.required_changes[0]}"
        )
    allowed = scoped_levers(include_quality_affecting=include_quality_affecting)
    blocked = [item for item in result.blocked if item.risk_class in allowed]
    if blocked:
        first = blocked[0]
        return (
            f"unblock {first.lever}, which this log cannot size ({first.status}); first step: "
            f"{first.required_changes[0]}"
        )
    return NO_OPPORTUNITY


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _basis(opportunity: Opportunity) -> str:
    return MEASURED_BASIS if opportunity.measured else PROJECTION_BASIS


def _section(identifier: str, title: str, body: str) -> str:
    return f'<section id="{identifier}"><h2>{_e(title)}</h2>{body}</section>'


def _paragraph(text: str) -> str:
    return f"<p>{_e(text)}</p>"


def _list(items: Iterable[str]) -> str:
    rows = "".join(f"<li>{_e(item)}</li>" for item in items)
    return f"<ul>{rows}</ul>" if rows else ""


def _range_cell(interval: Range | None, currency: str) -> str:
    if interval is None:
        return '<td class="figure none">not sized from this log</td>'
    return f'<td class="figure range">{_e(format_range(interval, currency))}</td>'


def _opportunity_rows(opportunities: Sequence[Opportunity], currency: str, *, ranked: bool) -> str:
    rows: list[str] = []
    for position, item in enumerate(opportunities, start=1):
        rank = f'<th scope="row">{position}</th>' if ranked else ""
        rows.append(
            "<tr>"
            + rank
            + f'<td class="lever">{_e(item.lever)}</td>'
            + f'<td class="risk risk-{_e(item.risk_class)}">{_e(item.risk_class)}</td>'
            + f'<td class="status">{_e(item.status)}</td>'
            + f'<td class="numeric">{item.eligible_requests}</td>'
            + f'<td class="numeric">{_e(format_money(item.eligible_spend, currency))}</td>'
            + _range_cell(item.confidence_interval, currency)
            + f'<td class="basis">{_e(_basis(item))}</td>'
            + "</tr>"
        )
    header = (
        "<tr>"
        + ('<th scope="col">#</th>' if ranked else "")
        + '<th scope="col">lever</th><th scope="col">risk class</th>'
        + '<th scope="col">status</th><th scope="col">eligible requests</th>'
        + '<th scope="col">eligible spend</th><th scope="col">range</th>'
        + '<th scope="col">basis</th>'
        + "</tr>"
    )
    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def _evidence(opportunities: Sequence[Opportunity]) -> str:
    blocks: list[str] = []
    for item in opportunities:
        assumptions = _list(item.assumptions)
        changes = _list(item.required_changes)
        blocks.append(
            '<article class="evidence">'
            f"<h3>{_e(item.lever)}</h3>"
            f'<p class="basis">{_e(_basis(item))}</p>'
            + (f"<h4>Assumptions</h4>{assumptions}" if assumptions else "")
            + (f"<h4>Required changes</h4>{changes}" if changes else "")
            + "</article>"
        )
    return "".join(blocks)


def _spend_rows(rows: Sequence[ModelSpend], currency: str) -> str:
    body = "".join(
        "<tr>"
        f'<th scope="row">{_e(row.provider)}/{_e(row.model)}</th>'
        f'<td class="numeric">{row.requests}</td>'
        f'<td class="numeric">{row.prompt_tokens}</td>'
        f'<td class="numeric">{row.cached_prompt_tokens}</td>'
        f'<td class="numeric">{row.completion_tokens}</td>'
        f'<td class="numeric amount">{_e(format_money(row.spend, currency))}</td>'
        "</tr>"
        for row in rows
    )
    header = (
        '<tr><th scope="col">provider/model</th><th scope="col">requests</th>'
        '<th scope="col">prompt tokens</th><th scope="col">cached prompt tokens</th>'
        '<th scope="col">completion tokens</th><th scope="col">observed spend</th></tr>'
    )
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>"


def _headline_section(result: AuditResult, *, include_quality_affecting: bool) -> str:
    summary = headline(result.opportunities, include_quality_affecting=include_quality_affecting)
    classes = ", ".join(scoped_levers(include_quality_affecting=include_quality_affecting))
    figure = (
        f'<p class="figure range headline-range">'
        f"{_e(format_amount(summary.low))} to {_e(format_amount(summary.high))} "
        f"{_e(result.currency)}</p>"
    )
    scope = _paragraph(f"Risk classes in scope: {classes}.")
    basis = _paragraph(PROJECTION_BASIS)
    levers = (
        _paragraph("Levers summed: " + ", ".join(summary.levers))
        if summary.levers
        else _paragraph(NO_OPPORTUNITY)
    )
    warning = f'<p class="warning">{_e(summary.warning)}</p>' if summary.warning is not None else ""
    return _section(
        "total-addressable-range",
        "Total addressable range",
        figure + scope + basis + levers + warning,
    )


def _top_section(result: AuditResult, *, include_quality_affecting: bool) -> str:
    ranked = result.ranked(include_quality_affecting=include_quality_affecting)[:TOP_OPPORTUNITIES]
    if not ranked:
        body = _paragraph(NO_OPPORTUNITY)
    else:
        body = _opportunity_rows(ranked, result.currency, ranked=True) + _evidence(ranked)
    return _section("top-opportunities", "Top opportunities", body)


def _quality_section(result: AuditResult, *, include_quality_affecting: bool) -> str:
    quarantined = [item for item in result.opportunities if item.risk_class == QUALITY_AFFECTING]
    intro = _paragraph(
        "These levers can change the responses your callers receive. They are "
        + (
            "included in the headline above because this report was rendered with "
            "--include-quality-affecting."
            if include_quality_affecting
            else "excluded from the headline above."
        )
    )
    blocks: list[str] = []
    for item in quarantined:
        command = VALIDATION_COMMANDS[item.lever]
        requirement = VALIDATION_REQUIREMENTS[item.lever]
        interval = (
            format_range(item.confidence_interval, result.currency)
            if item.confidence_interval is not None
            else "not sized from this log"
        )
        blocks.append(
            '<article class="quality-affecting">'
            f"<h3>{_e(item.lever)}</h3>"
            f'<p class="figure">{_e(interval)}</p>'
            f'<p class="basis">{_e(_basis(item))}</p>'
            f"<p>{_e(requirement)}</p>"
            "<p>Validate with this exact command:</p>"
            f"<pre><code>{_e(command)}</code></pre>"
            "</article>"
        )
    return _section("quality-affecting", "Quality-affecting levers", intro + "".join(blocks))


def _realized_section(result: AuditResult) -> str:
    realized = result.realized
    if not realized:
        body = _paragraph(
            "This log records no prefix-cache reads, so there is no realized cache behaviour "
            "to measure."
        )
    else:
        body = _opportunity_rows(realized, result.currency, ranked=False) + _evidence(realized)
    return _section("realized-cache-behaviour", "Realized cache behaviour", body)


def _blocked_section(result: AuditResult) -> str:
    blocked = result.blocked
    if not blocked:
        body = _paragraph("Every lever could be sized from this log.")
    else:
        body = _opportunity_rows(blocked, result.currency, ranked=False) + _evidence(blocked)
    return _section("blocked-levers", "Levers this log cannot size", body)


def _overlap_section(result: AuditResult) -> str:
    if not result.overlaps:
        body = _paragraph(
            "No request was eligible for two levers at once, so the ranked figures are additive."
        )
    else:
        body = _list(overlap.message() for overlap in result.overlaps)
    return _section("overlaps", "Overlaps between levers", body)


def _coverage_section(result: AuditResult) -> str:
    workload = result.workload
    rows = [
        ("Records read", str(workload.records)),
        ("Priced records", str(workload.priced_records)),
        ("Unpriced records", str(workload.unpriced_records)),
        ("Coverage", f"{workload.coverage_percent}%"),
        ("Records with status ok", str(workload.ok_records)),
        ("Records with cached token counts", str(workload.records_with_cached_counts)),
        ("Lines parsed", str(workload.parsed)),
        ("Lines skipped", str(workload.skipped)),
    ]
    if workload.unpriced_pairs:
        rows.append(
            (
                "Unpriced pairs",
                ", ".join(f"{provider}/{model}" for provider, model in workload.unpriced_pairs),
            )
        )
    for reason, count in sorted(workload.skip_reasons.items()):
        rows.append((f"Skipped: {reason}", str(count)))
    body = (
        "<dl>"
        + "".join(f"<div><dt>{_e(label)}</dt><dd>{_e(value)}</dd></div>" for label, value in rows)
        + "</dl>"
    )
    body += _list(result.fixes)
    return _section("coverage", "Priced and unpriced coverage", body)


def _observed_section(result: AuditResult) -> str:
    body = (
        f'<p class="figure amount">{_e(format_money(result.observed_spend, result.currency))}</p>'
        + _paragraph(PROJECTION_BASIS)
        + _paragraph(
            f"Summed over the {result.workload.priced_records} record(s) this price book could "
            "price. Unpriced records contribute nothing."
        )
    )
    return _section("observed-spend", "Observed spend", body)


def _provenance_section(result: AuditResult) -> str:
    book = result.price_book
    window = (
        "every record in the file"
        if result.window_seconds is None
        else (f"{result.window_seconds} second(s)")
    )
    rows = [
        ("Log", result.source),
        ("Log SHA-256", result.source_sha256),
        ("Log format", result.format_id),
        ("Records read", str(result.workload.records)),
        ("Records priced", str(result.workload.priced_records)),
        ("Dedup window", window),
        ("Currency", result.currency),
        ("Price book", book.path),
        ("Price book schema version", str(book.schema_version)),
        ("Price book effective dates", ", ".join(book.effective_dates) or "none"),
    ]
    body = (
        "<dl>"
        + "".join(f"<div><dt>{_e(label)}</dt><dd>{_e(value)}</dd></div>" for label, value in rows)
        + "</dl>"
    )
    cards = _list(f"{provider}/{model} effective {day}" for provider, model, day in book.entries)
    if cards:
        body += "<h3>Rate cards that priced this log</h3>" + cards
    body += "<h3>Reproduce this report</h3>"
    body += f"<pre><code>{_e(result.reproduction_command)}</code></pre>"
    return _section("provenance", "Provenance", body)


def _next_action_section(result: AuditResult, *, include_quality_affecting: bool) -> str:
    action = next_action(result, include_quality_affecting=include_quality_affecting)
    return _section(
        "recommended-next-action",
        "Recommended next action",
        f'<p class="action">{_e(action)}</p>',
    )


_STYLES = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2rem 1.5rem 4rem;
  font-family: ui-sans-serif, system-ui, sans-serif;
  line-height: 1.55;
  color: #10151c;
  background: #f6f7f9;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.1rem; margin: 0 0 0.75rem; letter-spacing: 0.02em; }
h3 { font-size: 0.95rem; margin: 1rem 0 0.35rem; }
h4 { font-size: 0.8rem; margin: 0.75rem 0 0.25rem; text-transform: uppercase; }
p { margin: 0 0 0.6rem; }
section {
  background: #ffffff;
  border: 1px solid #d8dde4;
  border-radius: 8px;
  padding: 1.1rem 1.25rem;
  margin: 0 0 1rem;
}
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #e4e8ee; }
td.numeric, th.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.figure { font-variant-numeric: tabular-nums; font-weight: 600; }
.headline-range { font-size: 1.75rem; margin: 0 0 0.5rem; }
.basis { font-size: 0.8rem; color: #4a5462; }
.risk-QUALITY_AFFECTING { color: #8a4b00; font-weight: 600; }
.risk-IDENTICAL { color: #1d5c33; font-weight: 600; }
.warning {
  border-left: 4px solid #b4610a;
  background: #fdf3e6;
  padding: 0.6rem 0.8rem;
  margin: 0.75rem 0 0;
}
.action { font-weight: 600; }
pre {
  background: #10151c;
  color: #eef2f7;
  padding: 0.7rem 0.85rem;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.8rem;
}
code { font-family: ui-monospace, monospace; }
dl { display: grid; grid-template-columns: 1fr; gap: 0.2rem; margin: 0; }
dl div { display: flex; gap: 0.75rem; justify-content: space-between; }
dt { color: #4a5462; font-size: 0.85rem; }
dd { margin: 0; font-size: 0.85rem; word-break: break-all; }
ul { margin: 0 0 0.5rem; padding-left: 1.1rem; font-size: 0.85rem; }
li { margin: 0 0 0.25rem; }
"""


def render_html(result: AuditResult, *, include_quality_affecting: bool = False) -> str:
    """Render one audit as a standalone, deterministic HTML document.

    The output references no external resource and depends on nothing but ``result`` and
    the scope flag, so two runs over the same log and price book are byte-identical.
    """
    if not isinstance(include_quality_affecting, bool):
        raise TypeError(
            "include_quality_affecting must be True or False; "
            "fix: pass the parsed --include-quality-affecting flag"
        )
    title = f"BranchPilot spend audit: {result.source}"
    body = "".join(
        (
            _headline_section(result, include_quality_affecting=include_quality_affecting),
            _top_section(result, include_quality_affecting=include_quality_affecting),
            _next_action_section(result, include_quality_affecting=include_quality_affecting),
            _observed_section(result),
            _section(
                "spend-by-model",
                "Spend by model",
                _spend_rows(result.spend_by_model, result.currency)
                if result.spend_by_model
                else _paragraph("No record in this log could be priced."),
            ),
            _coverage_section(result),
            _realized_section(result),
            _quality_section(result, include_quality_affecting=include_quality_affecting),
            _blocked_section(result),
            _overlap_section(result),
            _provenance_section(result),
            _section("limitations", "Limitations", _list(LIMITATIONS)),
        )
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(title)}</title>\n"
        f"<style>{_STYLES}</style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"<h1>{_e(title)}</h1>\n"
        f"<p>{_e(f'Log SHA-256 {result.source_sha256}')}</p>\n"
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )
