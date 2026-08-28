"""Registry-driven enforcement of the safety net in Masterplan Part 3.

The standard is Part 0: *a caveat in the documentation is not a substitute for a guard in the
code*. This module is the machine-readable half of that standard.

``FAILURES`` mirrors the register table in ``docs/internal/owned-failures.md`` row for row, and
``test_registry_matches_document`` keeps the two from drifting. ``LEVERS`` records, per lever,
whether it is reachable in a configured route today. ``test_no_serve_lever_has_an_open_failure``
is the gate: promoting a lever to ``serve`` while one of its failure rows is still ``planned``
becomes a red test rather than a judgement call.

Two ``LEVERS`` keys are not spend levers:

``audit-ingest``
    the offline audit surface; it reads logs and never serves traffic.
``platform``
    the shared control-plane surface every lever routes through. It is recorded ``shadow-only``
    because it is never promotable to ``serve`` in its own right; its rows are prerequisites for
    promoting *any* lever. The gate is lever-scoped and therefore does not fire on open
    ``platform`` rows -- that limit is disclosed in the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

DOCUMENT = Path(__file__).resolve().parents[1] / "docs" / "internal" / "owned-failures.md"

DECISIONS = frozenset({"GUARD", "DETECT", "REFUSE", "ACCEPTED-WITH-REASON"})
ROW_STATUSES = frozenset({"closed", "planned", "accepted"})
LEVER_STATUSES = frozenset({"serve", "planned", "shadow-only"})
MIN_REASON_CHARS = 40
_REGISTER_COLUMNS = 6
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class Failure:
    """One owned failure mode. Shape is validated here; policy is validated by the tests."""

    failure_id: str
    lever: str
    decision: str
    status: str
    guard_test: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(self.failure_id):
            raise ValueError(
                f"failure_id {self.failure_id!r} is not a stable slug; "
                "fix: use lowercase words joined by single hyphens, e.g. 'semantic-cache-false-hit'"
            )
        if not _SLUG.fullmatch(self.lever):
            raise ValueError(
                f"failure {self.failure_id!r} has lever {self.lever!r}; "
                "fix: set lever to a lowercase-slug lever id declared in LEVERS"
            )
        if self.decision not in DECISIONS:
            raise ValueError(
                f"failure {self.failure_id!r} has decision {self.decision!r}; "
                f"fix: set decision to one of: {', '.join(sorted(DECISIONS))}"
            )
        if self.status not in ROW_STATUSES:
            raise ValueError(
                f"failure {self.failure_id!r} has status {self.status!r}; "
                f"fix: set status to one of: {', '.join(sorted(ROW_STATUSES))}"
            )


# Lever id -> serve-eligibility. `serve` means shipped and reachable in a configured route.
LEVERS = MappingProxyType(
    {
        "adaptive-sampling": "serve",
        "semantic-cache": "planned",
        "exact-cache": "planned",
        "prefix-cache": "planned",
        "tier-routing": "planned",
        "batch-lane": "planned",
        "provider-arbitrage": "planned",
        "audit-ingest": "planned",
        "platform": "shadow-only",
    }
)

FAILURES = (
    Failure(
        failure_id="sampling-changes-output",
        lever="adaptive-sampling",
        decision="ACCEPTED-WITH-REASON",
        status="accepted",
        guard_test=(
            "tests/test_gateway_app.py::test_exact_sequential_early_stop_and_aggregate"
            "_repeated_usage"
        ),
        reason=(
            "the operator explicitly configures a strategy and a hash-bound deployment plan, no "
            "implicit activation is possible, and every decision is reported in the response "
            "decision headers and the flight recorder"
        ),
    ),
    Failure(
        failure_id="strategy-outside-calibrated-envelope",
        lever="adaptive-sampling",
        decision="ACCEPTED-WITH-REASON",
        status="accepted",
        guard_test="tests/test_deployment.py::test_rejects_hash_mismatch_and_symbolic_link",
        reason=(
            "deployment receives no gold labels and no future samples, so distributional match to "
            "the calibration benchmark is not decidable at request time; what is mechanically "
            "checkable is enforced (a plan is required per route, its content is bound by sha256, "
            "and a sample cap above the strategy horizon is refused at config load), and the "
            "residual binding gap is tracked as plan-not-bound-to-model-identity"
        ),
    ),
    Failure(
        failure_id="offline-savings-not-measured-live",
        lever="adaptive-sampling",
        decision="ACCEPTED-WITH-REASON",
        status="accepted",
        reason=(
            "live latency and infrastructure effects cannot be derived from a pre-generated "
            "response bank by any amount of code, so instead of estimating them the reporting "
            "path refuses to relabel samples or completion tokens as latency, GPU-seconds, "
            "energy, or dollars"
        ),
    ),
    Failure(
        failure_id="plan-not-bound-to-model-identity",
        lever="platform",
        decision="GUARD",
        status="planned",
    ),
    Failure(
        failure_id="semantic-cache-false-hit",
        lever="semantic-cache",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="cache-serves-stale-content",
        lever="exact-cache",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="cache-poisoning-across-principals",
        lever="exact-cache",
        decision="GUARD",
        status="planned",
    ),
    Failure(
        failure_id="prefix-cache-silently-inactive",
        lever="prefix-cache",
        decision="DETECT",
        status="planned",
    ),
    Failure(
        failure_id="tier-routing-degrades-quality",
        lever="tier-routing",
        decision="DETECT",
        status="planned",
    ),
    Failure(
        failure_id="tier-equivalence-asserted-by-operator",
        lever="provider-arbitrage",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="batch-latency-regression",
        lever="batch-lane",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="batch-work-lost-on-restart",
        lever="batch-lane",
        decision="GUARD",
        status="planned",
    ),
    Failure(
        failure_id="audit-ingest-retains-prompt-text",
        lever="audit-ingest",
        decision="GUARD",
        status="planned",
    ),
    Failure(
        failure_id="ambiguous-log-format-misparsed",
        lever="audit-ingest",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="response-shape-drift",
        lever="platform",
        decision="GUARD",
        status="planned",
    ),
    Failure(
        failure_id="spend-cap-blown-after-the-fact",
        lever="platform",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="credentials-leak-into-middleware-repr",
        lever="platform",
        decision="GUARD",
        status="closed",
        guard_test=(
            "tests/test_gateway_app.py::test_unauthenticated_request_reads_zero_body_messages"
            "_and_redacts_middleware"
        ),
    ),
    Failure(
        failure_id="secret-and-content-leakage-unswept",
        lever="platform",
        decision="DETECT",
        status="planned",
    ),
    Failure(
        failure_id="config-upgrade-breaks-load",
        lever="platform",
        decision="REFUSE",
        status="planned",
    ),
    Failure(
        failure_id="savings-number-not-defensible",
        lever="platform",
        decision="DETECT",
        status="planned",
    ),
    Failure(
        failure_id="masterplan-part0-quotes-disclaimer-phrases",
        lever="platform",
        decision="ACCEPTED-WITH-REASON",
        status="accepted",
        reason=(
            "the text is the definition of the standard and quotes the forbidden phrasing in "
            "order to forbid it; it configures no behavior and has no runtime surface, so there "
            "is nothing to guard"
        ),
    ),
    Failure(
        failure_id="masterplan-st01-lists-sweep-phrases",
        lever="platform",
        decision="ACCEPTED-WITH-REASON",
        status="accepted",
        reason=(
            "the card body is the specification of the sweep itself and must contain the phrase "
            "list verbatim; it configures no behavior and has no runtime surface, so there is "
            "nothing to guard"
        ),
    ),
)


def _register_rows() -> dict[str, tuple[str, str]]:
    """Parse the register table of the document into ``failure_id -> (decision, status)``."""
    if not DOCUMENT.is_file():
        raise AssertionError(
            f"the owned-failure register is missing at {DOCUMENT}; "
            "fix: restore docs/internal/owned-failures.md, it is the source this harness mirrors"
        )
    rows: dict[str, tuple[str, str]] = {}
    inside = False
    for number, line in enumerate(DOCUMENT.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            inside = False
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0] == "failure_id":
            inside = True
            if len(cells) != _REGISTER_COLUMNS:
                raise AssertionError(
                    f"{DOCUMENT.name}:{number} register header has {len(cells)} columns; "
                    "fix: use exactly failure_id | location | risk | decision | closure_card | "
                    "status"
                )
            continue
        if not inside:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) != _REGISTER_COLUMNS:
            raise AssertionError(
                f"{DOCUMENT.name}:{number} register row has {len(cells)} columns, expected "
                f"{_REGISTER_COLUMNS}; fix: a '|' inside a cell splits the row, escape it as '\\|'"
            )
        failure_id = cells[0].strip("`")
        if failure_id in rows:
            raise AssertionError(
                f"{DOCUMENT.name}:{number} repeats failure_id {failure_id!r}; "
                "fix: merge the duplicate rows, one failure mode gets exactly one row"
            )
        rows[failure_id] = (cells[3], cells[5])
    if not rows:
        raise AssertionError(
            f"no register rows parsed from {DOCUMENT.name}; fix: keep the register as a markdown "
            "table whose header row begins with the cell 'failure_id'"
        )
    return rows


def test_no_serve_lever_has_an_open_failure() -> None:
    """The gate: a lever reachable in a configured route owns no unclosed failure row."""
    unknown = sorted({row.lever for row in FAILURES} - set(LEVERS))
    assert not unknown, (
        f"failure rows name unknown lever(s) {unknown}; "
        f"fix: add them to LEVERS with a serve-eligibility of {sorted(LEVER_STATUSES)}"
    )
    open_rows = sorted(
        row.failure_id
        for row in FAILURES
        if LEVERS[row.lever] == "serve" and row.status == "planned"
    )
    assert not open_rows, (
        f"lever(s) marked 'serve' still own open failure row(s) {open_rows}; "
        "fix: either close the row in code and set its status to 'closed' with a guard_test, or "
        "demote the lever in LEVERS to 'shadow-only' until the guard lands"
    )


def test_accepted_failures_carry_a_reason() -> None:
    for row in FAILURES:
        if row.status != "accepted":
            continue
        assert row.decision == "ACCEPTED-WITH-REASON", (
            f"failure {row.failure_id!r} has status 'accepted' but decision {row.decision!r}; "
            "fix: set decision to ACCEPTED-WITH-REASON or change the status"
        )
        reason = row.reason.strip()
        assert len(reason) >= MIN_REASON_CHARS, (
            f"failure {row.failure_id!r} is accepted with a {len(reason)}-character reason; "
            f"fix: write at least {MIN_REASON_CHARS} characters saying why the guard is "
            "impossible or unnecessary"
        )


def test_closed_failures_name_a_guard_test() -> None:
    for row in FAILURES:
        if row.status != "closed":
            continue
        assert row.guard_test.strip(), (
            f"failure {row.failure_id!r} is closed but names no guard_test; "
            "fix: set guard_test to the test that fails if the guard regresses, as "
            "'tests/test_x.py::test_y'"
        )


def test_registry_matches_document() -> None:
    documented = _register_rows()
    registered = {row.failure_id: (row.decision, row.status) for row in FAILURES}

    missing = sorted(set(registered) - set(documented))
    assert not missing, (
        f"FAILURES rows absent from {DOCUMENT.name}: {missing}; "
        "fix: add one register row per failure in the same commit"
    )
    extra = sorted(set(documented) - set(registered))
    assert not extra, (
        f"{DOCUMENT.name} documents failures absent from FAILURES: {extra}; "
        "fix: add the matching Failure(...) row to FAILURES in the same commit"
    )
    drifted = sorted(
        f"{failure_id}: document {documented[failure_id]} != registry {registered[failure_id]}"
        for failure_id in registered
        if documented[failure_id] != registered[failure_id]
    )
    assert not drifted, (
        f"decision or status drift between {DOCUMENT.name} and FAILURES: {drifted}; "
        "fix: change both in the same commit"
    )


def test_lever_statuses_use_the_declared_vocabulary() -> None:
    invalid = sorted(
        f"{lever}={status}" for lever, status in LEVERS.items() if status not in LEVER_STATUSES
    )
    assert not invalid, (
        f"LEVERS carries undeclared serve-eligibility {invalid}; "
        f"fix: use one of {sorted(LEVER_STATUSES)}"
    )


def test_every_lever_owns_at_least_one_failure_row() -> None:
    covered = {row.lever for row in FAILURES}
    uncovered = sorted(set(LEVERS) - covered)
    assert not uncovered, (
        f"lever(s) {uncovered} have no failure row; "
        "fix: add the lever's Part 3 failure mode to FAILURES and to the register, or remove the "
        "lever from LEVERS"
    )


def test_failure_ids_are_unique() -> None:
    seen: dict[str, int] = {}
    for row in FAILURES:
        seen[row.failure_id] = seen.get(row.failure_id, 0) + 1
    duplicated = sorted(failure_id for failure_id, count in seen.items() if count > 1)
    assert not duplicated, (
        f"duplicate failure_id(s) in FAILURES: {duplicated}; "
        "fix: one failure mode gets exactly one row"
    )
