# Owned failures

> A caveat in the documentation is not a substitute for a guard in the code.
> — Masterplan Part 0

This file is the single register of every way BranchPilot can hurt the person running it, and what
we decided to do about each one. It exists so that "we documented it" stops counting as an answer.

`tests/test_safety_net.py` is the machine-readable mirror of the table below. The two are kept in
lockstep by `test_registry_matches_document`, and the promotion gate
(`test_no_serve_lever_has_an_open_failure`) turns "ship a lever with an open footgun" into a red
test instead of a judgement call.

**Changing a row is a code change.** Edit the table here and the `FAILURES` tuple in
`tests/test_safety_net.py` in the same commit, or the harness fails.

Snapshot commit: `2235664`. Line numbers in `location` are relative to that commit.

## 1. The sweep

Scope: `src/`, `docs/`, `examples/`, `README.md`. Case-insensitive.

Primary patterns (the list named by card `S-T01`):

```
operator must | user should | users should | be careful | note that | make sure
at your own risk | we assume | caller must | callers must | it is the responsibility of
```

Extension patterns (responsibility transfer that the primary list misses):

```
must be validated | must be measured | your responsibility | no guarantee | at-least-once
```

Result: six hits, three from each list. Every one of them is a row in the register — five get
their own row, and the sixth shares the Part 3 row it belongs to.

| # | Location | Text | Register row |
|---|---|---|---|
| 1 | `docs/MASTERPLAN.md:20-21` | quotes "the operator must ensure…", "users should be careful to…", "note that this may…" | `masterplan-part0-quotes-disclaimer-phrases` |
| 2 | `docs/MASTERPLAN.md:233-234` | this card's own list of phrases to sweep for | `masterplan-st01-lists-sweep-phrases` |
| 3 | `docs/MASTERPLAN.md:686` | "Operator must set `equivalence_asserted: true`" | `tier-equivalence-asserted-by-operator` |
| 4 | `README.md:317` | "Every selected strategy must be validated for its model… Distribution shift must be measured before deployment." | `strategy-outside-calibrated-envelope` |
| 5 | `README.md:319` | "Live sequential `n=1` request behavior, latency, and infrastructure effects must be measured in the target serving system." | `offline-savings-not-measured-live` |
| 6 | `docs/MASTERPLAN.md:138` | "at-least-once, sorry" — quoted as the industry-normal move | `batch-work-lost-on-restart` |

Hit 6 lands on a Part 3 column-2 cell, which quotes the anti-pattern we are replacing; it shares a
row with the Part 3 failure mode it belongs to rather than getting a duplicate row.

No hit was found anywhere under `src/` or `examples/`. Matches on `unsupported`, `must be finite`,
`must not contain`, and similar are `raise` sites — guards, not disclaimers — and are excluded.

Capability statements ("text-only, non-streaming") are not disclaimers and are tracked by the
Masterplan kill list (`P2-T01`), not here.

## 2. Levers and serve-eligibility

Mirrored by `LEVERS` in `tests/test_safety_net.py`.

| lever | status | meaning |
|---|---|---|
| `adaptive-sampling` | `serve` | shipped and reachable in a configured route today |
| `semantic-cache` | `planned` | not implemented (`M3-T02`) |
| `exact-cache` | `planned` | not implemented (`M3-T01`) |
| `prefix-cache` | `planned` | not implemented (`M2-T02`) |
| `tier-routing` | `planned` | not implemented (`M4-T01`) |
| `batch-lane` | `planned` | not implemented (`M5-T01`) |
| `provider-arbitrage` | `planned` | not implemented (`M4-T02`) |
| `audit-ingest` | `planned` | the offline audit surface (`M1-*`); reads logs, never serves traffic |
| `platform` | `shadow-only` | not a spend lever: the shared control-plane surface every lever routes through |

`platform` is recorded `shadow-only` because it is never promotable to `serve` in its own right. Its
rows are prerequisites for promoting *any* lever, not for enabling a lever of its own.

**Disclosed limit of the gate.** `test_no_serve_lever_has_an_open_failure` is lever-scoped: it fires
when a lever with status `serve` still owns a row with status `planned`. Open `platform` rows —
notably `spend-cap-blown-after-the-fact` and `secret-and-content-leakage-unswept` — apply to the
passthrough path that ships today and are *not* caught by that gate. They are open by record, with
a named closure card, and are the reason `platform` cannot be treated as finished.

## 3. Register

| failure_id | location | risk | decision | closure_card | status |
|---|---|---|---|---|---|
| `sampling-changes-output` | `src/branchpilot/gateway/app.py:439-441` | Stopping early can select a different answer than the full sample budget would have produced, changing what the caller receives. | ACCEPTED-WITH-REASON | - | accepted |
| `strategy-outside-calibrated-envelope` | `README.md:317` | A strategy calibrated on one model, decoding configuration, extractor, or task distribution can be pointed at a different one and degrade answer quality silently. | ACCEPTED-WITH-REASON | - | accepted |
| `offline-savings-not-measured-live` | `README.md:319` | Sample and token counts measured against a pre-generated response bank do not transfer to live latency, throughput, or dollars in the target serving system. | ACCEPTED-WITH-REASON | - | accepted |
| `plan-not-bound-to-model-identity` | `src/branchpilot/gateway/config.py:368-374` | A deployment plan records only its benchmark payload hash, so the config loader cannot refuse a plan that was calibrated for a different model, decoding configuration, or extractor than the route it is attached to. | GUARD | S-T04 | planned |
| `semantic-cache-false-hit` | PLANNED | A near-miss similarity match returns a wrong answer to a real user with no signal that a cache was involved. | REFUSE | M3-T02 | planned |
| `cache-serves-stale-content` | PLANNED | A cached response outlives the truth it encoded and keeps being served after the underlying answer changes. | REFUSE | M3-T01 | planned |
| `cache-poisoning-across-principals` | PLANNED | One principal's response is served to another because the cache key omits the principal. | GUARD | M3-T01 | planned |
| `prefix-cache-silently-inactive` | PLANNED | A volatile element in the system prefix defeats provider prompt caching and the operator never finds out, paying full input price while believing caching is working. | DETECT | M2-T02 | planned |
| `tier-routing-degrades-quality` | PLANNED | Routing traffic to a cheaper model reduces answer quality in a way the headline savings number hides. | DETECT | M4-T01 | planned |
| `tier-equivalence-asserted-by-operator` | `docs/MASTERPLAN.md:686` | Upstreams declared interchangeable may not serve identical weights, so arbitrage silently swaps the model behind a stable alias. | REFUSE | M4-T01 | planned |
| `batch-latency-regression` | PLANNED | A latency-sensitive synchronous request is answered on the batch path and the caller waits hours instead of seconds. | REFUSE | M5-T01 | planned |
| `batch-work-lost-on-restart` | `docs/MASTERPLAN.md:138` | A gateway restart drops in-flight batch work that the provider has already been paid for, with no record of what was lost. | GUARD | M5-T01 | planned |
| `audit-ingest-retains-prompt-text` | PLANNED | Running the audit over production logs copies prompt or completion text into memory, reports, or error messages, turning a free look into a data-handling review. | GUARD | M1-T01 | planned |
| `ambiguous-log-format-misparsed` | PLANNED | An ambiguous log file is best-effort guessed into the wrong format and every downstream figure is quietly wrong. | REFUSE | M1-T01 | planned |
| `response-shape-drift` | PLANNED | A lever path returns a response whose field set differs from the provider's, breaking a client that was working; existing coverage pins only the shipped alias path (`tests/test_gateway_app.py::test_routes_alias_forces_structure_and_returns_sdk_compatible_response`). | GUARD | S-T02 | planned |
| `spend-cap-blown-after-the-fact` | PLANNED | Spend passes its cap and the operator learns about it from a dashboard after the money is gone rather than from a refusal before it is spent. | REFUSE | P2-T05 | planned |
| `credentials-leak-into-middleware-repr` | `src/branchpilot/gateway/config.py:79-91` | An inbound or upstream key reaches a log line, traceback, or nested repr because a config object printed itself. | GUARD | - | closed |
| `secret-and-content-leakage-unswept` | PLANNED | Narrow redaction guards cover the objects we remembered; no adversarial sentinel sweep proves that keys, upstream hosts, model names, prompt text, and completion text are absent from every response, log record, metric label, and error message. | DETECT | S-T03 | planned |
| `config-upgrade-breaks-load` | PLANNED | An upgrade changes the config schema and a working deployment fails to start with an error that does not name the field or the replacement. | REFUSE | S-T04 | planned |
| `savings-number-not-defensible` | PLANNED | A headline savings figure is reported without its counterfactual definition, so it cannot be defended and quality-affecting levers are folded in with response-identical ones. | DETECT | M8-T01 | planned |
| `masterplan-part0-quotes-disclaimer-phrases` | `docs/MASTERPLAN.md:20-21` | The standard itself quotes the disclaimer phrasing it forbids, so the sweep will always hit it. | ACCEPTED-WITH-REASON | - | accepted |
| `masterplan-st01-lists-sweep-phrases` | `docs/MASTERPLAN.md:233-234` | The card that defines the sweep necessarily contains the phrase list the sweep searches for. | ACCEPTED-WITH-REASON | - | accepted |

## 4. Accepted entries, in full

An `ACCEPTED-WITH-REASON` row is a promise that the guard is impossible or unnecessary, not that it
was inconvenient. The short reasons below are the ones stored in the harness.

- **`sampling-changes-output`** — the operator explicitly configures a strategy and a hash-bound
  deployment plan, no implicit activation is possible, and every decision is reported in the
  response decision headers and the flight recorder.
- **`strategy-outside-calibrated-envelope`** — deployment receives no gold labels and no future
  samples, so distributional match to the calibration benchmark is not decidable at request time;
  what is mechanically checkable is enforced (a plan is required per route, its content is bound by
  sha256, and a sample cap above the strategy horizon is refused at config load), and the residual
  binding gap is tracked separately as `plan-not-bound-to-model-identity`.
- **`offline-savings-not-measured-live`** — live latency and infrastructure effects cannot be
  derived from a pre-generated response bank by any amount of code, so instead of estimating them
  the reporting path refuses to relabel samples or completion tokens as latency, GPU-seconds,
  energy, or dollars.
- **`masterplan-part0-quotes-disclaimer-phrases`** — the text is the definition of the standard and
  quotes the forbidden phrasing in order to forbid it; it configures no behavior and has no runtime
  surface, so there is nothing to guard.
- **`masterplan-st01-lists-sweep-phrases`** — the card body is the specification of the sweep
  itself and must contain the phrase list verbatim; it configures no behavior and has no runtime
  surface, so there is nothing to guard.
