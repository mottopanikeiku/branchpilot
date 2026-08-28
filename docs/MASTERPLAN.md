# BranchPilot Masterplan v3

Supersedes v1 (credibility-first) and v2 (revenue-first). Both were wrong targets.

**The target: the most usable thing in its category, used by default, with every dangerous edge
owned by us rather than disclaimed to the user.**

No pricing. No tiers. No accounts. No hosted service. No telemetry by default.
Success = people reach for it without thinking, and it never burns them.

Repository facts verified against commit `5eedf26`. Market figures sourced 2026-08-27, cited
inline, `[VERIFY]` where they must be re-checked before external use.

---

## Part 0 — The one-sentence standard

> A caveat in the documentation is not a substitute for a guard in the code.

Every time this project would write "the operator must ensure…", "users should be careful to…",
or "note that this may…", the correct move is to make the system enforce it, refuse it, or
detect and report it. Where that is genuinely impossible, the exception gets written down with a
reason. That is what "owning the safety net" means here, and it is the plan's spine.

Three consequences, applied to every card in Part 7:

1. **A lever does not ship until its failure mode is closed by construction.** Not documented — closed.
2. **Defaults are the safe path.** A user who reads nothing and configures nothing gets a correct,
   conservative system. Knobs exist for experts; they are never required for value.
3. **Errors carry the fix.** Every failure names the file, the field, and the exact change.

---

## Part 1 — The brutal read

### 1.1 What we actually have

A hardened, credential-safe, OpenAI-compatible proxy with a per-request decision loop, a
hash-bound plan/artifact system, and a deterministic evidence renderer. The proxy sits **in the
request path of LLM traffic** — the highest-leverage position in the stack, and the one place
where being unusable or unsafe is fatal.

### 1.2 What we built, versus what matters

We shipped *adaptive sequential sampling* — decide whether to draw another sample. Ranked against
every other lever for reducing LLM compute, it is **last**:

| # | Lever | Reported impact | Traffic it applies to |
|---|---|---|---|
| 1 | Prefix/prompt cache exploitation | cached reads bill at ~10% of standard input on current flagships (90% off); OpenAI auto-caching discount is provider-published | almost any workload with a stable system prompt or shared context |
| 2 | Batch/async lane | flat 50% off input **and** output on both OpenAI and Anthropic; stacks with caching to ~95% off the repeated portion | anything tolerating ≤24h latency |
| 3 | Response cache (exact + semantic) | production semantic hit rates **20–45%** (not the 90% marketing figure); reported reduction 20–73% | repetitive Q&A, support, RAG |
| 4 | Model tier routing | **no sourced production range exists.** RouteLLM (primary, published) reports ">2x cost reduction in certain cases"; the widely repeated "85% reduction / 95% of GPT-4 performance" is benchmark-specific and must not be cited. Magnitude depends entirely on the easy/hard mix in your traffic and must be measured | any traffic with easy/hard variance |
| 5 | Provider arbitrage on open weights | large price dispersion for identical weights | open-weight traffic |
| 6 | **Adaptive sampling (our headline)** | a fraction of a *multi-sample* budget | only workloads already drawing k>1 |

All figures are **market-reported, not our claims.** Every one has a sourced row with an access
date and a VERIFY-BY date in `docs/internal/market-notes.md`. Anything absent from that file may
not be cited anywhere. Figures previously carried in this plan that did NOT survive verification
are recorded in its "Unverified / omitted" section — including a widely circulated
$720/mo → $72/mo prompt-caching anecdote whose source returned HTTP 403 and is therefore dropped.

**Conclusion: we built lever 6, marketed it as the product, and required a GPU plus pre-generated
trajectories to see any of it.** Levers 1–4 are larger, apply to vastly more traffic, need no
research risk, and all sit behind the proxy we already hardened.

### 1.3 Where usability stands today

| Friction | Current reality |
|---|---|
| First value requires | `uv sync --extra train`, synthetic data generation, CPU training, or a running vLLM |
| Decisions before value | choose a strategy, a cost λ, a sample budget, an extractor, a plan family |
| Works on my traffic? | unanswerable without generating trajectories |
| Streaming | rejected outright (`stream: Literal[False]`) — disqualifying for most chat apps |
| Providers | OpenAI-shaped only |
| Answer formats | numeric or exact-match only |
| Visual surface | none |
| Container | none |

Every row is a reason someone closes the tab.

### 1.4 Category context

| Player | Stars (2026-08-28) | Position | Exploitable gap |
|---|---|---|---|
| **LiteLLM** | **57,446** | the default OSS gateway. MIT except a separately licensed `enterprise/`; 100+ providers; ~1,645 contributor pages; shipped a release the day we measured; semantic cache across Qdrant/Redis/Valkey; working virtual keys and per-key/team budgets | tells you what you **spent**, never what you **would have** spent. And its own cost-tracking docs *warn* that end users can self-declare `user` to dodge spend attribution — a footgun disclaimed to the operator rather than closed in code. That single line is our whole thesis, written by the incumbent |
| **Portkey Gateway** | **12,842** | **MIT, not Apache-2.0** — the relicense claim in earlier drafts was contradicted by the source. Broadest catalogue (1,600+ LLMs claimed); real semantic-cache engineering (Milvus/Pinecone, configurable threshold, namespaces); guardrails we lack entirely | OSS repo is **stale**: last push 2026-05-25, last release 2026-01-12. Semantic caching is Enterprise-gated on the hosted product |
| **Helicone** | **6,106** | **maintenance mode confirmed** since the Mintlify acquisition on 2026-03-03 — security, new models, and bug fixes only. Proved the market: 14.2T tokens and ~16,000 organisations in three years off a one-line integration | the actively-maintained OSS-observability slot is genuinely open, and that adoption curve came from one-line integration — which is our U2 |
| **Kong AI Gateway** | **44,052** (whole gateway) | Apache-2.0 core, AI plugins tiered; semantic routing and semantic cache | semantic cache requires `ai_gateway_enterprise`; not developer-first |
| **Cloudflare AI Gateway** | n/a (closed) | edge incumbent, real analytics | **exact-match cache only**, semantic search documented as planned; not self-hostable |
| **OpenRouter** | n/a (closed) | 103 providers / 380 models; sticky routing to keep upstream prompt caches warm | not self-hostable; no counterfactual |

**Counterfactual savings measurement: none of the six.** That is the wedge, verified rather than assumed.

Reality check on reach, stated honestly: in a hand-picked sample of six repositories above 100k
stars (n8n 202,653; ollama 179,598; dify 153,710; open-webui 150,180; langchain 145,162;
ComfyUI 130,349), five ship a surface a non-developer can open; the exception is the category's
most ubiquitous dependency. The infra-gateway ceiling in that sample is LiteLLM at 57,446 —
roughly one third of the app tier. This is **correlational on a selection-biased sample of six**
chosen because they were already known to exceed 100k. It is an argument for building the cockpit
(`M9-*`); it is **not** evidence that a cockpit produces 100k stars, and it must never be cited
as such.

---

## Part 2 — The reframe

| From | To |
|---|---|
| "decides when to stop sampling" | "decides how much compute a request deserves" |
| one lever, the smallest | six levers, sampling last |
| needs a GPU and trajectories | needs a log file, or nothing at all |
| research artifact | default piece of plumbing |
| caveats in docs | guards in code |

### 2.1 Adoption path — three surfaces, each free and reversible

```
AUDIT                          SHADOW                         SERVE
point at logs, get a report -> mirror traffic, change      -> serve traffic, ledger records
zero infra, zero risk          zero response bytes            what each lever actually saved
```

Nothing is gated. Nothing phones home. Each step is independently useful and independently
abandonable. The user never has to commit before they have evidence from their own traffic.

---

## Part 3 — The safety net we own

This is the differentiator, expressed as engineering rather than assurance. Each row is a real
way a spend-control plane hurts its user. Column 2 is the industry-normal move: push the risk
onto the operator. Column 3 is our obligation.

| Failure mode | Normal move | **We own it** | Card |
|---|---|---|---|
| Semantic cache returns a wrong answer to a real user | config flag + a docs warning | route enabling semantic cache **starts in shadow and physically cannot serve** until an agreement report exists; conservative default threshold; every hit records its similarity | `M3-T02` |
| Cache serves stale content | document TTL | per-route TTL **required**; no infinite default; `no-store` hints honored; explicit invalidation command | `M3-T01` |
| Latency silently regresses from batching | document the tradeoff | batch is a **separate endpoint**; a synchronous request can never be converted to batch | `M5-T01` |
| Response shape drift breaks the client | "mostly compatible" | byte-compatibility asserted against real provider SDK types plus golden fixtures, on every lever path | `S-T02` |
| Spend cap blown, discovered later | dashboard alert after the fact | caps enforced **pre-upstream** against the ledger; the request is refused before money is spent | `P2-T05` |
| Credentials leak into logs, metrics, or reprs | "use a secret manager" | env-only ingestion, permanently redacted reprs, cardinality-bounded metric labels, adversarial test asserting absence | existing + `S-T03` |
| Routing quietly degrades quality | "monitor your evals" | escalation predicates are observed-only and enumerated; quality-affecting levers are quarantined from the headline number and require shadow first | `M4-T01`, `M1-T05` |
| Cache poisoning across tenants | document key scoping | cache key includes the principal **by default**; sharing requires explicit opt-in | `M3-T01` |
| Prefix caching silently not working | user never finds out | volatile-prefix detector actively reports the exact cause — the single most common reason caching fails in production | `M2-T02` |
| Running the audit leaks prompts | "redact your logs first" | ingest hashes and **discards text on read**; safe on production logs by construction, no data-handling review needed | `M1-T01` |
| Upgrade breaks a working config | changelog entry | versioned config schema, `branchpilot migrate`, load fails closed naming the exact field and fix | `S-T04` |
| Savings number cannot be defended | disclaimer | counterfactual definition recorded per request; response-identical levers reported separately from quality-affecting ones; ranges, never point estimates | `M8-T01` |
| Ambiguous log format silently misparsed | best-effort guess | detection **refuses** and names the candidates; malformed records raise with an index; parsed/skipped counts always reported | `M1-T01` |
| Gateway dies and loses in-flight batch work | "at-least-once, sorry" | durable batch state; restart resumes; no silent loss | `M5-T01` |
| Helpful error text leaks the upstream identity to an untrusted caller | put the actionable message in the HTTP body | **operator-facing and client-facing errors are different surfaces.** Rich `fix:` text goes to logs keyed by request id; the HTTP body carries a constant. Caught in Batch 1: an adapter refusal was returning `provider 'anthropic' does not support ... models.<alias>.options` to the client | `upstream.py` boundary, regression-tested |

### 3.1 Gating rule

**A lever may not move from `shadow` to `serve`, nor appear in the cockpit as available, until its
row above is closed in code and covered by a test.** `S-T01` builds the enforcement harness that
makes this mechanical rather than aspirational.

---

## Part 4 — The hyperusability standard

Testable, not vibes. `S-T05` turns each into an automated check where possible.

| # | Rule | Test |
|---|---|---|
| U1 | **Zero decisions to first value.** `branchpilot audit logs.jsonl` with no flags produces a full report: format auto-detected, prices resolved, opportunities ranked | CLI test with a bare invocation on each supported format |
| U2 | **One line to adopt.** Change `base_url`, or change one import via `branchpilot.dropin`. Nothing else | drop-in test asserting the shim adds nothing but base URL and headers |
| U3 | **Under 60 seconds to a dollar figure** from a cold `pip install` | timed test on a 10k-record fixture |
| U4 | **Every OPERATOR-facing error names the fix.** Format: what failed, which file/field, the exact change. Scope is deliberate: CLI output, config-load failures, library exceptions, and logs. It does **NOT** extend to gateway HTTP response bodies — those go to an untrusted caller and must stay constant and sanitized, with the actionable detail logged server-side keyed by request id. Batch 1 shipped this leak by reading U4 too broadly; see the last row of Part 3 | test asserting every operator-facing error contains `fix:`, plus `S-T03` asserting HTTP bodies never carry provider identity, config paths, or `fix:` text |
| U5 | **No account, no key, no network.** Audit, qualify, and report are fully offline | test asserting zero sockets opened during an audit |
| U6 | **Safe defaults, always.** Default config for every lever is the conservative one | config test snapshotting defaults against an approved table |
| U7 | **Explain any decision.** Every request's decision trace is retrievable and human-readable | the existing flight recorder, extended to all six levers |
| U8 | **Uninstall is clean.** Removing the gateway restores original behavior with one `base_url` revert; no residue in provider accounts | documented + tested rollback path |
| U9 | **Docs are not load-bearing.** A user who reads nothing succeeds at the audit and the base_url swap | measured by issue triage: "how do I" issues are a bug in the product, not the docs |
| U10 | **Works on the boring stack.** SQLite default, no Redis required, no Kubernetes required, single process is a supported deployment | CI runs the full path with defaults only |

---

## Part 5 — Kill list

| Asset | Action | Why |
|---|---|---|
| Learned RL policy as headline | demote to optional plugin; keep code and published FAIL/NO-GO under `/evidence/` | lever 6 of 6, and the risky half of it |
| GSM8K framing in the front door | move to `/evidence/` | signals research toy; nobody's production workload is GSM8K |
| `quickstart` as the first command | replace with `audit` | synthetic data + torch extra is the wrong first impression |
| `stream: Literal[False]` | remove; implement `P2-T01` | disqualifying for most real apps |
| Modal apparatus in README | keep in repo, out of the front page | research infra, zero adoption value |
| v2 pricing / open-core / enterprise sections | deleted | not the goal |

Evidence artifacts are never deleted or rewritten. They stay hash-verified under `/evidence/`.

---

## Part 6 — Phases, gated on usability

### P0 — The answer to "does this help me?" (weeks 0–6)
`M1-T01..T06`, `G1-T01..T02`, `M10-T01`, `S-T01`, `S-T05`.

**Exit:** `branchpilot audit` runs bare on ≥5 real third-party log dumps with zero flags; U1, U3,
U4, U5 tests green; the safety-net harness (`S-T01`) is enforcing; README's first command is `audit`.

### P1 — The levers that actually matter (weeks 4–16)
`P2-T01..T03`, `M2-*`, `M3-*`, `M4-*`, `M5-*`, `S-T02..T04`.

**Exit:** one default-ish config cuts spend materially on a public reference workload with
**byte-identical response shape**; every Part 3 row for shipped levers closed and tested;
`security-reviewer` PASS; streaming works in both modes.

### P2 — Nothing left to object to (weeks 12–20)
`M7-*`, `M10-T02..T03`, `M11-*`, `P2-T04`.

**Exit:** shadow mode proven to change zero response bytes; a LiteLLM or Helicone user migrates
with one command; `docker compose up` works on a laptop with defaults only.

### P3 — The surface people show each other (weeks 16–28)
`M9-*`, `G1-T03..T04`.

**Exit:** `docker run` → cockpit populated in under 5 seconds; `designer` PASS desktop + mobile;
accessibility ≥95; U7 satisfied for all six levers.

### Kill criteria

| Signal | Threshold | Action |
|---|---|---|
| Audit finds nothing | median identified opportunity <8% of spend across 10 real workloads | provider-native features have absorbed the category; narrow to observability + qualification, or stop |
| Users still need docs to get value | "how do I" issues >40% of intake after P2 | the product is not usable; stop building features and fix the path |
| A lever ships and burns someone | any incident where a lever changed a user's output unexpectedly | halt feature work, close the class of failure, publish the post-mortem |
| Nobody adopts despite P3 | <500 stars and <20 real deployments 90 days post-cockpit | distribution problem; re-examine channel before writing more code |

---

## Part 7 — Task cards

Format: `ID | Title / Depends / Agent / Files / Contract / Acceptance / Verify / Non-goals`.
Weaker agents: execute exactly one card, run only its Verify command, do not commit, report output.

### S — Safety net enforcement (build these early; they gate everything else)

```
S-T01 | Disclaimer-to-guard sweep and enforcement harness
Depends: none
Agent: task
Files: tests/test_safety_net.py, docs/internal/owned-failures.md
Contract:
  - Grep the repository (src + docs + examples) for disclaimer language: "operator must",
    "user should", "be careful", "note that", "make sure to", "at your own risk", "we assume".
  - Produce docs/internal/owned-failures.md: one row per hit with (location, the risk, the
    decision: GUARD | DETECT | REFUSE | ACCEPTED-WITH-REASON, and the card that closes it).
  - Build tests/test_safety_net.py as a registry-driven harness: a table of
    (failure_id, lever, guard_test_name, status). The test FAILS if a lever is enabled in the
    default config while any of its failure rows is unclosed.
  - This inverts the usual dynamic: shipping a lever without closing its footgun becomes a
    red test, not a judgement call.
Acceptance:
  - Every current disclaimer hit is classified; ACCEPTED entries carry a written reason.
  - Harness fails when a fixture lever is marked serve-eligible with an open failure row.
  - Harness passes on current main.
Verify: uv run --locked --extra dev pytest tests/test_safety_net.py
Non-goals: fixing the individual footguns (that is each lever's card).
```

```
S-T02 | Response byte-compatibility golden suite
Depends: none
Agent: task
Files: tests/test_response_compat.py, tests/golden/responses/*.json
Contract:
  - Golden fixtures of provider responses (non-streaming and streaming chunk sequences) for each
    supported provider shape.
  - For every lever path (passthrough, cache hit, routed, sampled, batch-materialized), assert
    the response validates against the real provider SDK type AND that the field set matches the
    golden fixture exactly, except for a documented allow-list (id, created, and the additive
    `branchpilot` extension).
  - Extension fields live under a single `branchpilot` key. No top-level additions ever.
  - Streaming: chunk boundaries may differ in deferred mode, but the concatenated content and
    the terminal sentinel must match.
Acceptance:
  - Adding a stray top-level field to any response path fails the test.
  - All six lever paths covered.
Verify: uv run --locked --extra dev pytest tests/test_response_compat.py
Non-goals: proving semantic equivalence of different models' text.
```

```
S-T03 | Secret and content leakage suite
Depends: none
Agent: task
Files: tests/test_leakage.py
Contract:
  - Single suite asserting, across HTTP responses, log records, metric output, audit records,
    ledger records, ingest records, error messages, exception reprs, and middleware reprs:
    no inbound key, no upstream key, no upstream host, no upstream model name, no prompt text,
    no completion text.
  - Implementation: run a request through the full stack with sentinel values injected into every
    secret and content field, then scan every captured artifact for those sentinels.
Acceptance: sentinel scan clean; deliberately adding a prompt to a log line fails the test.
Verify: uv run --locked --extra dev pytest tests/test_leakage.py
Non-goals: TLS or network-level concerns.
```

```
S-T04 | Versioned config schema and migrations
Depends: none
Agent: task
Files: src/branchpilot/gateway/config.py, src/branchpilot/config_migrate.py,
       src/branchpilot/cli.py, tests/test_config_migrate.py
Contract:
  - Config gains a required `schema_version` integer. Loader rejects unknown versions.
  - `branchpilot migrate config FILE [--write]` upgrades a config forward across versions,
    printing a diff. Forward-only, explicit, no guessing.
  - A rejected config error message states: the field, why it is invalid, and the exact
    replacement (a `fix:` clause per U4).
  - Every schema change ships with its migration in the same commit. Enforced by a test that
    fails when the current version has no migration path from the previous one.
Acceptance:
  - v1 fixture migrates to current, then loads.
  - Missing migration for the current version fails the test.
  - Error messages contain `fix:`.
Verify: uv run --locked --extra dev pytest tests/test_config_migrate.py
Non-goals: backward migration.
```

```
S-T05 | Usability standard checks
Depends: M1-T04
Agent: task
Files: tests/test_usability.py
Contract:
  - U1: bare `audit` on each supported format fixture exits 0 with a complete report.
  - U3: cold-path timing on a 10k-record fixture under a declared budget (generous in CI,
    reported as a number so regressions are visible).
  - U4: reflectively collect every public exception raised by the CLI and config loaders on a
    matrix of bad inputs; assert each message contains a `fix:` clause.
  - U5: monkeypatch socket creation to raise; assert audit/qualify/report complete.
  - U6: snapshot the default config for every lever against an approved table; any default
    change requires updating the table in the same commit.
  - U10: run the full audit + gateway smoke with defaults only (SQLite, memory cache, no Redis).
Acceptance: all six checks green; U6 table diff surfaces any silent default change.
Verify: uv run --locked --extra dev pytest tests/test_usability.py
Non-goals: subjective UX review (that is `designer`).
```

### M1 — Audit: the answer to "does this help me?"

```
M1-T01 | Multi-format, text-free log ingestion
Depends: none
Agent: task
Files: src/branchpilot/ingest/{__init__,formats,detect}.py, tests/test_ingest_formats.py,
       tests/fixtures/ingest/*.jsonl
Contract:
  - read_requests(path, *, format="auto", mapping=None) -> Iterator[RequestRecord]
  - RequestRecord frozen dataclass: id, timestamp, model, provider, messages_hash,
    system_prefix_hash, prompt_tokens, cached_prompt_tokens|None, completion_tokens,
    latency_ms|None, status, group_key|None, raw_index.
  - Formats v1: openai-jsonl, anthropic-jsonl, litellm-jsonl, helicone-export,
    openrouter-export, generic-jsonl (+ user mapping).
  - HARD REQUIREMENT: prompt and completion text is hashed (sha256, 16-byte hex prefix) and
    discarded on read. No text is ever retained in memory beyond the hashing call, written to
    disk, or logged. This is what makes the audit safe to run on production logs without a
    data-handling review — the central usability unlock.
  - Streaming iterator; peak memory independent of file size.
  - Ambiguous detection RAISES naming both candidates. Malformed record raises with its index.
  - read_requests_report() returns parsed/skipped counts and a reason histogram, always surfaced.
Acceptance:
  - 6-record fixture per format maps to expected RequestRecords.
  - Introspection test: no str field of any RequestRecord contains any substring of fixture
    prompt/completion text.
  - 200 MiB synthetic fixture processed with bounded RSS.
  - Ambiguous file raises; skipped counts reported.
Verify: uv run --locked --extra dev pytest tests/test_ingest_formats.py
Non-goals: gateway, pricing, rendering.
```

```
M1-T02 | Price book
Depends: none
Agent: task
Files: src/branchpilot/pricing/{__init__,book.py}, src/branchpilot/pricing/prices.json,
       tests/test_pricing.py, docs/pricing-data.md
Contract:
  - prices.json: (provider, model) -> {input, output, cached_input, batch_input, batch_output,
    currency, effective_date, source_url}.
  - Validation: finite, non-negative, cached_input <= input, batch_* <= standard.
  - price(...) -> Decimal. decimal.Decimal end to end; float in a cost path is a review rejection.
  - Unknown model RAISES UnknownModelError listing nearest configured ids. Never guesses.
  - Staleness guard warns past 90 days from effective_date.
  - --price-book FILE override.
Acceptance:
  - Hand-computed Decimal fixtures match exactly.
  - Unknown model raises with candidates.
  - grep-based test: no `float(` in pricing/.
Verify: uv run --locked --extra dev pytest tests/test_pricing.py
Non-goals: live price scraping.
```

```
M1-T03 | Opportunity detectors
Depends: M1-T01, M1-T02
Agent: task
Files: src/branchpilot/audit/{__init__,detectors,result}.py, tests/test_audit_detectors.py
Contract:
  Each detector returns Opportunity(lever, eligible_requests, eligible_spend, projected_saving,
  confidence_interval, assumptions[], required_changes[], risk_class, status):
  D1 prefix_cache — cluster by system_prefix_hash; repeated prefixes above the provider minimum;
     saving = repeated_prefix_tokens x (input - cached_input) MINUS the cache-write premium.
  D2 batch_lane — user-supplied eligibility predicate; saving from price-book batch rates, never
     a hardcoded 50%.
  D3 exact_dedup — identical messages_hash within a window; saving = full duplicate spend.
  D4 semantic_dedup — requires an embedding backend; without one, projected_saving=None,
     status="needs_embeddings".
  D5 tier_routing — requires labels or a difficulty proxy; without either, reports addressable
     spend only, status="needs_evaluation", projected_saving=None.
  D6 sampling — multi-sample groups only; reuses the existing replay engine; returns zero on
     single-sample workloads without erroring.
  Rules:
  - A detector that cannot substantiate a number returns None with a status. Never a guess.
  - Every projection carries an assumptions list that the report renders.
  - Overlapping eligibility deduplicated by documented precedence D3 > D1 > D2 > D4 > D5 > D6,
    with the overlap reported.
Acceptance:
  - Per-detector fixtures with hand-computed Decimal expectations.
  - Overlap test: a request eligible for D1 and D3 counted once, precedence honored.
  - Single-sample workload: D6 = 0, no crash.
  - No projection without assumptions.
Verify: uv run --locked --extra dev pytest tests/test_audit_detectors.py
Non-goals: applying changes.
```

```
M1-T04 | `branchpilot audit` — zero-flag path
Depends: M1-T03
Agent: task
Files: src/branchpilot/cli.py, src/branchpilot/audit/render.py, tests/test_cli_audit.py
Contract:
  - `branchpilot audit LOGS` works with no other arguments. Optional:
    --format, --price-book, --window, --html, --json, --currency.
  - Terminal output order: observed spend, spend by model, top-3 opportunities with ranges,
    total addressable range, and ONE recommended next action.
  - HTML: executive summary with a single headline range; per-lever detail with assumptions and
    required changes; workload profile; methodology; limitations; reproduction command; input
    SHA-256; price-book version and dates.
  - Standalone HTML: zero external resource references (regex-enforced on src=, href=, url(), @import).
  - Deterministic: identical input + price book -> byte-identical HTML.
  - Every figure is a RANGE labeled "projection from observed tokens x configured prices",
    never "measured savings".
  - Empty result renders a clean "no material opportunity found" summary, not a blank page.
Acceptance: byte-identical repeat; zero external refs; empty-case renders cleanly; bare
  invocation on every fixture format exits 0.
Verify: uv run --locked --extra dev pytest tests/test_cli_audit.py
Non-goals: uploading anything. The audit is strictly local.
```

```
M1-T05 | Risk classification of levers
Depends: M1-T03
Agent: task
Files: src/branchpilot/audit/risk.py, tests/test_audit_risk.py
Contract:
  - Two classes, displayed beside every figure:
    IDENTICAL — response-identical or provider-guaranteed (exact dedup, prefix cache, batch lane,
      provider arbitrage).
    QUALITY-AFFECTING — can change outputs (semantic cache, tier routing, sampling).
  - The default headline range includes ONLY IDENTICAL levers. QUALITY-AFFECTING appear in a
    separate section with the exact command to validate them first, and are folded into the
    headline only under --include-quality-affecting, which adds a warning block.
Acceptance: default headline excludes quality-affecting levers; the flag widens the range and
  adds the warning.
Verify: uv run --locked --extra dev pytest tests/test_audit_risk.py
Non-goals: running the validation itself.
```

```
M1-T06 | Public cost calculator page
Depends: M1-T02
Agent: designer
Files: assets/calculator/index.html, tests/test_calculator_parity.py, .github/workflows/pages.yml
Contract:
  - Client-side only. Inputs: monthly spend, provider, model mix, repeated-prefix share,
    batch-eligible share, duplicate rate. Outputs a range per lever.
  - Uses the SAME price book (embedded at deploy from prices.json) and the SAME formulas as
    M1-T03. A parity test asserts JS and Python agree across a fixture matrix.
  - Ends with the exact `branchpilot audit` command. No email gate, no analytics, no cookies,
    no network calls.
Acceptance: parity green; Lighthouse accessibility >=95; works offline.
Verify: uv run --locked --extra dev pytest tests/test_calculator_parity.py
Non-goals: lead capture.
```

### P2 — Platform debt (blocks the levers)

```
P2-T01 | Streaming, both modes
Depends: none
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/{schemas,app,upstream}.py, tests/test_gateway_streaming.py,
       docs/gateway/streaming.md
Contract:
  - Accept stream=true. Per-route mode:
    "passthrough" — single-sample routes stream upstream chunks byte-faithfully. REQUIRED for
      cache/routing-only routes; without it those levers are unusable in real chat apps.
    "deferred" — multi-sample routes complete sampling, then emit the selected answer as SSE.
  - stream_options.include_usage honored in both. All existing byte caps and limits apply.
  - Disconnect releases both semaphores, leaves no orphan task.
  - Docs state plainly, in the same paragraph as the feature, that deferred mode does not
    improve first-token latency.
Acceptance: OpenAI SDK stream=True works in both modes; passthrough bytes match upstream exactly;
  disconnect restores capacity; golden chunk fixtures from S-T02 pass.
Verify: uv run --locked --extra dev pytest tests/test_gateway_streaming.py
Non-goals: mid-stream stopping.
```

```
P2-T02 | Multi-provider translation
Depends: none
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/providers/{__init__,openai,anthropic,gemini,bedrock}.py,
       src/branchpilot/gateway/upstream.py, tests/test_providers.py
Contract:
  - Adapters translate the canonical request to each wire format and normalize responses
    (content, finish reason, usage including cached tokens, logprobs when available) back.
  - Protocol-based; fixture-driven tests; no live calls.
  - All existing bounds per provider: byte cap, no redirects, no env proxies, identity encoding,
    no retries.
  - A provider that does not report real completion-token usage is REJECTED at config load with
    a message naming the missing field and a `fix:` clause. Our entire value depends on real usage.
Acceptance: per-provider round-trip exact; usage normalization exact; missing-usage fixture fails
  config load with a fix clause.
Verify: uv run --locked --extra dev pytest tests/test_providers.py
Non-goals: embeddings/images/audio endpoints.
```

```
P2-T03 | Persistence with SQLite default
Depends: none
Agent: task
Files: src/branchpilot/store/{__init__,sqlite,postgres}.py,
       src/branchpilot/store/migrations/, tests/test_store.py
Contract:
  - Async store interface for ledger, cache metadata, batch state, rate-limit buckets.
  - SQLite is the DEFAULT and a fully supported production option for single-node (U10).
    Postgres behind extra `postgres`, exact pin. Redis is never required.
  - Forward-only versioned migrations; `branchpilot store migrate`.
  - No ORM. Parameterized SQL only; grep test forbids string-formatted SQL.
Acceptance: migration idempotent from empty; 32-writer concurrency test loses no ledger record
  and does not deadlock; defaults-only path works with zero external services.
Verify: uv run --locked --extra dev pytest tests/test_store.py
Non-goals: replication.
```

```
P2-T04 | Container, compose, Helm
Depends: P2-T03, M9-T02
Agent: task
Files: Dockerfile, docker-compose.yml, .dockerignore, deploy/helm/**, docs/deploy/*.md,
       .github/workflows/release.yml
Contract: digest-pinned multi-stage build, non-root, read-only rootfs, gateway extra only,
  cockpit assets embedded. `docker compose up` with NO edits brings up gateway + mock upstream +
  cockpit on SQLite. Helm chart with probes, ServiceMonitor, PDB, HPA. Image attested on release.
Acceptance: compose up -> documented curl 200 and cockpit loads, defaults only;
  `helm template | kubeconform -strict -` passes; image contains no torch or compilers.
Verify: docker build . && docker compose up -d && <documented curl>
Non-goals: cloud-specific ingress.
```

```
P2-T05 | Principals, quotas, caps, metrics, lifecycle
Depends: P2-T03
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/{limits,metrics,app}.py, src/branchpilot/gateway_entry.py, tests/*
Contract:
  - Named principals with per-principal concurrency, rpm, and SPEND caps. Spend caps enforced
    against the ledger BEFORE the upstream call — the request is refused rather than discovered
    over budget afterwards.
  - Token-bucket limiter, monotonic clock, no external store.
  - Prometheus /metrics, opt-in, cardinality bounded by route count. Never key, user, prompt,
    or answer labels.
  - /readyz (no upstream contact) and graceful SIGTERM drain with a configured deadline.
Acceptance: rpm and spend caps produce the documented status with zero upstream calls; metrics
  cardinality bounded; SIGTERM drains in-flight sessions; S-T03 leakage suite green.
Verify: uv run --locked --extra dev pytest tests/test_gateway_limits.py tests/test_gateway_metrics.py tests/test_gateway_lifecycle.py
Non-goals: distributed rate limiting.
```

### M2–M5 — The levers

```
M2-T01 | Prefix analyzer
Depends: M1-T01
Agent: task
Files: src/branchpilot/cache/prefix.py, tests/test_cache_prefix.py
Contract: from a RequestRecord stream compute per-cluster longest cacheable prefix, token length,
  hit frequency, provider minimum, and a CachePlan with breakpoints, write premium, read saving,
  and break-even hit count.
Acceptance: fixture with a stable 2k-token system prompt yields the correct breakpoint and a
  break-even count matching hand computation.
Verify: uv run --locked --extra dev pytest tests/test_cache_prefix.py
Non-goals: mutating requests.
```

```
M2-T02 | Prefix-cache injection + volatile-prefix detector
Depends: M2-T01, P2-T02
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/cache_control.py, src/branchpilot/gateway/config.py,
       tests/test_gateway_cache_control.py
Contract:
  - Route option `prefix_cache: {enabled, breakpoints, min_tokens}`. Anthropic-family: inject
    cache_control markers at computed breakpoints. OpenAI-family: enforce prefix stability.
  - VOLATILE-PREFIX DETECTOR (the owned failure): hash the leading N tokens over a rolling
    window; when the hash churns beyond a threshold, emit a metric and exactly one log per route
    naming the likely culprit (timestamp, uuid, or per-user string ahead of the stable prefix).
    Silent cache misses are the #1 reason prefix caching fails in production; the user must not
    have to discover this from their invoice.
  - Cached-token counts from upstream recorded in usage, metrics, and ledger.
Acceptance: markers at expected positions; volatile prefix triggers exactly one warning per route
  per process plus a metric; cached-token accounting flows to the ledger.
Verify: uv run --locked --extra dev pytest tests/test_gateway_cache_control.py
Non-goals: implementing our own KV cache.
```

```
M3-T01 | Exact-match response cache
Depends: P2-T03
Agent: task | review: security-reviewer
Files: src/branchpilot/cache/{store,exact}.py, src/branchpilot/gateway/app.py,
       tests/test_cache_exact.py
Contract:
  - Stores: memory (default), disk, Redis (extra). Redis never required.
  - Key = sha256 of canonicalized (model alias, messages, output-affecting decoding options,
    AND the principal by default). Cross-principal sharing requires explicit
    `cache_scope: "shared"` — poisoning across tenants is closed by default, not documented.
  - TTL per route is REQUIRED; there is no infinite default. `no-store` hints honored.
    `branchpilot cache invalidate --route R [--key K]` provided.
  - Hit returns a byte-compatible completion with `x-branchpilot-cache: hit`, zero upstream
    tokens in usage, original usage preserved under `branchpilot.cached_usage`.
  - NEVER cache temperature>0 unless `cache_stochastic: true` is set per route, which logs once
    with the correctness caveat.
  - Byte-size cap with LRU eviction.
Acceptance: identical request served with zero upstream calls and S-T02 compat green; missing TTL
  fails config load with a fix clause; cross-principal isolation proven; temperature>0 not cached
  by default; cap enforced under flood.
Verify: uv run --locked --extra dev pytest tests/test_cache_exact.py
Non-goals: semantic matching.
```

```
M3-T02 | Semantic cache, shadow-first by construction
Depends: M3-T01
Agent: task | review: security-reviewer
Files: src/branchpilot/cache/semantic.py, pyproject.toml (extra `embeddings`),
       tests/test_cache_semantic.py, docs/cache/semantic.md
Contract:
  - Behind extra `embeddings`. Pluggable embedder: local model or operator-configured endpoint.
    No default that phones home.
  - THE OWNED FAILURE: a route enabling semantic cache starts in `shadow` and **the serve path is
    not reachable** until an agreement report exists for that route. This is a state machine, not
    a warning: `serve` mode with no agreement artifact fails config load.
  - Conservative default threshold; every hit records its similarity score in metrics and ledger.
  - `branchpilot cache agreement --route R` generates the report: per-threshold precision on the
    shadowed traffic, plus the recommended threshold.
  - Docs state honestly that production hit rates are typically 20–45%, not 90%+.
Acceptance: shadow never serves (upstream call count unchanged); `serve` without an agreement
  artifact fails config load with a fix clause; agreement report precision matches a labeled
  fixture; core install never imports an embedding library.
Verify: uv run --locked --extra dev pytest tests/test_cache_semantic.py
Non-goals: training an embedder.
```

```
M4-T01 | Tier routing
Depends: P2-T02
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/routing.py, src/branchpilot/gateway/config.py,
       tests/test_gateway_routing.py, docs/gateway/routing.md
Contract:
  - Ordered tiers per route: [{upstream, model, options, max_completion_tokens, escalate_when}].
  - Escalation predicates are observed-only and ENUMERATED: vote_confidence_below,
    verifier_rejected, parse_failed, finish_reason_length, self_reported_uncertainty_above.
    No free-form expressions. No client influence — a client tier hint returns 400 with zero
    upstream calls.
  - Per-tier usage and cost recorded separately in the response extension, metrics, and ledger.
  - Total bounded by per-tier caps plus a route-level total-token cap.
  - Route is QUALITY-AFFECTING: requires shadow validation before serve, same machinery as M3-T02.
Acceptance: escalation only on the declared predicate; caps never exceeded; client hint rejected
  pre-upstream; serve without shadow validation fails config load.
Verify: uv run --locked --extra dev pytest tests/test_gateway_routing.py
Non-goals: learned routing.
```

```
M4-T02 | Provider arbitrage on open weights
Depends: M4-T01
Agent: task
Files: src/branchpilot/gateway/arbitrage.py, tests/test_gateway_arbitrage.py
Contract:
  - A tier may list interchangeable upstreams serving the same declared model identity.
  - Operator must set `equivalence_asserted: true`; absent it, config load fails with a fix
    clause. We do not verify weights and will not pretend to.
  - Cheapest healthy by price book; health window on consecutive failures and p95 latency;
    optional sticky sessions.
  - Failover attempts each candidate at most once per request and is surfaced in the extension.
Acceptance: cheapest healthy chosen; unhealthy skipped; at-most-once per candidate; missing
  assertion fails config load.
Verify: uv run --locked --extra dev pytest tests/test_gateway_arbitrage.py
Non-goals: live price scraping.
```

```
M5-T01 | Batch lane as an explicit endpoint
Depends: P2-T02, P2-T03
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/batch.py, tests/test_gateway_batch.py, docs/gateway/batch.md
Contract:
  - POST /v1/branchpilot/batch and GET /v1/branchpilot/batch/{id}.
  - THE OWNED FAILURE: a synchronous /v1/chat/completions request is NEVER converted to batch.
    Silent latency changes break SLAs. Batch is opt-in by endpoint, enforced by test.
  - Durable state via P2-T03: a crash mid-flight resumes and completes; no silent loss.
  - Batch prices from the price book; ledger records the standard-price counterfactual.
Acceptance: mock provider batch round-trip; restart mid-flight resumes; sync endpoint never
  routes to batch; ledger shows both prices.
Verify: uv run --locked --extra dev pytest tests/test_gateway_batch.py
Non-goals: cross-provider batch aggregation.
```

### M7–M9 — Zero-risk adoption and the visual surface

```
M7-T01 | Shadow mode
Depends: M3-T01, M4-T01
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/shadow.py, tests/test_gateway_shadow.py, docs/gateway/shadow.md
Contract:
  - Route mode `shadow`: the client is served by the CONTROL path (baseline tier, no cache
    serving, no adaptivity) and the response is returned unmodified. The optimized path runs in
    parallel at a configurable sample rate; both outcomes recorded.
  - Client response is BYTE-IDENTICAL to control. Test-enforced via S-T02 goldens.
  - Shadow traffic has its own rate limit and spend cap; exceeding it disables shadowing and logs
    once, never affecting client traffic.
  - Produces the savings-and-agreement report used by M3-T02/M4-T01 promotion.
  - Shadow cost is real spend, reported as "evaluation cost", never netted against projected savings.
Acceptance: byte-identical responses across the golden matrix; cap disables shadowing without
  client impact; report separates projected saving from evaluation cost.
Verify: uv run --locked --extra dev pytest tests/test_gateway_shadow.py
Non-goals: automatic promotion.
```

```
M8-T01 | Decision and savings ledger
Depends: P2-T03, M1-T02
Agent: task
Files: src/branchpilot/ledger/{__init__,record,store,rollup}.py, tests/test_ledger.py,
       docs/ledger.md
Contract:
  - Immutable append-only record per served request: id, route, principal, tiers used, per-tier
    tokens (prompt/cached/completion), cache outcome and similarity if any, sampling decisions,
    price-book version, cost_paid (Decimal), cost_counterfactual (Decimal), per-lever
    attribution, evidence tier (MEASURED via shadow vs DERIVED).
  - Counterfactual definition recorded PER RECORD so a later config change cannot retroactively
    alter history. Definition: route baseline tier, standard prices, no cache, no adaptivity,
    exactly one call.
  - Rollups by window. No prompt or completion text, ever.
  - `branchpilot report savings --month YYYY-MM` renders deterministic HTML/JSON with the
    methodology, the counterfactual definition, and the MEASURED/DERIVED split.
Acceptance: rollup arithmetic exact under Decimal on a 10k-record fixture; historical records
  immutable under config change; no text content (S-T03 green); deterministic rendering.
Verify: uv run --locked --extra dev pytest tests/test_ledger.py
Non-goals: billing, invoicing, signatures.
```

```
M9-T01 | Read-only cockpit API
Depends: M8-T01, P2-T03
Agent: task | review: security-reviewer
Files: src/branchpilot/gateway/admin.py, tests/test_gateway_admin.py
Contract:
  - Separate sub-app, distinct port and admin token. NEVER the inbound inference key.
    Read-only in v1; no config mutation.
  - /admin/api/{summary,routes,levers,timeseries,recent,health}. `recent` returns metadata only,
    never prompt or completion text.
  - Bounded windows, pagination, no unbounded scans.
Acceptance: inference key cannot reach admin and vice versa; no endpoint returns text content;
  bounded response times on a 1M-record ledger fixture.
Verify: uv run --locked --extra dev pytest tests/test_gateway_admin.py
Non-goals: config editing from the UI.
```

```
M9-T02 | Cockpit UI
Depends: M9-T01
Agent: designer
Files: ui/**, src/branchpilot/gateway/static/**, docs/cockpit.md, tests/test_ui_assets.py
Contract:
  - SPA, no runtime CDN dependency; built assets embedded in wheel and image.
  - Screens: (1) Spend — live spend, saving to date, saving rate, projection;
    (2) Waste — ranked unexploited opportunities, each with the exact config diff to apply;
    (3) Routes — per-route spend, tier mix, cache hit rate, escalation rate;
    (4) Decisions — the flight recorder for any recent request, covering all six levers (U7);
    (5) Report — download the savings report.
  - Dark-first, dense, legible at 1280x640.
  - Keyboard navigable, WCAG AA contrast, no color-only encoding.
  - Zero telemetry, zero third-party origins (asset-parsing test enforces).
Acceptance: `docker run` -> cockpit reachable and populated; designer PASS desktop + mobile;
  Lighthouse accessibility >=95; asset test finds no external origin.
Verify: uv run --locked --extra dev pytest tests/test_ui_assets.py + documented docker run + designer review
Non-goals: multi-tenant org management.
```

```
M9-T03 | `--demo` seeded dataset
Depends: M9-T02
Agent: task
Files: src/branchpilot/demo/seed.py, tests/test_demo_seed.py
Contract:
  - `branchpilot-gateway --demo` boots a mock upstream and a synthetic ledger; every cockpit
    screen is populated within 5 seconds.
  - Demo data is watermarked "DEMO DATA" in the UI and in every report generated from it.
    A screenshot of demo data must be self-identifying — non-negotiable.
Acceptance: populated <5s; watermark on every screen; demo-derived reports carry it in HTML and JSON.
Verify: uv run --locked --extra dev pytest tests/test_demo_seed.py
Non-goals: fake numbers anywhere outside demo mode.
```

### M10–M11 — Meet people where they are

```
M10-T01 | LiteLLM config importer
Depends: none
Agent: task
Files: src/branchpilot/importers/litellm.py, src/branchpilot/cli.py, tests/test_import_litellm.py
Contract: `branchpilot import litellm --config FILE --output DIR` converts model_list, api_base,
  api_key env references, and budgets into a BranchPilot config + plan. Unsupported constructs are
  never silently dropped: MIGRATION-NOTES.md lists each with the equivalent or an explicit
  "not supported".
Acceptance: fixture converts to a config that load_gateway_config accepts; notes list every
  unmapped key.
Verify: uv run --locked --extra dev pytest tests/test_import_litellm.py
Non-goals: a runtime shim for LiteLLM's SDK.
```

```
M10-T02 | Helicone migration path
Depends: M10-T01
Agent: task
Files: src/branchpilot/importers/helicone.py, docs/migrate/helicone.md,
       tests/test_import_helicone.py
Contract: import Helicone export logs into the audit path; convert base-URL/header configuration.
  Docs state Helicone's maintenance status factually with a source link and never disparage the
  project.
Acceptance: export fixture produces a valid audit run; links resolve.
Verify: uv run --locked --extra dev pytest tests/test_import_helicone.py
Non-goals: importing dashboards.
```

```
M10-T03 | `branchpilot init`
Depends: P2-T02
Agent: task
Files: src/branchpilot/cli.py, src/branchpilot/templates/*.tmpl, tests/test_cli_init.py
Contract: `branchpilot init --output-dir DIR [--upstream-url URL] [--upstream-model NAME]`
  writes a valid config + plan that load without edits, prints the exact export lines and run
  command, and refuses to overwrite without --force. Every generated default is the safe one (U6).
Acceptance: generated pair loads with fake env vars; generated plan loads; no secret written to disk.
Verify: uv run --locked --extra dev pytest tests/test_cli_init.py
Non-goals: learned plans.
```

```
M11-T01 | Drop-in SDK shims
Depends: P2-T01
Agent: task
Files: src/branchpilot/dropin/{openai,anthropic}.py, tests/test_dropin.py, docs/dropin.md
Contract: `from branchpilot.dropin.openai import OpenAI` returns a client pointed at a
  BranchPilot base_url from env, otherwise identical to the upstream SDK client. Adds NO behavior
  beyond base_url and default headers — it is not a wrapper layer (U2).
Acceptance: identical behavior against a mock; test asserts no added request fields beyond base
  URL and headers.
Verify: uv run --locked --extra dev pytest tests/test_dropin.py
Non-goals: reimplementing provider SDKs.
```

### G1 — Communication

```
G1-T01 | Sourced market notes
Depends: none
Agent: scout
Files: docs/internal/market-notes.md
Contract: every external figure used anywhere in product or docs, with source URL, access date,
  and a VERIFY-BY date 90 days out. Competitor feature matrix with source links.
Acceptance: every number cited in README/docs/calculator traces to a row here.
Verify: reviewer claim-trace.
Non-goals: unsourced market sizing.
```

```
G1-T02 | README rewrite
Depends: M1-T04
Agent: task | review: reviewer
Files: README.md
Contract:
  - Above the fold: one sentence of what it does, the `audit` command, a real report screenshot,
    the base_url swap. Nothing else.
  - Order: Audit -> Shadow -> Serve -> Levers -> Deploy -> Evidence -> Limitations.
  - Sampling and the learned policy appear under Levers, not the hero.
  - Every number links to an artifact or is labeled a sourced market range.
Acceptance: reviewer PASS tracing each claim; "research" does not appear above the fold.
Verify: reviewer claim-trace pass.
Non-goals: deleting evidence pages.
```

```
G1-T03 | Comparison pages
Depends: G1-T01
Agent: task | review: reviewer
Files: docs/compare/{litellm,portkey,helicone,cloudflare-ai-gateway,diy}.md
Contract: one page per alternative. Honest columns including where they are better. Every claim
  sourced and dated. Each page ends with the audit command.
Acceptance: reviewer PASS; zero unsourced comparative claims; competitor strengths stated plainly.
Verify: uv run --locked --extra docs mkdocs build --strict
Non-goals: FUD.
```

```
G1-T04 | Launch sequence
Depends: P3 exit
Agent: designer | review: reviewer
Files: docs/internal/launch-kit.md
Contract: Show HN, r/LocalLLaMA, technical thread. Hook = the audit one-liner plus one real
  figure from our own traffic or a consenting user, and the cockpit screenshot. No superlatives,
  no fabricated testimonials. Purchased stars, sock accounts, and follow-for-follow are
  prohibited outright — detectable and terminal for a dev-tool project.
Acceptance: reviewer PASS that every claim is reproducible by a reader in one command.
Verify: manual claim-trace.
Non-goals: manufactured social proof.
```

```
G1-T05 | Integration contributions
Depends: P2 exit
Agent: scout (identify) + task (execute)
Files: docs/internal/partners.md
Contract: 15 projects where BranchPilot is genuinely additive; for each, the smallest genuinely
  useful PR to offer and the value in one sentence. Real contributions, never link drops.
Acceptance: 15 entries, 5 PRs opened.
Verify: manual link check.
Non-goals: spam.
```

---

## Part 8 — Ownership and parallel batches

One concurrent card per file. Serialize where listed.

| File | Cards | Rule |
|---|---|---|
| `gateway/app.py` | P2-T01, P2-T02, P2-T05, M3-T01, M4-T01, M5-T01, M9-T01 | **STRICTLY SERIAL** |
| `gateway/config.py` | P2-T02, P2-T05, S-T04, M2-T02, M3-T01, M3-T02, M4-T01, M7-T01 | **STRICTLY SERIAL** |
| `gateway/upstream.py` | P2-T01, P2-T02, M2-T02 | serial |
| `cli.py` | M1-T04, M8-T01, M10-T01, M10-T03, S-T04, P2-T03 | serial |
| `ingest/**` | M1-T01 | exclusive |
| `pricing/**` | M1-T02 | exclusive |
| `audit/**` | M1-T03, M1-T04, M1-T05 | parallel by file |
| `cache/**` | M2-T01, M3-T01, M3-T02 | parallel by file |
| `store/**` | P2-T03 | exclusive |
| `ledger/**` | M8-T01 | exclusive |
| `ui/**` | M9-T02 | exclusive |
| `tests/test_safety_net.py` | S-T01 | exclusive; other cards ADD rows, never edit the harness |
| `README.md` | G1-T02 | exclusive |
| `benchmarks/**` | append-only; never edit a manifested file | absolute |

**Batch 1 — start now, six agents, zero collisions:**
`S-T01`, `M1-T01`, `M1-T02`, `P2-T02`, `P2-T03`, `G1-T01`.

**Batch 2:** `S-T02`, `S-T03`, `M1-T03` → `M1-T04` → `M1-T05` (serial chain), `M2-T01`, `M10-T01`.

**Batch 3:** `P2-T01`, then the `gateway/app.py` sequence under ONE integration owner in this
order: `P2-T05` → `M3-T01` → `M4-T01` → `M5-T01` → `M9-T01`.

### Dispatch template

```
# Target
<Files block verbatim> ; explicit non-goals from the card

# Change
<Contract block verbatim>

# Acceptance
<Acceptance block verbatim>
Run ONLY the card's Verify command. Do not run the full suite, formatters on unrelated files,
builds, or servers. Do not commit. Report the exact command and its complete output.
```

Batch `context` MUST include: Part 0's one-sentence standard, the relevant Part 3 rows, the
Decimal-only money rule, the no-raw-text rule, the `fix:`-clause error requirement, and the
ownership rows for that batch.

### Review gates

| Touches | Reviewer | Blocking output |
|---|---|---|
| `gateway/**` | `security-reviewer` | PASS with per-finding disposition |
| Any lever moving toward `serve` | `security-reviewer` | confirmation its Part 3 row is closed in code |
| Any figure or comparative claim | `reviewer` | traced to artifact or sourced note |
| `ui/**`, landing, calculator | `designer` | desktop + mobile + accessibility PASS |

---

## Part 9 — Metrics

No revenue metrics. These measure usability and reach.

| Metric | Definition | Target by P3 exit |
|---|---|---|
| **Time to first value** | cold `pip install` → dollar figure from own logs | < 60 s |
| **Decisions to first value** | required flags/config choices before a report | **0** |
| **Owned-failure coverage** | Part 3 rows closed in code / rows applicable to shipped levers | **100%** |
| **Response-byte compatibility** | golden-suite pass rate across all lever paths | **100%** |
| **"How do I" issue share** | share of intake that is a usability failure, not a bug | < 20% |
| **Defaults-only success** | can a user succeed with SQLite, no Redis, no k8s, no flags | yes, CI-enforced |
| **Audit → deployment** | audits run → gateways deployed | > 15% |
| **Shadow → serve** | shadow deployments promoted | > 40% |
| **Installs** | container pulls + pip installs | 25k/mo |
| **Stars** | pure lagging indicator | 5k |

Anti-metrics: purchased stars, paid promotion, follow-for-follow, trending manipulation.

---

## Part 10 — Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Provider-native features absorb the levers | High | own the measurement and orchestration layer; no provider computes your cross-provider counterfactual |
| R2 | LiteLLM/Portkey ship the same levers | High | compete on owned failure modes and default safety — the thing feature lists do not capture |
| R3 | A semantic-cache false hit reaches a user | Critical | serve path unreachable without an agreement artifact (`M3-T02`); conservative default threshold |
| R4 | Silent latency regression | High | batch is endpoint-only; deferred streaming documented in-line; sync never converts |
| R5 | A savings figure is disputed | High | per-record counterfactual; IDENTICAL vs QUALITY-AFFECTING split; ranges only |
| R6 | Key leak or in-path outage | Critical | existing hardening + `S-T03` + adversarial suite + `security-reviewer` gate |
| R7 | Six levers, all mediocre | High | a lever ships only when its detector, gateway path, ledger attribution, cockpit tile, AND Part 3 row are all done |
| R8 | Cockpit becomes a maintenance sink | Medium | read-only v1; no config mutation until demand proves it |
| R9 | Complexity defeats usability | High | U1–U10 are tests, not aspirations; `S-T05` fails the build when they regress |
| R10 | Reach plateaus | Likely | ~40k is the infra ceiling; the cockpit is the only credible path past it. Optimize usability; reach follows or it does not |

---

## Part 11 — Non-goals

- No pricing, tiers, licensing splits, accounts, or hosted service.
- No telemetry by default; if ever added, opt-in, aggregate-only, documented, and trivially disabled.
- Not an inference engine, model host, fine-tuning platform, agent framework, prompt manager, or eval suite.
- No per-request markup of any kind.
- No claim we cannot reproduce from a ledger record or a priced token count.
- No auto-converting sync traffic to batch; no serving semantic-cache hits without an agreement artifact.
- No deleting or rewriting existing evidence artifacts.
- No manufactured social proof.
- **No shipping a lever whose failure mode we would have to disclaim.**

---

## Part 12 — First moves

Batch 1, six parallel agents, zero file collisions:

1. `S-T01` disclaimer-to-guard sweep + enforcement harness — `task`
2. `M1-T01` text-free log ingestion — `task`
3. `M1-T02` Decimal price book — `task`
4. `P2-T02` multi-provider translation — `task` (+ `security-reviewer`)
5. `P2-T03` persistence, SQLite default — `task`
6. `G1-T01` sourced market notes — `scout`

Then the critical path: `M1-T03` → `M1-T04` → `G1-T02`. That chain turns the repository from
"clone it, install torch, generate synthetic data, read 448 lines of README" into
"`branchpilot audit logs.jsonl`" — which is the whole point.

`S-T01` is first for a reason: it makes the safety net mechanical before there are six levers to
retrofit.

---

## Appendix A — Conventions

- Commit subject: `type: imperative summary` (`feat|fix|perf|docs|test|ci|chore|bench`).
- Money is `decimal.Decimal` end to end; float in a cost path is a review rejection.
- Ingest and ledger records never contain prompt or completion text.
- Every operator-facing error message contains a `fix:` clause. Gateway HTTP bodies are the
  exception: constant and sanitized, with the detail logged server-side (see U4 and Part 3).
- Response payload byte-compatibility is a test, not an aspiration.
- Protocol commits land before result commits for any `benchmarks/` bundle.

## Appendix B — Auto-reject in review

- A figure without a ledger record, a priced token count, or a sourced range.
- A point estimate where a range is required.
- `float` in pricing/ledger; f-string SQL in store/.
- Raw prompt or completion text in any record, log line, or metric label.
- A new disclaimer where a guard is possible.
- A lever marked serve-eligible with an open Part 3 row.
- Any change to response bytes on a cache, routing, or arbitrage path.
- An operator-facing error message without a `fix:` clause, or a gateway HTTP body that carries
  provider identity, operator config paths, or `fix:` text.
- A required flag or config edit on the path to first value.
- A new hard dependency on Redis, Postgres, or Kubernetes for the default path.
