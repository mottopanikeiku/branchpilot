# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/); before 1.0 a breaking change raises the minor
version and is listed under **Changed** or **Removed** with the exact migration line.

## [Unreleased]

### Changed

- Rewrote the README around the sequential sampling loop and the negative GSM8K
  learned-policy result; kept earlier usage and study instructions in `docs/earlier-usage.md`.
- Documented the runtime, strategies, learner, and evaluator as the core, and gateway,
  pricing, audit, importers, cache analysis, and persistence as secondary tools.
- Clarified that backward Q targets are exact along each logged trajectory, not a
  guarantee of optimal stopping from observable prefixes.

- **BREAKING — `branchpilot audit` now performs a spend audit.** The trajectory-integrity
  profiler that previously answered to `audit` is now `integrity`. It is the same command with
  the same flags and behavior; only the name changed. There is deliberately no alias and no
  deprecation period, so a stale invocation fails loudly rather than doing something unexpected.

  ```
  # before
  branchpilot audit --data train.jsonl --compare validation.jsonl
  # after
  branchpilot integrity --data train.jsonl --compare validation.jsonl
  ```

- Gateway upstream errors arising from a provider translation refusal now return a constant,
  sanitized message to the client. The actionable detail — which provider rejected the request and
  which config field to change — is logged server-side against the request id instead. Previously
  the adapter's message was returned verbatim in the HTTP body, exposing the upstream provider
  identity and operator configuration paths to the caller.

### Added

- MATH-500 response collection on a pinned vLLM image and L4, with committed
  problem splits and the stopping-policy success rule fixed before sampling.
- Symbolic answer voting with Math-Verify. Vote labels depend only on answers
  already sampled; correctness labels remain separate from policy features.
- The original GSM8K learned policy is included unchanged for transfer comparisons.
- A second negative stopping result on 200 internally held-out MATH-500 problems.
  The retrained policy missed the unchanged success rule at all three primary
  costs; committed outcomes also compare the unchanged GSM8K controller and
  simple rules at matched expected sample budgets.
- Source distributions now exclude all `.venv*` directories, including the
  workflow's temporary Torch-free environment.

- CPU-only accuracy/sample trade-off figure from committed GSM8K prompt outcomes,
  with all fixed counts, confidence thresholds, agreement streaks, and learned costs.
  The committed script recomputes paired prompt-bootstrap intervals and writes a
  numeric summary for test and validation; this is not a new model run.
- `branchpilot audit LOGS` — reads existing provider traffic logs and reports where compute is
  being spent, with per-lever opportunities and ranges. Runs offline with no flags, no account,
  and no network access.
- Traffic log ingestion for six formats (OpenAI, Anthropic, LiteLLM, Helicone, OpenRouter, and a
  mapped generic JSONL). Prompt and completion text is hashed and discarded on read.
  This keeps less text in memory; hashing alone does not establish that a log is safe
  to handle or that a deployment needs no data-handling review.
- Price book with exact `Decimal` arithmetic and per-entry source URLs and effective dates. An
  unknown model raises with close-match suggestions rather than estimating from a similar model.
- Provider request/response translation for Anthropic, Gemini, and Bedrock alongside OpenAI. Each
  provider's token-accounting convention is handled explicitly; a response whose usage cannot be
  reconciled is refused rather than reconstructed.
- Prefix-cache analysis, including detection of volatile system prefixes — an injected timestamp,
  uuid, or per-user string ahead of the stable prefix silently defeats provider caching, and this
  reports it per model with a churn rate.
- LiteLLM configuration importer. Every unmapped key is recorded in a generated
  `MIGRATION-NOTES.md`; a literal secret in the source config is refused rather than copied.
- Async persistence layer for the ledger, cache metadata, batch state, and rate-limit buckets.
  SQLite is the default and requires no external service; Postgres is available behind an extra.
- Safety-net enforcement tests cover declared failure modes and reject configurations
  whose recorded modes have not been addressed. They do not prove deployment safety.
- Leakage suite asserting that inbound keys, upstream keys, upstream hosts, upstream model names,
  prompts, and completions never appear in any response body, header, log record, metric label,
  or exception representation.
- Response-shape golden suite pinning the client-visible payload across every provider and lever
  path, with all BranchPilot additions confined to a single `branchpilot` extension key.

### Fixed

- `PilotSession` now raises as soon as a strategy returns CONTINUE at the session
  horizon. Previously `run` and `run_async` requested one more sample before failing.

## [0.3.0] — 2026-08-27

### Added

- Text-only OpenAI-compatible gateway with adaptive sequential sampling, bounded admission
  control, and per-request decision headers.
- Deployable stopping strategies: fixed count, vote confidence, consecutive agreement, and
  learned Safetensors policies.
- Validation-only deployment planner. Exported plans are schema-versioned and bind the complete
  selection payload by SHA-256.
- `its_hub` scaling adapter and a sequential OpenAI adapter.
- Torch-free core wheel with signed release artifacts and SLSA build provenance.

### Evidence

- The v0.2 learned-policy success criterion was not met.
- The train-only v3 capacity comparison did not pass; no fresh holdout was authorized
  and zero fresh-holdout model requests were issued.

[Unreleased]: https://github.com/mottopanikeiku/branchpilot/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mottopanikeiku/branchpilot/releases/tag/v0.3.0
