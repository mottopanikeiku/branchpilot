# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/); before 1.0 a breaking change raises the minor
version and is listed under **Changed** or **Removed** with the exact migration line.

## [Unreleased]

### Changed

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

- `branchpilot audit LOGS` — reads existing provider traffic logs and reports where compute is
  being spent, with per-lever opportunities and ranges. Runs offline with no flags, no account,
  and no network access.
- Traffic log ingestion for six formats (OpenAI, Anthropic, LiteLLM, Helicone, OpenRouter, and a
  mapped generic JSONL). **Prompt and completion text is hashed and discarded on read**, so the
  audit is safe to run against production logs without a data-handling review. Peak memory is
  independent of file size.
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
- Safety-net enforcement harness: a lever cannot be marked serve-eligible while any of its known
  failure modes is unclosed. Shipping a footgun is a failing test rather than a judgement call.
- Leakage suite asserting that inbound keys, upstream keys, upstream hosts, upstream model names,
  prompts, and completions never appear in any response body, header, log record, metric label,
  or exception representation.
- Response-shape golden suite pinning the client-visible payload across every provider and lever
  path, with all BranchPilot additions confined to a single `branchpilot` extension key.

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

- Canonical v0.2 learned-policy criterion published as an explicit **FAIL**.
- Train-only v3 capacity gate published as an explicit **NO-GO**; no fresh holdout was authorized
  and zero fresh-holdout model requests were issued.

[Unreleased]: https://github.com/mottopanikeiku/branchpilot/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mottopanikeiku/branchpilot/releases/tag/v0.3.0
