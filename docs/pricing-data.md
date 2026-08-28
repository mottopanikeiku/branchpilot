# Price data

BranchPilot prices token usage from a **price book**: a JSON file of sourced rate cards read by
`branchpilot.pricing.PriceBook`. This page documents the file format, the conventions the numbers
follow, what is seeded today, what was deliberately left out, and how to extend it.

## Rate convention

**Every rate is quoted per 1,000,000 tokens** ("per 1M tokens", the unit every provider publishes).
`PriceBook.price()` divides by `Decimal(1_000_000)` exactly once, at the end:

```
cost = ((tokens_in - cached_in) * input
        + cached_in * cached_input
        + tokens_out * output) / 1_000_000
```

with `batch_input` / `batch_output` substituted for `input` / `output` when `batch=True`.

**Every rate is stored as a quoted JSON string, never a JSON number.** `"3.00"`, not `3.00`. A JSON
number is read back as a binary approximation; a quoted string parses into `decimal.Decimal`
exactly. The loader **rejects** unquoted rates rather than accepting and rounding them — there is
no inexact arithmetic anywhere in a cost path, and `tests/test_pricing.py` asserts that mechanically
against the whole `src/branchpilot/pricing/` package.

Token counts are integers. A non-integer token count is refused with a `fix:` clause rather than
silently truncated.

## File format

```json
{
  "schema_version": 1,
  "entries": [
    {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "input": "3",
      "output": "15",
      "cached_input": "0.30",
      "batch_input": "1.50",
      "batch_output": "7.50",
      "currency": "USD",
      "effective_date": "2026-08-27",
      "source_url": "https://platform.claude.com/docs/en/about-claude/pricing"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `provider` | Adapter id the request is billed through: `anthropic`, `openai`, `gemini`. |
| `model` | The exact model id that appears in provider logs and API calls. |
| `input` | Standard, uncached prompt tokens, per 1M. |
| `output` | Completion tokens (including reasoning/thinking tokens where the provider bills them as output), per 1M. |
| `cached_input` | Prompt-cache **read** (cache hit) rate, per 1M. |
| `batch_input` | Prompt tokens on the asynchronous batch lane, per 1M. |
| `batch_output` | Completion tokens on the asynchronous batch lane, per 1M. |
| `currency` | 3-letter uppercase ISO-4217 code. Every seeded entry is `USD`. |
| `effective_date` | ISO-8601 `YYYY-MM-DD`: the day the rate was read from `source_url` and observed to be in effect. Drives the staleness horizon. |
| `source_url` | Absolute `http(s)` URL of the page the rate was read from. |

Every field is required. Every field is validated on load, and every rejection names the file, the
entry index, the field, and the exact change to make.

### Validation rules enforced on load

- Each rate is a quoted decimal string, finite, and non-negative.
- `cached_input <= input` — a cache read is never dearer than a fresh read.
- `batch_input <= input` and `batch_output <= output` — the batch lane is never dearer than the
  synchronous lane.
- `currency` matches `^[A-Z]{3}$`.
- `effective_date` is a real calendar day in `YYYY-MM-DD` form.
- `source_url` is an absolute `http` or `https` URL.
- No duplicate `(provider, model)` pair.
- `schema_version` equals the version this build reads; unknown field names are refused rather
  than ignored.

## Unknown models raise. They are never estimated.

`PriceBook.price()` on a `(provider, model)` pair with no entry raises `UnknownModelError` listing
the closest configured model ids for that provider (`difflib.get_close_matches`) and pointing at
`--price-book FILE`. There is no nearest-neighbour fallback and no default rate: a wrong number in
a spend report is worse than a refusal, because a refusal is visible.

This is why omitting an unverifiable model is the correct outcome rather than a gap — see
[Deliberate omissions](#deliberate-omissions).

## Staleness

`PriceBook.staleness_warnings(today=None)` returns one `StalenessWarning` per entry whose
`effective_date` is more than **90 days** before today, oldest first. Each warning carries the
entry, its age in days, the horizon, and a `message()` naming the `source_url` to re-check.

Library code never prints. The caller — CLI, report renderer, cockpit — decides whether a stale
rate is a note, a banner, or a refusal.

## Cache and batch stacking: the conservative choice

Some providers stack the batch discount with the prompt-cache discount, so a cache read inside a
batch request bills below `cached_input`. Schema v1 has no `batch_cached_input` field, so
`price(batch=True)` applies `batch_input` / `batch_output` to fresh and completion tokens and keeps
the **standard** `cached_input` rate for cache reads.

That is deliberate and it is the conservative direction: it can only overstate the cost of the
batch lane, which understates the saving attributed to moving traffic onto it. A schema v2 adding
`batch_cached_input` would tighten this; until the field exists, no stacking factor is inferred.

## Seeded entries

25 entries, all read on **2026-08-27** from the provider's own pricing page. Nothing here is
scraped at runtime and BranchPilot opens no socket to price a request.

### Anthropic — 9 entries

Source: <https://platform.claude.com/docs/en/about-claude/pricing> (standard, prompt-caching, and
batch tables). Model ids confirmed against
<https://platform.claude.com/docs/en/about-claude/models/overview> and
<https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions>.

| Model | input | output | cached_input | batch_input | batch_output |
|---|---|---|---|---|---|
| `claude-fable-5` | 10 | 50 | 1 | 5 | 25 |
| `claude-opus-5` | 5 | 25 | 0.50 | 2.50 | 12.50 |
| `claude-opus-4-8` | 5 | 25 | 0.50 | 2.50 | 12.50 |
| `claude-opus-4-7` | 5 | 25 | 0.50 | 2.50 | 12.50 |
| `claude-opus-4-6` | 5 | 25 | 0.50 | 2.50 | 12.50 |
| `claude-sonnet-5` | 2 | 10 | 0.20 | 1 | 5 |
| `claude-sonnet-4-6` | 3 | 15 | 0.30 | 1.50 | 7.50 |
| `claude-sonnet-4-5` | 3 | 15 | 0.30 | 1.50 | 7.50 |
| `claude-haiku-4-5` | 1 | 5 | 0.10 | 0.50 | 2.50 |

Two published multipliers hold across this table and are visible in the numbers: batch is 50% of
both input and output, and a cache read is 10% of base input.

### OpenAI — 15 entries

Source: <https://developers.openai.com/api/docs/pricing> (standard and batch tables, short-context
columns).

| Model | input | output | cached_input | batch_input | batch_output |
|---|---|---|---|---|---|
| `gpt-5.2` | 1.75 | 14.00 | 0.175 | 0.875 | 7.00 |
| `gpt-5.4-mini` | 0.75 | 4.50 | 0.075 | 0.375 | 2.25 |
| `gpt-5.4-nano` | 0.20 | 1.25 | 0.02 | 0.10 | 0.625 |
| `gpt-5.1` | 1.25 | 10.00 | 0.125 | 0.625 | 5.00 |
| `gpt-5` | 1.25 | 10.00 | 0.125 | 0.625 | 5.00 |
| `gpt-5-mini` | 0.25 | 2.00 | 0.025 | 0.125 | 1.00 |
| `gpt-5-nano` | 0.05 | 0.40 | 0.005 | 0.025 | 0.20 |
| `gpt-4.1` | 2.00 | 8.00 | 0.50 | 1.00 | 4.00 |
| `gpt-4.1-mini` | 0.40 | 1.60 | 0.10 | 0.20 | 0.80 |
| `gpt-4.1-nano` | 0.10 | 0.40 | 0.025 | 0.05 | 0.20 |
| `gpt-4o` | 2.50 | 10.00 | 1.25 | 1.25 | 5.00 |
| `gpt-4o-mini` | 0.15 | 0.60 | 0.075 | 0.075 | 0.30 |
| `o3` | 2.00 | 8.00 | 0.50 | 1.00 | 4.00 |
| `o4-mini` | 1.10 | 4.40 | 0.275 | 0.55 | 2.20 |
| `o3-mini` | 1.10 | 4.40 | 0.55 | 0.55 | 2.20 |

### Google Gemini — 1 entry

Source: <https://ai.google.dev/gemini-api/docs/pricing> (paid tier, standard and batch tables).
`cached_input` is the published context-caching read rate.

| Model | input | output | cached_input | batch_input | batch_output |
|---|---|---|---|---|---|
| `gemini-3.5-flash` | 1.50 | 9.00 | 0.15 | 0.75 | 4.50 |

## Deliberate omissions

Each of these is absent because schema v1 cannot express the published rate faithfully, or because
a required field is not published. An absent model raises `UnknownModelError` with candidate
suggestions — a loud, correct outcome. A fabricated rate would be a silent, wrong one.

| Omitted | Why |
|---|---|
| OpenAI context-tiered models (`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.5-pro`) | Published as two rate tiers keyed on prompt length (short vs. long context). Schema v1 has one rate per model and `price()` takes no context-length argument, so a single number would be wrong for a large share of traffic. Needs a context-tier field. |
| Gemini 3.7 Flash, 3.6 Flash | Published as promotional rates with a dated step-up ("$0.75 through December 31, 2026, $1.50 starting January 1, 2027"). Schema v1 has one `effective_date` per entry and no validity window, so one entry cannot represent both. Needs a date-ranged rate. |
| Gemini context-cache **storage** ($/1M tokens/hour) | A time-based charge, not a per-token charge. There is no field for it and `price()` takes no duration. |
| Fast-mode / priority / flex service tiers (OpenAI, Anthropic, Gemini) | Separate published rate tables per service tier. Schema v1 keys on `(provider, model)` only. Needs a service-tier field. |
| `claude-mythos-5` | Limited availability; its API model id is not published on the pages read, so the entry key could not be verified. |
| `claude-opus-4-5`, `claude-haiku-3-5`, `claude-opus-4-1`, `claude-opus-4`, `claude-sonnet-4` | Rates are published, but the exact API model ids were not seen verbatim on the pages read; an entry keyed on a guessed id never matches a log line, so it would be dead weight. |
| `gpt-5-pro`, `gpt-5.2-pro`, `o1-pro`, `o3-pro`, `gpt-4o-2024-05-13`, `gpt-4-turbo-2024-04-09`, `gpt-4-0613`, `gpt-3.5-turbo*`, `davinci-002`, `babbage-002` | No cached-input rate is published for these, and `cached_input` is a required field. Filling it with `input` would misreport any prefix-cache opportunity as zero saving. |
| `gpt-5.6-cyber`, `gpt-5.5-cyber`, `gpt-5.4-cyber` (Daybreak) | No batch rate is published, and `batch_input` / `batch_output` are required fields. |
| Realtime, audio, image, embedding, and TTS models | Billed per modality (and sometimes per character or per minute), which the schema's single input/output pair cannot represent. |
| Amazon Bedrock, Google Vertex, Microsoft Foundry, Claude Platform on AWS | Rates are set and invoiced by the cloud provider, with regional and endpoint-type premiums (a documented 10% uplift for regional endpoints). Each needs its own provider key and region dimension. |
| Data-residency and inference-geography multipliers (1.1x) | A multiplier on an existing entry, not a separate rate card. Needs a modifier concept. |
| Provider discounts, committed-use, and private offers | Not publishable; per-account. Override with your own price book. |

## Extending or overriding

Anything not seeded — a negotiated rate, a self-hosted model, a provider not listed above — goes in
your own file:

```python
from branchpilot.pricing import PriceBook

book = PriceBook.load("my-prices.json")  # or PriceBook.load() for the packaged rates
```

On the command line the equivalent is `--price-book FILE`.

When adding an entry: read the rate off the provider's own pricing page, put that page's URL in
`source_url`, and put the day you read it in `effective_date`. Those two fields are what make a
spend figure defensible, and the 90-day staleness horizon is what stops a rate rotting unnoticed.

## Refreshing the seeded rates

1. Re-read each `source_url` in `src/branchpilot/pricing/prices.json`.
2. Update changed rates and set `effective_date` to the day you read them.
3. Update the tables on this page, including the omissions table if a blocking reason has gone away
   (for example, a promotional rate becoming the standard one).
4. Run `uv run --locked --extra dev pytest tests/test_pricing.py`. The seed assertions cover the
   flagship Anthropic rates and the published batch and cache multipliers, so a transcription slip
   in those rows fails the suite.
