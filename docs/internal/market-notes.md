# Market notes — the only place external figures live

Card `G1-T01`. This file is the single source of truth for every number about the outside world
that BranchPilot cites anywhere: README, `docs/`, comparison pages, landing page, cost calculator,
CLI output, talk slides.

## Rules of use

1. **No figure may be cited elsewhere without a row here.** If it is not in this file, it does not
   go in the README, the docs, the calculator, or a slide. Adding the number to this file *with its
   source* is the price of using it.
2. **Ranges are cited as ranges.** Where the market reports 20–45%, we write 20–45%. Collapsing a
   range to its top end is the marketing inflation this file exists to prevent.
3. **Primary vs `SECONDARY` is marked per row.** Primary = the party that sets or measures the
   number (provider pricing/docs page, repository API, published research). `SECONDARY` = anyone
   reporting on it. A `SECONDARY` row must be labeled `SECONDARY` at every citation site too, not
   just here.
4. **Every row carries an access date and a `VERIFY-BY` date 90 days out.** The access date is when
   the page was actually opened; nothing in this file was written from memory or from a search
   snippet without opening the page.
5. **Expired rows must be re-verified or removed before any external use.** A row past its
   `VERIFY-BY` is not "probably still fine" — it is unusable until someone re-opens the source and
   updates the dates. Provider pricing changed materially at least three times in the year before
   this file was written.
6. **Market-reported, never ours.** Every number in sections 1–4 describes the market or a third
   party. BranchPilot's own savings claims must come from `/evidence/` artifacts and the ledger, and
   must never be laundered through a row in this file.
7. **Unverifiable figures go in section 5, not in the text with a hedge.** "Reportedly", "up to",
   and "some teams see" are not sourcing.

Access dates below are `2026-08-27` for web pages and `2026-08-28` for GitHub/OpenRouter API
snapshots (the API clock had already rolled over to UTC 2026-08-28 when they were captured).
`VERIFY-BY` is exactly 90 days after the row's own access date.

---

## 1. Provider pricing and discount mechanics

### 1.1 OpenAI

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 1.1.1 | Batch API: **50% cost discount** vs synchronous, 24h completion window, separate higher rate-limit pool | <https://developers.openai.com/api/docs/guides/batch> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.2 | Prompt caching is **enabled by default** on supported models; cached input discounted **up to 90%** | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.3 | GPT-5.6+: cache **read = 0.1× uncached input**, cache **write = 1.25× uncached input** | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.4 | GPT-5.5 and earlier: **no cache-write charge**; cached-input rate is model-dependent | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.5 | Break-even math, quoted from the docs: one write + one full read = **1.35×** vs 2× uncached; one write + nine reads = **2.15×** vs 10× | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.6 | Minimum cacheable prefix: **1,024 visible input tokens** (GPT-5.6+), **2,048** (earlier) | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.7 | Cache lifetime: GPT-5.6+ **30 min** minimum after last write/reuse (`ttl` only accepts `"30m"`); earlier models `in_memory` ≈ **5–10 min** idle (up to 1h) or `24h` extended retention | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.8 | Caches are **machine-local**; traffic above **15 requests/minute** can overflow to another machine and miss. `prompt_cache_key` influences routing but **does not guarantee** a hit | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.9 | Caches are **not shared across organizations** and **cannot be reused across regional processing boundaries** | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.10 | Explicit mode allows **up to 4 cache writes per request**; cache reads consider **up to the latest 50 breakpoints** | <https://developers.openai.com/api/docs/guides/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.1.11 | Regional-processing (data residency) endpoints carry a **10% uplift** for models released on/after 2026-03-05 | <https://developers.openai.com/api/docs/pricing> | primary | 2026-08-27 | 2026-11-25 |

Per-1M-token rates, standard tier, short context (from the same pricing page, row 1.1.12):

| Model | Input | Cached input | Cache write | Output | Cached-read multiplier |
|---|---|---|---|---|---|
| `gpt-5.6-sol` | $4.00 | $0.40 | $5.00 | $20.00 | 0.10× |
| `gpt-5.6-terra` | $2.00 | $0.20 | $2.50 | $12.00 | 0.10× |
| `gpt-5.6-luna` | $0.20 | $0.02 | $0.25 | $1.20 | 0.10× |
| `gpt-5.5` (<272K ctx) | $5.00 | $0.50 | — | $30.00 | 0.10× |
| `gpt-5.4` (<272K ctx) | $2.50 | $0.25 | — | $15.00 | 0.10× |
| `gpt-5.1` / `gpt-5` | $1.25 | $0.125 | — | $10.00 | 0.10× |
| `gpt-4.1` | $2.00 | $0.50 | — | $8.00 | 0.25× |
| `gpt-4o` | $2.50 | $1.25 | — | $10.00 | 0.50× |

Row 1.1.12 — source <https://developers.openai.com/api/docs/pricing>, primary, accessed
2026-08-27, VERIFY-BY 2026-11-25.

Two caveats that must travel with row 1.1.12 whenever it is cited:

- `gpt-5.6-sol` carries **promotional pricing available at least through 2026-11-21** per the same
  page. Any calculator using it must label it promotional.
- The cached-read multiplier is **not a constant across the catalogue**: 0.10× on GPT-5.x, 0.25× on
  `gpt-4.1`, 0.50× on `gpt-4o`. "OpenAI caching saves 90%" is true for GPT-5.x and false for
  `gpt-4o`. A price book must key the multiplier per model, never per provider.

### 1.2 Anthropic

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 1.2.1 | Batch API: **50% discount on both input and output** tokens | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.2 | Prompt cache multipliers: **5-minute write 1.25×**, **1-hour write 2×**, **read 0.1×** of base input | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.3 | Caching pays off after **one** read at the 5-minute TTL, **two** reads at the 1-hour TTL | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.4 | Default cache lifetime **5 minutes**, refreshed at no extra cost on each use; lifetime is measured **from the start of the request**, so a 4-minute streamed response leaves ≈1 minute of reuse window | <https://platform.claude.com/docs/en/build-with-claude/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.5 | Caching covers the full prefix in the order `tools`, `system`, `messages`, up to the `cache_control` block | <https://platform.claude.com/docs/en/build-with-claude/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.6 | Batch limits: **100,000 requests or 256 MB**, most batches finish **< 1 hour**, results readable when all complete **or after 24h**, batches **expire at 24h**, results downloadable for **29 days** | <https://platform.claude.com/docs/en/build-with-claude/batch-processing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.7 | Batch requests **may exceed the configured workspace spend limit** because of concurrent processing | <https://platform.claude.com/docs/en/build-with-claude/batch-processing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.8 | `stream: true`, `speed` (fast mode), threads, `cache_hint`/`context_hint`, and `max_tokens: 0` are **rejected** inside a batch | <https://platform.claude.com/docs/en/build-with-claude/batch-processing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.9 | `inference_geo: "us"` (data residency) applies a **1.1× multiplier to every token category**, including cache writes and reads, on Claude 4.6+ | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |
| 1.2.10 | Claude 4.7+ use a newer tokenizer producing **approximately 30% more tokens for the same text** | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |

Per-1M-token rates (row 1.2.11):

| Model | Base input | 5m cache write | 1h cache write | Cache read | Output | Batch input | Batch output |
|---|---|---|---|---|---|---|---|
| Claude Fable 5 | $10 | $12.50 | $20 | $1 | $50 | $5 | $25 |
| Claude Opus 5 | $5 | $6.25 | $10 | $0.50 | $25 | $2.50 | $12.50 |
| Claude Opus 4.5–4.8 | $5 | $6.25 | $10 | $0.50 | $25 | $2.50 | $12.50 |
| Claude Sonnet 5 | $2 | $2.50 | $4 | $0.20 | $10 | $1 | $5 |
| Claude Sonnet 4.5 / 4.6 | $3 | $3.75 | $6 | $0.30 | $15 | $1.50 | $7.50 |
| Claude Haiku 4.5 | $1 | $1.25 | $2 | $0.10 | $5 | $0.50 | $2.50 |

Row 1.2.11 — source <https://platform.claude.com/docs/en/about-claude/pricing>, primary, accessed
2026-08-27, VERIFY-BY 2026-11-25. The same page states that Claude Sonnet 5's $2/$10 introductory
pricing **became the standard price** and the scheduled 2026-09-01 increase to $3/$15 will not
occur — a price book must not carry a stale scheduled-increase rule.

### 1.3 Do the discounts stack?

**Yes, on both providers, and both confirm it on a primary page.**

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 1.3.1 | Anthropic, explicit sentence: the cache multipliers "**stack with other pricing modifiers, including the Batch API discount and data residency**" | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |
| 1.3.2 | OpenAI, demonstrated by the rate card: the Batch tab publishes its own **cached-input and cache-write columns at exactly 50% of the standard tab** (`gpt-5.6-sol` $0.40 → $0.20 cached input; $5.00 → $2.50 cache write) | <https://developers.openai.com/api/docs/pricing> | primary | 2026-08-27 | 2026-11-25 |
| 1.3.3 | OpenAI's Batch **guide** contains no statement about caching either way — searched, zero mentions of "cache". The stacking claim rests on the rate card (1.3.2), not the guide | <https://developers.openai.com/api/docs/guides/batch> | primary | 2026-08-27 | 2026-11-25 |
| 1.3.4 | Anthropic recommends the **1-hour cache TTL for batches**, since batches routinely exceed the 5-minute default | <https://platform.claude.com/docs/en/build-with-claude/batch-processing> | primary | 2026-08-27 | 2026-11-25 |
| 1.3.5 | Anthropic **fast mode is not available with the Batch API**; caching and data-residency multipliers *do* stack on top of fast mode | <https://platform.claude.com/docs/en/about-claude/pricing> | primary | 2026-08-27 | 2026-11-25 |

Derived, and safe to cite because it is arithmetic over rows 1.1.3, 1.2.2, 1.3.1 and 1.3.2 rather
than a market claim: **on the repeated portion of a prompt, batch + cache-read is 0.5 × 0.1 = 0.05×
the standard synchronous input rate, i.e. 95% off.** Cite it as arithmetic, with the two rows, and
never as a savings estimate for a whole workload — it applies only to input tokens that are both
cache-read *and* batch-eligible.

### 1.4 Cross-provider cache multipliers — `SECONDARY`

One aggregator, opened and read, useful only for providers whose own pricing pages are not cited
here. Everything in this table is `SECONDARY` and must be labeled so.

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 1.4.1 | Google: cache read **0.25×**; minimum cacheable 4,096 tokens (2.5 Pro) / 1,024 (2.5 Flash) | <https://openrouter.ai/docs/guides/best-practices/prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 1.4.2 | DeepSeek: cache read **0.1×**, writes free | <https://openrouter.ai/docs/guides/best-practices/prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 1.4.3 | Grok: cache read **0.25×**, writes free. Moonshot: **0.25×**, writes free. Groq: **0.5×**, writes free | <https://openrouter.ai/docs/guides/best-practices/prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 1.4.4 | Alibaba Qwen: explicit caching only, write **1.25×**, read **0.1×**, 5-minute TTL | <https://openrouter.ai/docs/guides/best-practices/prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 1.4.5 | **Disagreement worth recording:** this page states OpenAI cache reads are "0.25x or 0.50x" of input. OpenAI's own rate card (row 1.1.12) shows **0.10×** for the entire GPT-5.x family. Where an aggregator and a provider disagree, the provider wins and the aggregator is stale | <https://openrouter.ai/docs/guides/best-practices/prompt-caching> vs <https://developers.openai.com/api/docs/pricing> | `SECONDARY` vs primary | 2026-08-27 | 2026-11-25 |

Row 1.4.5 is the reason the price book must read provider pages, not aggregators.

### 1.5 Open-weight price dispersion — computed, primary

Lever 5 ("provider arbitrage on open weights") needs a real number rather than the phrase "large
price dispersion". Computed directly from OpenRouter's public endpoints API — identical model
weights, different inference providers, input price per 1M tokens.

| # | Model | Priced endpoints | Cheapest | Dearest | Input spread | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|---|---|---|---|
| 1.5.1 | `openai/gpt-oss-120b` | 20 | AkashML $0.03 | Cerebras $0.35 | **11.7×** | <https://openrouter.ai/api/v1/models/openai/gpt-oss-120b/endpoints> | primary | 2026-08-28 | 2026-11-26 |
| 1.5.2 | `deepseek/deepseek-v3.2` | 14 | GMICloud $0.209 | SambaNova $3.00 | **14.4×** | <https://openrouter.ai/api/v1/models/deepseek/deepseek-v3.2/endpoints> | primary | 2026-08-28 | 2026-11-26 |
| 1.5.3 | `qwen/qwen3.5-397b-a17b` | 10 | Alibaba $0.39 | Venice $0.75 | **1.9×** | <https://openrouter.ai/api/v1/models/qwen/qwen3.5-397b-a17b/endpoints> | primary | 2026-08-28 | 2026-11-26 |

Rows 1.5.1–1.5.3 are primary in the sense that matters here: the marketplace's own live rate data
for each listed inference provider, read from its public API rather than from a write-up.

Cite as: **1.9×–14.4× input-price dispersion for identical open weights across 10–20 providers**,
never as a single headline multiple. Dispersion is per-model and moves weekly; the honest framing
is "dispersion exists and is large on some models", not "you will save 14×". Providers also differ
in throughput, context limit, quantisation and privacy terms, none of which this row measures.

---

## 2. Lever effectiveness ranges

These are the contested numbers. Every row records the range **and** who disagrees.

### 2.1 The inflation to name explicitly

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 2.1.1 | The vendor slide figure (~90–95%) refers to **match accuracy when a hit is found, not how often a hit is found**. Production hit rates are **20–45%** | <https://tianpan.co/blog/2026-04-09-semantic-caching-llm-production> | `SECONDARY` | 2026-08-27 | 2026-11-25 |

Row 2.1.1 is the single most important row in this file for honesty purposes. Any BranchPilot page
that mentions semantic cache hit rates must state the accuracy-vs-frequency distinction, because
the entire category conflates them. It is `SECONDARY` — a named practitioner blog, not a
measurement we or a provider made — and must be labeled `SECONDARY` at the citation site.

### 2.2 Semantic cache hit rate

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 2.2.1 | High-repetition categories **40–60%**; low-repetition or volatile categories **5–15%**; **20–30% of production traffic** is left uncached entirely because remote vector search (30 ms) needs a 15–20% hit rate to break even | <https://arxiv.org/abs/2510.26835> | primary (published research) | 2026-08-27 | 2026-11-25 |
| 2.2.2 | By workload: FAQ/support **40–60%**, classification **50–70%**, RAG Q&A **15–25%**, open-ended chat **10–20%**, agentic tool calls **5–15%**. Blended realistic figure for a mixed system: **~25%** | <https://tianpan.co/blog/2026-04-09-semantic-caching-llm-production> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.2.3 | Research papers report **60–70%**; production is lower. Exact (non-semantic) cache layer alone catches **15–30%** of traffic in most production systems | <https://tianpan.co/blog/2026-04-09-semantic-caching-llm-production> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.2.4 | An 87.5% hit rate was measured on a **deliberately favourable** benchmark (10 topics × 10 paraphrases); the same author states real distributions have a long tail where **60–70% of queries are unique**, and cites an EdTech deployment at **45.1%** on real student queries | <https://dev.to/vinay_budideti/i-built-a-semantic-cache-that-cuts-llm-api-costs-by-72-what-actually-worked-and-what-didnt-19ia> | `SECONDARY` | 2026-08-27 | 2026-11-25 |

**The disagreement, stated plainly.** 2.2.1 (research, category-resolved) and 2.2.2 (practitioner,
workload-resolved) agree closely: high-repetition workloads land in the 40–60% band, agentic and
open-ended traffic land in the 5–20% band. 2.2.3 records that *research settings* run 60–70% —
higher than either. 2.2.4 shows how an 87.5% number gets produced: a benchmark with 10× designed
overlap. The range is real, the spread is driven by **workload composition**, not by
implementation quality, and no single number is defensible for an unseen workload. This is why
BranchPilot must measure hit rate on the user's own logs (`M1-*`) rather than quote any of these.

### 2.3 Semantic cache cost reduction

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 2.3.1 | **71.8%** cost reduction ($0.24 vs $0.87) on 100 real Anthropic calls — but on the favourable benchmark of 2.2.4, and achieved partly by *adapting* cached answers with a cheaper model (35 of 100 queries), which is a **quality-affecting** lever, not a response-identical one | <https://dev.to/vinay_budideti/i-built-a-semantic-cache-that-cuts-llm-api-costs-by-72-what-actually-worked-and-what-didnt-19ia> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.3.2 | A 20–25% hit rate on a $5,000/month bill saves **roughly $1,000/month** before cache infrastructure cost | <https://tianpan.co/blog/2026-04-09-semantic-caching-llm-production> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.3.3 | Vendor claim: managed semantic caching "can reduce costs by **up to 90%**" and give "~15× speedup in some workloads" — a vendor selling the product, with no workload definition | <https://redis.io/blog/prompt-caching-vs-semantic-caching/> | `SECONDARY` (vendor) | 2026-08-27 | 2026-11-25 |

**The disagreement.** 2.3.3 (90%, vendor) versus 2.3.2 (~20%, derived from a realistic hit rate)
is a 4.5× gap on the same lever. 2.3.1 sits in between only because it silently mixes in model
downgrading. Cost reduction from a response-identical cache cannot exceed its hit rate, so any
figure above the hit-rate band is either quality-affecting, benchmark-favourable, or both. Cite
this lever as **"reduction is bounded above by hit rate; reported production range 20–45% of cached
traffic"**, and keep quality-affecting variants in a separate column, per masterplan `M8-T01`.

### 2.4 Prefix / prompt cache effectiveness (lever 1)

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 2.4.1 | Named company (ProjectDiscovery, agent platform, 20–40 LLM steps/task): cache hit rate went **7% → 84%**, overall cost **−59%**, post-optimisation **−66%**, trailing 10 days **−70%**, 9.8B tokens served from cache. Derived from **actual reported spend**, not estimated pricing | <https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.4.2 | Weekly progression in the same deployment: 4.2% → 7.6% → **73.7%** (the week the fix shipped) → 78.2% → 84.3% → 85.0% | <https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.4.3 | Cache rate **scales with task complexity**: 1-step tasks averaged 35.5%, 2–3-step tasks 30.0%, long tasks far higher — caching helps the most expensive tasks most | <https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |
| 2.4.4 | The single highest-impact fix was **relocating dynamic content out of the cached prefix** (working memory, skills, runtime context sitting mid-prefix "was silently killing our cache hits"). Also required: stable template placeholders, frozen date-only datetime, and pinning traffic to one provider because **caches are provider-specific** (Anthropic Direct vs Bedrock vs Vertex do not share) | <https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching> | `SECONDARY` | 2026-08-27 | 2026-11-25 |

Row 2.4.4 is the direct market evidence for masterplan card `M2-T02` (volatile-prefix detector):
a competent team ran at **7%** cache rate without knowing why, and a prefix-layout fix moved them
to 74% overnight. "Prefix caching silently not working" is the most valuable thing an audit can
detect, and 2.4.1's own numbers are what make that case — not our assertion.

Cite 2.4.1 as **one named deployment**, never as a general expectation. It is `SECONDARY`, n=1,
and self-reported, though it states its methodology (effective rate derived from real spend
compared against the same volume at standard rates) more rigorously than most.

### 2.5 Model-routing cost reduction

| # | Figure | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|
| 2.5.1 | RouteLLM (the paper the category cites): routing between a strong and weak model "**significantly reduces costs — by over 2 times in certain cases** — without compromising the quality of responses", with demonstrated transfer across model pairs | <https://arxiv.org/abs/2406.18665> | primary (published research) | 2026-08-27 | 2026-11-25 |
| 2.5.2 | Maintenance reality check on the reference implementation: `lm-sys/RouteLLM` is Apache-2.0, **5,410 stars**, and its last push was **2024-08-10** — roughly two years untouched at access date | <https://api.github.com/repos/lm-sys/RouteLLM> | primary | 2026-08-28 | 2026-11-26 |

**The disagreement, and what we may not say.** The widely repeated "**up to 85% cost reduction
while maintaining 95% of GPT-4 performance**" figure is a *repository README / benchmark-specific*
claim, benchmark-dependent (MT Bench being the most favourable of the three benchmarks). The
abstract of the paper itself — the primary source we opened — commits only to "**over 2×** in
certain cases". We cite **2× (paper abstract, primary)** and treat 85% as unverified; see row
5.4. The masterplan's "~20–40% commonly reported in week one" is also unsourced; see row 5.5.

Additional honesty note for any routing page: routing is a **quality-affecting** lever. Per
masterplan `M4-T01` / `M8-T01` its savings must be reported separately from response-identical
levers and must not appear in a headline number.

---

## 3. Competitor matrix

Star counts, licenses and activity are from the GitHub REST API at access date; capability claims
are from each project's own documentation. Every row's sources are listed under the table.

| | LiteLLM | Portkey Gateway | Helicone | Cloudflare AI Gateway | Kong AI Gateway | OpenRouter |
|---|---|---|---|---|---|---|
| License | MIT, **except `enterprise/`** which is separately licensed | MIT | Apache-2.0 | Proprietary SaaS | Apache-2.0 core; AI plugins tiered | Proprietary SaaS |
| Self-hostable | Yes | Yes | Yes | **No** | Yes (on-prem or Konnect) | **No** |
| Stars (2026-08-28) | **57,446** | **12,842** | **6,106** | n/a (no public repo) | **44,052** (`Kong/kong`, whole gateway) | n/a (no public repo) |
| Providers / models | "100+ LLMs" (own README) | "1,600+ LLMs" (own repo description) | Multi-provider proxy | OpenAI, Anthropic, Google, Workers AI, Replicate, more | OpenAI, Anthropic, Azure AI, more | **103 providers, 380 models** (own API) |
| Semantic cache | **Yes** — Qdrant, Redis, Valkey semantic caches | **Enterprise-only** (or self-hosted with Milvus/Pinecone); default threshold 0.95 | Not documented as a feature | **No** — exact match only; docs say semantic search is planned | **Yes** — but `tier: ai_gateway_enterprise` | Not offered; ships **provider sticky routing** to keep upstream prompt caches warm |
| Routing | Yes (router, fallbacks, semantic routing) | Yes (configs, fallbacks, load balancing) | Routing + multi-provider failover (per acquirer's announcement) | Retry + model fallback ("dynamic routing") | Yes, incl. **semantic routing** and load balancing | Yes (Auto Router, Pareto Router, provider order) |
| Spend tracking | **Yes** — per key/user/team, `x-litellm-response-cost` header, model cost map, `/spend` endpoints | Yes (observability suite) | Yes (cost + analytics; that was the product) | Yes — request, token and cost analytics | Yes — LLM metrics, metering & billing via Konnect | Yes (activity, generation API, usage accounting) |
| Counterfactual savings measurement | **None** | **None** | **None** | **None** | **None** | **None** |
| Maintenance status | Very active — release `v1.100.0-dev.2` dated 2026-08-28; **1,645** contributor pages | **Stale**: last push 2026-05-25, last release `v1.15.2` 2026-01-12 | **Maintenance mode** — acquired by Mintlify 2026-03-03; security updates, new models, bug/perf fixes only | Actively maintained (caching doc last updated 2026-08-27) | Actively maintained (last push 2026-08-16) | Actively maintained (live catalogue) |

Sources for the matrix, all opened at access date:

| Row scope | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|
| LiteLLM stars/activity/releases | <https://api.github.com/repos/BerriAI/litellm> | primary | 2026-08-28 | 2026-11-26 |
| LiteLLM license (MIT except `enterprise/`) | <https://raw.githubusercontent.com/BerriAI/litellm/main/LICENSE> | primary | 2026-08-28 | 2026-11-26 |
| LiteLLM "100+ LLMs" | <https://raw.githubusercontent.com/BerriAI/litellm/main/README.md> | primary | 2026-08-28 | 2026-11-26 |
| LiteLLM caches (incl. Qdrant/Redis/Valkey semantic) | <https://docs.litellm.ai/docs/proxy/caching> | primary | 2026-08-27 | 2026-11-25 |
| LiteLLM spend tracking | <https://docs.litellm.ai/docs/proxy/cost_tracking> | primary | 2026-08-27 | 2026-11-25 |
| Portkey stars/license/activity/releases/"1,600+" | <https://api.github.com/repos/Portkey-AI/gateway> | primary | 2026-08-28 | 2026-11-26 |
| Portkey cache tiering, 0.95 threshold, TTLs | <https://portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic> | primary | 2026-08-27 | 2026-11-25 |
| Helicone stars/license/activity | <https://api.github.com/repos/Helicone/helicone> | primary | 2026-08-28 | 2026-11-26 |
| Helicone maintenance mode, acquisition date, scale | <https://www.helicone.ai/blog/joining-mintlify> | primary | 2026-08-27 | 2026-11-25 |
| Cloudflare features, plans, providers | <https://developers.cloudflare.com/ai-gateway/> | primary | 2026-08-27 | 2026-11-25 |
| Cloudflare exact-match-only cache, key composition, TTLs | <https://developers.cloudflare.com/ai-gateway/features/caching/> | primary | 2026-08-27 | 2026-11-25 |
| Kong capabilities, providers, Konnect features | <https://developer.konghq.com/ai-gateway/> | primary | 2026-08-27 | 2026-11-25 |
| Kong semantic cache enterprise tier, vector DBs, headers | <https://developer.konghq.com/plugins/ai-semantic-cache/> | primary | 2026-08-27 | 2026-11-25 |
| Kong stars/license/activity | <https://api.github.com/repos/Kong/kong> | primary | 2026-08-28 | 2026-11-26 |
| OpenRouter 103 providers / 380 models | <https://openrouter.ai/api/v1/providers>, <https://openrouter.ai/api/v1/models> | primary | 2026-08-28 | 2026-11-26 |
| OpenRouter sticky routing, cache handling | <https://openrouter.ai/docs/guides/best-practices/prompt-caching> | primary | 2026-08-27 | 2026-11-25 |

### 3.1 Where each competitor is genuinely better than BranchPilot

This is the useful part of the table. BranchPilot's current state is taken from `docs/MASTERPLAN.md`
Part 1.3 (verified against commit `5eedf26`): OpenAI-shaped upstreams only, streaming rejected,
no cache lever shipped, no batch lane, no container, no visual surface.

- **LiteLLM** — better on essentially every axis of breadth and adoption *today*: 57,446 stars vs
  our zero-adoption position, 100+ providers vs our OpenAI-shaped-only, 1,645 contributor pages,
  a shipped release the same day we measured, semantic caching against three vector backends, and
  spend tracking with virtual keys and per-key/team/user budgets that already work. If someone
  needs a working multi-provider gateway this afternoon, LiteLLM is the correct answer and we
  should say so. Its gap is narrow and specific: it tells you what you **spent**, never what you
  **would have** spent, and its footguns (end users can self-declare `user` and dodge spend
  attribution — documented as a warning in its own cost-tracking page) are disclaimed to the
  operator rather than closed in code.
- **Portkey Gateway** — better on provider breadth by a wide margin (1,600+ LLMs claimed) and on
  guardrails, which we do not have at all. Its semantic cache has real engineering behind it
  (Milvus/Pinecone backends, configurable threshold, cache namespaces, org-level TTL policy). Gap:
  semantic caching is **Enterprise-gated** on the hosted product, and the OSS repo has not been
  pushed since 2026-05-25 nor released since 2026-01-12. Also a documented correctness footgun we
  intend to close rather than document: **the system prompt is ignored when matching semantic cache
  hits**, so two requests with different system prompts can share a cached answer.
- **Helicone** — better at observability UX and at proving the market: 14.2 trillion tokens and
  16,000 organisations in three years, from a one-line integration. That is the adoption curve we
  want and have not earned. Gap: it is in maintenance mode since 2026-03-03 — security updates,
  new models and bug fixes only — so the actively-maintained OSS-observability slot is genuinely
  open.
- **Cloudflare AI Gateway** — better on operational effort by a distance no self-hosted tool can
  match: available on all plans, one line of code, no infrastructure, global edge, and analytics
  including cost. Its cache key includes the provider auth header by default, which is a **safer
  tenant-isolation default than most self-hosted caches**, and it is honest that caching is
  disabled by default and exact-match only. Gaps: not self-hostable, no semantic cache (documented
  as planned), volatile cache with no simultaneous-request coalescing, and your traffic transits a
  third party.
- **Kong AI Gateway** — better for anyone who already runs Kong, and better than us on governance
  breadth we do not attempt: PII sanitisation across 20 categories and 9 languages, prompt
  guards, RAG injection, MCP and A2A traffic governance, canary release, audit logging, plus both
  semantic caching and semantic routing with pgvector/Redis/Valkey backends. Gap: the AI plugins
  that matter for cost are `tier: ai_gateway_enterprise`, it is an enterprise API-platform product
  rather than a developer-first one, and it documents rather than closes a real footgun — "as most
  AI services always send `no-cache` in the response headers, setting `cache_control` to `true`
  will always result in a cache bypass."
- **OpenRouter** — better than us at the thing lever 5 needs: 103 providers and 380 models behind
  one key, with live per-endpoint pricing, plus **provider sticky routing** (account/model/
  conversation-scoped, 10-minute idle expiry, `session_id` override) that is a genuinely clever
  answer to a problem we have only written a card about — it activates only when the provider's
  cache read price is actually cheaper. It also normalises cache markers across providers. Gaps:
  proprietary, not self-hostable, your traffic and prompts transit a third party, and it optimises
  routing rather than measuring counterfactual savings.

### 3.2 The one column where every row says "None"

No competitor in the matrix measures **counterfactual savings** — what the traffic would have cost
without the lever. Every one of them reports actual spend. This claim is a **negative**, established
by reading each project's own spend/analytics documentation (rows in the source table above) and
finding no such feature; it is not proof of absence. Before this appears on a comparison page it
must be phrased as: *"we found no counterfactual-savings measurement documented in any of the six
as of 2026-08-27; corrections welcome"* — and re-checked at `VERIFY-BY`.

---

## 4. Star-count comparators

Grounding for any reach or ambition discussion. All from the GitHub REST API, accessed 2026-08-28,
VERIFY-BY 2026-11-26.

| # | Repository | Stars | License | Surface it ships | Source | Type | Accessed | VERIFY-BY |
|---|---|---|---|---|---|---|---|---|
| 4.1 | `n8n-io/n8n` | 202,653 | Other (fair-code) | End-user app — visual workflow automation canvas | <https://api.github.com/repos/n8n-io/n8n> | primary | 2026-08-28 | 2026-11-26 |
| 4.2 | `ollama/ollama` | 179,598 | MIT | Model runtime + CLI/app — local model library | <https://api.github.com/repos/ollama/ollama> | primary | 2026-08-28 | 2026-11-26 |
| 4.3 | `langgenius/dify` | 153,710 | Other | End-user app — visual agent/RAG builder | <https://api.github.com/repos/langgenius/dify> | primary | 2026-08-28 | 2026-11-26 |
| 4.4 | `open-webui/open-webui` | 150,180 | Other | End-user app — chat UI | <https://api.github.com/repos/open-webui/open-webui> | primary | 2026-08-28 | 2026-11-26 |
| 4.5 | `langchain-ai/langchain` | 145,162 | MIT | Developer framework — library, no end-user surface | <https://api.github.com/repos/langchain-ai/langchain> | primary | 2026-08-28 | 2026-11-26 |
| 4.6 | `Comfy-Org/ComfyUI` | 130,349 | GPL-3.0 | End-user app — visual node graph for image generation | <https://api.github.com/repos/Comfy-Org/ComfyUI> | primary | 2026-08-28 | 2026-11-26 |

Reference points from section 3 for scale: LiteLLM **57,446**, Kong (entire gateway, not just AI)
**44,052**, Portkey Gateway **12,842**, Helicone **6,106**.

**What this table honestly supports.** Five of the six repositories above 100k stars ship a surface
a non-developer can open and use; the exception (4.5) is the single most widely used framework in
the category and reached its position through ubiquity as a dependency, not through a UI. The
infra-gateway ceiling in this sample is LiteLLM at 57k — roughly one third of the app tier. That is
evidence for the claim "a visual surface correlates with six-figure reach in this era", and it is
**correlational on a hand-picked sample of six**. It is not evidence that building a cockpit
produces 100k stars, and it must never be cited to that effect. `n = 6`, selected because they were
already known to exceed 100k, which is textbook selection bias — stated here so nobody cites it as
a study.

---

## 5. Unverified / omitted

Figures that were sought and are **not** available for use. Each names the reason. None of these
may appear in the README, docs, calculator, or a slide until it has a sourced row in section 1–4.

| # | Figure | Where it currently appears | Why it is omitted |
|---|---|---|---|
| 5.1 | "One documented agent: **$720/mo → $72/mo** via three `cache_control` markers" | `docs/MASTERPLAN.md` Part 1.2, lever 1 | The originating article (`labeveryday.medium.com`, Sept 2025) returned **HTTP 403** on access at 2026-08-27; it could not be opened, so per rule 4 it cannot be cited. It is also n=1, a personal project, and predates two Anthropic pricing revisions. **Replace with row 2.4.1**, which is a named company reporting from actual spend and is verifiable. Remove from the masterplan text at the next edit of that file. |
| 5.2 | "Reported reduction **20–73%**" for response caching | `docs/MASTERPLAN.md` Part 1.2, lever 3 | The 73% end traces to a VentureBeat article that returned **HTTP 429** on access at 2026-08-27 and was never opened. The closest figure that was opened is 71.8% (row 2.3.1), which is benchmark-favourable and partly quality-affecting. Cite rows 2.3.1–2.3.3 with their caveats instead of a bare range. |
| 5.3 | "OpenAI auto-caching **~50%**" | `docs/MASTERPLAN.md` Part 1.2, lever 1 | **Verified false as a general statement** and corrected rather than omitted: OpenAI's cached-read multiplier is model-specific — 0.10× on the entire GPT-5.x family, 0.25× on `gpt-4.1`, 0.50× only on `gpt-4o` (row 1.1.12). "~50%" describes one legacy model. Use row 1.1.12 and key the multiplier per model. |
| 5.4 | Model routing "**up to 85%** cost reduction while maintaining 95% of GPT-4 performance" | not currently in repo docs; ubiquitous in the category | The primary paper's abstract (row 2.5.1) commits only to "over 2× in certain cases". The 85% figure is a repository/benchmark claim tied to MT Bench specifically, and the reference implementation has not been pushed since 2024-08-10 (row 2.5.2). Cite 2× from the paper; do not cite 85%. |
| 5.5 | Model tier routing "**~20–40%** commonly reported in week one" | `docs/MASTERPLAN.md` Part 1.2, lever 4 | No source found that reports a week-one production range. The MMLU/GSM8K figures floating around the category (45%/35%) come from secondary write-ups of the RouteLLM benchmarks, not from production deployments, and the pages carrying them were not opened. **No usable row exists for lever 4's magnitude** — say "routing savings depend on the easy/hard mix in your traffic and must be measured", and let the audit produce the number. |
| 5.6 | LiteLLM "~40k stars, 1,300+ contributors" | `docs/MASTERPLAN.md` Part 1.4 | Stale, not wrong-in-kind. Actual at 2026-08-28: **57,446 stars**; contributor listing paginates to **1,645** pages at one per page (including anonymous), so "1,300+" understates it. Use the matrix in section 3. |
| 5.7 | Portkey "**Apache 2.0** since Mar 2026" | `docs/MASTERPLAN.md` Part 1.4 | **Contradicted by the source.** The GitHub API reports the license as **MIT** (SPDX `MIT`, `LICENSE` on `main`) at 2026-08-28. No evidence of an Apache-2.0 relicense was found. Use MIT, and drop the date claim entirely — no source for it was located. |
| 5.8 | Anthropic / OpenAI enterprise or committed-use discount rates | nowhere | Not published. Both providers reference negotiated discounts and private offers without rates. Any calculator must treat list price as the ceiling and say so; it must never model a discount we cannot source. |
| 5.9 | Total LLM API market size, spend growth, or share | nowhere | Deliberately not researched. Market sizing is an explicit non-goal of card `G1-T01`, and no such figure is needed to answer "does this help me?". |
| 5.10 | Gemini / Vertex and Bedrock first-party per-token rates | nowhere | Only reached through an aggregator this pass (rows 1.4.1–1.4.4, `SECONDARY`). Before any Gemini or Bedrock figure is used in the price book or a doc, open `cloud.google.com/vertex-ai/generative-ai/pricing` and `aws.amazon.com/bedrock/pricing/` and add primary rows here. |
| 5.11 | Helicone semantic caching support | section 3 shows "not documented as a feature" | Its own docs were not opened for this specific capability; the negative is from the acquisition announcement and repo metadata only. Confirm before asserting absence on a comparison page. |

---

## 6. Corrections this file makes to `docs/MASTERPLAN.md`

Recorded here so the next editor of that file has a checklist. This file does not edit the
masterplan; `G1-T02`/`G1-T03` must apply these.

1. Part 1.2, lever 1 — remove the `$720 → $72` example (row 5.1); replace with row 2.4.1.
2. Part 1.2, lever 1 — replace "OpenAI auto-caching ~50%" with the per-model multipliers, row 1.1.12
   (row 5.3).
3. Part 1.2, lever 2 — "stacks with caching to ~95% off the repeated portion" is **confirmed** and
   now has primary sourcing (rows 1.3.1, 1.3.2 and the arithmetic note in §1.3).
4. Part 1.2, lever 3 — replace "20–73%" with rows 2.3.1–2.3.3 and their caveats (row 5.2).
5. Part 1.2, lever 3 — "production semantic hit rates 20–45% (not the 90% marketing figure)" is
   **confirmed** (rows 2.1.1, 2.2.1, 2.2.2), and the reason the 90% figure exists is now on the
   record: it is match accuracy, not hit frequency.
6. Part 1.2, lever 4 — "~20–40% commonly reported in week one" has **no source**; drop it (row 5.5).
7. Part 1.2, lever 5 — "large price dispersion for identical weights" now has a computed figure:
   1.9×–14.4× (rows 1.5.1–1.5.3).
8. Part 1.4 — LiteLLM is 57,446 stars, not ~40k (row 5.6); Portkey is MIT, not Apache-2.0, and its
   OSS repo is stale (row 5.7); Helicone maintenance mode is **confirmed** with a primary source and
   an exact date, 2026-03-03 (section 3 source table).
9. Part 1.4 — the reach paragraph's examples check out, with the caveat that the sample is
   selection-biased; see the note under section 4.
10. All four `[VERIFY]` markers in Part 1.4 are now resolved by section 3 and may be removed.
