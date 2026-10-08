# Core and secondary tools

BranchPilot's core is a bounded loop that decides whether to request another language-model sample. The learned controller is one strategy for that loop, alongside fixed counts and two answer-agreement rules. The gateway, spend tools, importers, and databases are secondary applications, not prerequisites for the GSM8K comparison.

This is a documentation boundary, not a package migration. Existing module paths and APIs remain unchanged. Some secondary modules ship in the base package; "secondary" here describes their role, not a Python installation extra.

## Start with the core

| Part | Files | What it does |
|---|---|---|
| Runtime | [`runtime.py`](../src/branchpilot/runtime.py) | `PilotSession` accepts one observation at a time, calls the strategy, and runs a synchronous or asynchronous sampler until STOP. |
| Strategies | [`strategies.py`](../src/branchpilot/strategies.py) | `StoppingStrategy` defines the observed-prefix interface. Fixed count, leading vote share, and consecutive agreement require no training. |
| Observations and answers | [`schema.py`](../src/branchpilot/schema.py), [`answers.py`](../src/branchpilot/answers.py), [`features.py`](../src/branchpilot/features.py) | `Sample` holds text, parsed answer, tokens, and optional log probability. `Rollout` adds a gold answer for offline work. Prefix features aggregate only observed samples. |
| Learner | [`training.py`](../src/branchpilot/training.py) | Optional PyTorch training builds pathwise targets from labeled logged trajectories and fits a small two-output network. |
| Learned inference | [`policy.py`](../src/branchpilot/policy.py) | NumPy inference compares STOP/CONTINUE scores; Safetensors stores the weights and feature normalization. |
| Evaluation | [`evaluate.py`](../src/branchpilot/evaluate.py) | Replays learned, fixed, and heuristic strategies on logged prefixes, scores final answers, and computes prompt-level paired bootstrap intervals. |
| Validation and deployment | [`calibration.py`](../src/branchpilot/calibration.py), [`deployment.py`](../src/branchpilot/deployment.py) | Selects a strategy from validation measurements for a sample budget and loads a JSON plan with a bound learned artifact when needed. |

The basic sequence is:

1. Request the first sample through a caller-supplied callback.
2. Pass a `Sample` to `PilotSession.observe`. The strategy receives the question, observed samples, prompt-token count, sample-cost parameter, and horizon—not a gold answer or future sample.
3. Continue by requesting one more sample, or stop and return the prefix's selected answer. Built-in strategies and the learned policy force STOP at the horizon. `run` and `run_async` do not call the sampler after STOP.

The selected answer comes from prefix voting, with ties resolved by available mean log probability and then first occurrence. Unparsed answers count as separate observations rather than a shared agreement. Feature inputs include vote share, margin, entropy, diversity, parse/log-probability coverage, completion lengths, and prompt structure. The learned policy also uses the remaining horizon and cost. Its continuation advantage is projected to be nonincreasing across the trained cost grid before interpolation.

The runtime does not generate model text itself. A STOP decision avoids requesting later samples; it does not undo generation that a server has already launched.

## What "exact backward Q regression" means

The stored algorithm identifier is `exact-backward-q-regression-v1`. It describes target construction on each complete logged trajectory, not a guarantee of optimal decisions from an observed prefix.

For a logged prefix ending at sample `t`, let `r_t` be 1 if its selected answer matches the gold answer and 0 otherwise. For horizon `H` and additional-sample penalty `lambda`, the target calculation in `training.py` is:

```text
V_H = r_H
Q_stop(t) = r_t
Q_continue(t) = -lambda + V_(t+1)       for t < H
V_t = max(Q_stop(t), Q_continue(t))
```

The terminal continuation value is stored as the terminal reward but masked out of continuation training. The network regresses these targets with Huber loss from observed-prefix features and cost/horizon inputs. It does not solve a conditional-expectation Bellman equation over possible future samples.

**Methodological limitation:** pathwise targets can use the logged future and future correctness labels. Choosing the best continuation along each realized future is different from choosing without seeing that future. Regressing these targets onto prefix features is not guaranteed to produce the optimal non-clairvoyant stopping rule. This is a target-construction limitation, not evidence that deployment reads future samples: `decide_observed` receives only the prefix.

Evaluation measures the fitted rule's actual stopping behavior on a fixed response bank. Utility is `accuracy - lambda * (samples - 1)`: the first sample is compulsory and only additional samples receive the penalty. `lambda` is not a dollar, token, energy, GPU-time, or latency price. Without supplied validation-selected baseline names, `benchmark` chooses its comparator on the same evaluation outcomes and labels that comparison exploratory.

## Secondary applications

| Tool | Files | Role and installation boundary |
|---|---|---|
| Gateway and provider translation | [`gateway/app.py`](../src/branchpilot/gateway/app.py), [`gateway/config.py`](../src/branchpilot/gateway/config.py), [`gateway/upstream.py`](../src/branchpilot/gateway/upstream.py), [`gateway/providers/__init__.py`](../src/branchpilot/gateway/providers/__init__.py) | Text-only, non-streaming HTTP serving with configured routes and sequential upstream requests. Provider adapters translate OpenAI, Anthropic, Gemini, and Bedrock formats. Server dependencies use the `gateway` extra. |
| In-process provider integrations | [`integrations/openai.py`](../src/branchpilot/integrations/openai.py), [`integrations/its_hub.py`](../src/branchpilot/integrations/its_hub.py) | Connect the loop to the OpenAI SDK or `its_hub`; use the `openai` or `its-hub` extra. |
| Pricing | [`pricing/book.py`](../src/branchpilot/pricing/book.py), [`pricing/prices.json`](../src/branchpilot/pricing/prices.json), [`pricing-data.md`](pricing-data.md) | Decimal rate-card arithmetic for traffic analysis, separate from the learner's sample penalty. |
| Spend audit and log ingestion | [`audit/detectors.py`](../src/branchpilot/audit/detectors.py), [`audit/risk.py`](../src/branchpilot/audit/risk.py), [`audit/render.py`](../src/branchpilot/audit/render.py), [`ingest/formats.py`](../src/branchpilot/ingest/formats.py) | Reads traffic logs and sizes opportunities under logged usage and configured rates. Those projections are not GSM8K accuracy or measured serving savings. |
| Configuration importer | [`importers/litellm.py`](../src/branchpilot/importers/litellm.py) | Translates LiteLLM YAML into gateway configuration, a fixed-one-sample plan, and migration notes; uses the `importers` extra. |
| Persistence | [`store/base.py`](../src/branchpilot/store/base.py), [`store/sqlite.py`](../src/branchpilot/store/sqlite.py), [`store/postgres.py`](../src/branchpilot/store/postgres.py) | Ledger, cache metadata, batch state, and rate-limit storage. SQLite uses the standard library; Postgres uses the `postgres` extra. These stores are not required by `PilotSession`. |
| Cache analysis | [`cache/prefix.py`](../src/branchpilot/cache/prefix.py) | Groups logged system prefixes, describes observed cache behavior, and produces priced cache plans. It is not the core sampling loop or proof of realized cache savings. |

[`pyproject.toml`](../pyproject.toml) defines the actual installation extras. The public core import does not import PyTorch or provider SDKs. The combined [`cli.py`](../src/branchpilot/cli.py) does import audit, ingestion, and pricing modules; documentation separation does not claim those CLI imports have been removed.

## Reproducing the GSM8K result

There are three different tasks:

- **Inspect or redraw the saved result on CPU.** [`gsm8k-v2.json`](../benchmarks/gsm8k-v2.json) contains metrics, paired comparisons, and per-policy prompt outcomes. [`gsm8k-v2-validation.json`](../benchmarks/gsm8k-v2-validation.json) and [`gsm8k-v2-baselines.json`](../benchmarks/gsm8k-v2-baselines.json) preserve validation measurements and comparator choices. [`report.py`](../src/branchpilot/report.py) renders the saved benchmark without a model, training, gateway, provider account, or database. For example: `nice -n 19 uv run branchpilot report --benchmark benchmarks/gsm8k-v2.json --svg artifacts/gsm8k-replay.svg`. This redraws existing evidence; it is not a new model run.
- **Retrain or replay from the original model responses.** This needs the labeled train/validation/test JSONL banks and, for inference replay, the learned Safetensors policy. Their paths and hashes are recorded in the benchmark and [`manifest-v2.json`](../benchmarks/manifest-v2.json). The policy is committed unchanged as [`gsm8k-policy.safetensors`](../benchmarks/math500/gsm8k-policy.safetensors) (same SHA-256 as `gsm8k-v2.json`), but the GSM8K response banks are not tracked in this checkout. A fresh clone alone therefore cannot rerun the original learner and response-level evaluation. Training additionally needs the `train` extra; NumPy policy inference does not.
- **Regenerate model responses.** [`protocol.json`](../benchmarks/protocol.json) specifies Qwen2.5-1.5B-Instruct, 1,600 training prompts, 400 validation prompts, the 1,319-question official test, and eight samples per prompt, with pinned model/data revisions and decoding settings. [`modal_app.py`](../modal_app.py) collects these with vLLM on an L4 GPU. That historical collection is not a CPU-only reproduction path and is not needed to redraw the saved result.

The v2 success criterion was not met: the primary-cost paired utility intervals in the test JSON do not satisfy the rule in the protocol. These are one-model, one-task, one-generation-seed response-bank results, not measurements of sequential serving latency or general superiority of the learned strategy.

For data checks and result provenance, see [`integrity.py`](../src/branchpilot/integrity.py), [`provenance.py`](../src/branchpilot/provenance.py), and [`checksums-v2.txt`](../benchmarks/checksums-v2.txt). They support the original evaluation workflow. Gateway/provider code, pricing, traffic audit/importers, persistence, and cache analysis are not needed to reproduce its accuracy/sample calculations.

The [earlier usage and study notes](earlier-usage.md) preserve the long-form gateway
examples, data contract, and original study commands. They are historical instructions;
GPU collection is not part of the CPU-only redraw described above.
