from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_PATH = "/root/.cache/huggingface"

app = modal.App("branchpilot-rollouts")
cache = modal.Volume.from_name("branchpilot-huggingface", create_if_missing=True)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "datasets==4.0.0",
        "vllm==0.10.2",
    )
    .add_local_python_source("branchpilot")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={CACHE_PATH: cache},
)
def collect_rollouts(
    train_size: int = 256,
    test_size: int = 128,
    max_samples: int = 8,
    seed: int = 17,
) -> dict[str, Any]:
    import time

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    from branchpilot.answers import extract_answer

    if train_size < 1 or test_size < 1 or max_samples < 1:
        raise ValueError("split sizes and max_samples must be positive")

    model = LLM(
        model=MODEL,
        download_dir=CACHE_PATH,
        dtype="half",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=1536,
        seed=seed,
    )
    tokenizer = model.get_tokenizer()
    sampling = SamplingParams(
        n=max_samples,
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        seed=seed,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    system_prompt = (
        "Solve the arithmetic word problem carefully. Show concise reasoning, then put only the "
        "final numeric value after '####'."
    )

    def collect_split(split: str, size: int) -> tuple[list[dict[str, Any]], int, float]:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        if size > len(dataset):
            raise ValueError(f"requested {size} records from a {len(dataset)}-record split")
        selected = dataset.select(range(size))
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["question"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in selected
        ]
        started = time.perf_counter()
        generated = model.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.perf_counter() - started
        completion_tokens = 0
        records: list[dict[str, Any]] = []
        for index, (row, request) in enumerate(zip(selected, generated, strict=True)):
            gold = extract_answer(row["answer"])
            if gold is None:
                raise ValueError(f"could not parse GSM8K gold answer at {split}:{index}")
            samples = []
            for output in request.outputs:
                token_count = len(output.token_ids)
                completion_tokens += token_count
                cumulative_logprob = output.cumulative_logprob
                mean_logprob = (
                    None
                    if cumulative_logprob is None or token_count == 0
                    else float(cumulative_logprob) / token_count
                )
                samples.append(
                    {
                        "text": output.text,
                        "answer": extract_answer(output.text),
                        "token_count": token_count,
                        "mean_logprob": mean_logprob,
                    }
                )
            records.append(
                {
                    "schema_version": 1,
                    "uid": f"gsm8k-{split}-{index}",
                    "question": row["question"],
                    "gold": gold,
                    "samples": samples,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "metadata": {
                        "dataset": "openai/gsm8k",
                        "split": split,
                        "index": index,
                        "model": MODEL,
                        "temperature": sampling.temperature,
                        "seed": seed,
                    },
                }
            )
        return records, completion_tokens, elapsed

    train, train_tokens, train_seconds = collect_split("train", train_size)
    test, test_tokens, test_seconds = collect_split("test", test_size)
    total_tokens = train_tokens + test_tokens
    total_seconds = train_seconds + test_seconds
    return {
        "train": train,
        "test": test,
        "manifest": {
            "model": MODEL,
            "train_records": train_size,
            "test_records": test_size,
            "samples_per_record": max_samples,
            "completion_tokens": total_tokens,
            "generation_seconds": total_seconds,
            "throughput_tokens_per_second": total_tokens / max(total_seconds, 1e-9),
            "seed": seed,
        },
    }


@app.local_entrypoint()
def main(
    train_size: int = 256,
    test_size: int = 128,
    max_samples: int = 8,
    seed: int = 17,
    output_dir: str = "artifacts/gsm8k",
) -> None:
    from branchpilot.schema import Rollout, write_jsonl

    payload = collect_rollouts.remote(train_size, test_size, max_samples, seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "train.jsonl", (Rollout.from_dict(row) for row in payload["train"]))
    write_jsonl(destination / "test.jsonl", (Rollout.from_dict(row) for row in payload["test"]))
    (destination / "manifest.json").write_text(
        json.dumps(payload["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote reproducible GPU rollouts and manifest to {destination}")
