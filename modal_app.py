from __future__ import annotations

import gzip
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import modal

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
CACHE_PATH = "/root/.cache/huggingface"
CUDA_IMAGE = (
    "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@"
    "sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb"
)
SYSTEM_PROMPT = (
    "Solve the arithmetic word problem carefully. Show concise reasoning, then put only the "
    "final numeric value after '####'."
)
PARSER = "branchpilot.numeric-complete-v3"

app = modal.App("branchpilot-rollouts")
cache = modal.Volume.from_name("branchpilot-huggingface", create_if_missing=True)
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("gcc", "g++")
    .pip_install(
        "datasets==4.0.0",
        "transformers==4.55.2",
        "tokenizers==0.21.1",
        "vllm==0.10.2",
    )
    .add_local_python_source("branchpilot")
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_jsonl(records: list[dict[str, Any]]) -> bytes:
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    ).encode()
    return gzip.compress(payload, compresslevel=6, mtime=0)


@app.function(
    image=image,
    gpu="L4",
    timeout=3 * 60 * 60,
    scaledown_window=5 * 60,
    volumes={CACHE_PATH: cache},
)
def collect_rollouts(
    train_size: int = 768,
    validation_size: int = 256,
    test_size: int = 512,
    max_samples: int = 8,
    max_tokens: int = 512,
    seed: int = 17,
    model_id: str = MODEL,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    import importlib.metadata
    import time

    import torch
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    from branchpilot.answers import extract_answer

    if min(train_size, validation_size, test_size, max_samples, max_tokens) < 1:
        raise ValueError("split sizes, max_samples, and max_tokens must be positive")

    train_dataset = load_dataset(
        DATASET,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
    )
    test_dataset = load_dataset(
        DATASET,
        DATASET_CONFIG,
        split="test",
        revision=DATASET_REVISION,
    )
    if train_size + validation_size > len(train_dataset):
        raise ValueError("requested train and validation records exceed GSM8K train")
    if test_size > len(test_dataset):
        raise ValueError("requested test records exceed the official GSM8K test split")

    train_indices = list(range(len(train_dataset)))
    random.Random(seed).shuffle(train_indices)
    validation_indices = train_indices[train_size : train_size + validation_size]
    train_indices = train_indices[:train_size]
    test_indices = list(range(len(test_dataset)))
    random.Random(seed + 1).shuffle(test_indices)
    test_indices = test_indices[:test_size]

    model = LLM(
        model=model_id,
        revision=model_revision,
        tokenizer_revision=model_revision,
        download_dir=CACHE_PATH,
        dtype="half",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        seed=seed,
    )
    tokenizer = model.get_tokenizer()
    sampling = SamplingParams(
        n=max_samples,
        temperature=0.7,
        top_p=0.95,
        logprobs=1,
        max_tokens=max_tokens,
        seed=seed,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    def collect_split(
        dataset: Any,
        dataset_split: str,
        output_split: str,
        indices: list[int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        selected = dataset.select(indices)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
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
        parse_counts = {
            "parsed_explicit": 0,
            "parsed_fallback": 0,
            "unparsed": 0,
            "truncated": 0,
        }
        records: list[dict[str, Any]] = []
        for source_index, row, request in zip(indices, selected, generated, strict=True):
            gold = extract_answer(row["answer"])
            if gold is None:
                raise ValueError(
                    f"could not parse GSM8K gold answer at {dataset_split}:{source_index}"
                )
            if len(request.outputs) != max_samples:
                raise ValueError(
                    f"expected {max_samples} outputs at {dataset_split}:{source_index}, "
                    f"received {len(request.outputs)}"
                )
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
                finish_reason = str(output.finish_reason or "unknown")
                explicit_answer = extract_answer(output.text)
                if finish_reason == "length":
                    answer = None
                    parse_status = "truncated"
                elif explicit_answer is not None:
                    answer = explicit_answer
                    parse_status = "parsed_explicit"
                else:
                    answer = extract_answer(output.text, strict=False)
                    parse_status = "parsed_fallback" if answer is not None else "unparsed"
                parse_counts[parse_status] += 1
                samples.append(
                    {
                        "text": output.text,
                        "answer": answer,
                        "token_count": token_count,
                        "mean_logprob": mean_logprob,
                        "finish_reason": finish_reason,
                        "parse_status": parse_status,
                    }
                )
            records.append(
                {
                    "schema_version": 2,
                    "uid": f"gsm8k-{dataset_split}-{source_index}",
                    "question": row["question"],
                    "gold": gold,
                    "samples": samples,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "metadata": {
                        "dataset": DATASET,
                        "dataset_config": DATASET_CONFIG,
                        "dataset_revision": DATASET_REVISION,
                        "source_split": dataset_split,
                        "output_split": output_split,
                        "source_index": source_index,
                        "model": model_id,
                        "model_revision": model_revision,
                        "temperature": sampling.temperature,
                        "top_p": sampling.top_p,
                        "seed": seed,
                        "parser": PARSER,
                    },
                }
            )
        return records, {
            "records": len(records),
            "completion_tokens": completion_tokens,
            "generation_seconds": elapsed,
            "throughput_tokens_per_second": completion_tokens / max(elapsed, 1e-9),
            "parse_counts": parse_counts,
            "source_indices": indices,
            "source_indices_sha256": _canonical_hash(indices),
        }

    train, train_metrics = collect_split(train_dataset, "train", "train", train_indices)
    validation, validation_metrics = collect_split(
        train_dataset,
        "train",
        "validation",
        validation_indices,
    )
    test, test_metrics = collect_split(test_dataset, "test", "test", test_indices)
    manifest = {
        "manifest_schema": 2,
        "data_schema": 2,
        "dataset": {
            "id": DATASET,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "train_fingerprint": train_dataset._fingerprint,
            "test_fingerprint": test_dataset._fingerprint,
        },
        "model": {
            "id": model_id,
            "revision": model_revision,
            "dtype": "float16",
        },
        "sampling": {
            "samples_per_prompt": max_samples,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "logprobs": 1,
            "seed": seed,
            "stop": sampling.stop,
        },
        "prompt": {
            "system": SYSTEM_PROMPT,
            "sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "parser": PARSER,
        },
        "runtime": {
            "cuda_image": CUDA_IMAGE,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "datasets": importlib.metadata.version("datasets"),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "vllm": importlib.metadata.version("vllm"),
        },
        "splits": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }
    return {
        "train_jsonl_gzip": _encode_jsonl(train),
        "validation_jsonl_gzip": _encode_jsonl(validation),
        "test_jsonl_gzip": _encode_jsonl(test),
        "manifest": manifest,
    }


def _decode_jsonl(payload: bytes):
    from branchpilot.schema import Rollout

    text = gzip.decompress(payload).decode()
    return [Rollout.from_dict(json.loads(line)) for line in text.splitlines() if line]


def _git_provenance(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@app.local_entrypoint()
def main(
    train_size: int = 768,
    validation_size: int = 256,
    test_size: int = 512,
    max_samples: int = 8,
    max_tokens: int = 512,
    seed: int = 17,
    model_id: str = MODEL,
    model_revision: str = MODEL_REVISION,
    output_dir: str = "artifacts/gsm8k",
) -> None:
    from branchpilot import __version__
    from branchpilot.artifacts import atomic_write_text, sha256_file
    from branchpilot.integrity import profile_rollouts, validate_disjoint, validate_unique
    from branchpilot.schema import write_jsonl

    payload = collect_rollouts.remote(
        train_size,
        validation_size,
        test_size,
        max_samples,
        max_tokens,
        seed,
        model_id,
        model_revision,
    )
    train = _decode_jsonl(payload["train_jsonl_gzip"])
    validation = _decode_jsonl(payload["validation_jsonl_gzip"])
    test = _decode_jsonl(payload["test_jsonl_gzip"])
    validate_unique(train)
    validate_unique(validation)
    validate_unique(test)
    validate_disjoint(train, validation)
    validate_disjoint(train, test)
    validate_disjoint(validation, test)

    destination = Path(output_dir)
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "validation.jsonl", validation)
    write_jsonl(destination / "test.jsonl", test)
    manifest = payload["manifest"]
    repository = Path(__file__).resolve().parent
    source_files = [
        repository / "modal_app.py",
        repository / "uv.lock",
        *sorted((repository / "src" / "branchpilot").glob("*.py")),
    ]
    manifest["software"] = {
        "branchpilot": __version__,
        "git": _git_provenance(repository),
        "source_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in source_files
            if path.is_file()
        },
    }
    manifest["profiles"] = {
        "train": profile_rollouts(train).to_dict(),
        "validation": profile_rollouts(validation).to_dict(),
        "test": profile_rollouts(test).to_dict(),
    }
    atomic_write_text(
        destination / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(f"Wrote pinned GPU rollouts, validation split, and manifest to {destination}")
