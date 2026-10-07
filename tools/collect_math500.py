"""Draw pinned MATH-500 response banks on one Modal L4 with vLLM."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/math500-protocol.json"
IMAGE = (
    "vllm/vllm-openai:v0.10.2@"
    "sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db"
)
app = modal.App("branchpilot-math500")
image = (
    modal.Image.from_registry(IMAGE)
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .entrypoint([])
)
TIMEOUT = int(os.environ.get("BRANCHPILOT_CLOUD_TIMEOUT", "300"))


def encode(records: list[dict]) -> bytes:
    payload = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in records
    )
    return gzip.compress(payload.encode(), compresslevel=9, mtime=0)


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=TIMEOUT, max_containers=1)
def collect(protocol: dict, protocol_sha256: str, generation_commit: str, pilot: bool) -> dict:
    import importlib.metadata

    import torch
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    payload = urllib.request.urlopen(protocol["dataset"]["url"], timeout=60).read()
    if hashlib.sha256(payload).hexdigest() != protocol["dataset"]["sha256"]:
        raise ValueError("MATH-500 source checksum differs from the committed protocol")
    dataset = [json.loads(line) for line in payload.splitlines() if line]
    if len(dataset) != protocol["dataset"]["rows"]:
        raise ValueError("MATH-500 source row count differs from the committed protocol")
    config = protocol["collection"]
    model_info = protocol["models"][0]
    llm = LLM(
        model=model_info["id"],
        revision=model_info["revision"],
        tokenizer_revision=model_info["revision"],
        dtype="half",
        download_dir="/tmp/branchpilot-model",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=config["max_model_len"],
        max_num_seqs=128,
        seed=config["seed"],
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        n=config["samples_per_prompt"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        logprobs=config["logprobs"],
        max_tokens=config["max_tokens"],
        seed=config["seed"],
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    startup_seconds = time.perf_counter() - started
    outputs = {}
    metrics = {}
    split_indices = (
        {"pilot": config["pilot_indices"]}
        if pilot
        else {name: split["indices"] for name, split in protocol["splits"].items()}
    )
    for name, indices in split_indices.items():
        rows = []
        split_started = time.perf_counter()
        for offset in range(0, len(indices), 50):
            selected = indices[offset : offset + 50]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config["system_prompt"]},
                        {"role": "user", "content": dataset[index]["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for index in selected
            ]
            requests = llm.generate(prompts, sampling, use_tqdm=False)
            for index, request in zip(selected, requests, strict=True):
                source = dataset[index]
                if len(request.outputs) != config["samples_per_prompt"]:
                    raise ValueError(f"Incomplete bank for {source['unique_id']}")
                samples = []
                for output in sorted(request.outputs, key=lambda item: item.index):
                    tokens = len(output.token_ids)
                    finish = str(output.finish_reason or "unknown")
                    logprob = output.cumulative_logprob
                    samples.append(
                        {
                            "text": output.text,
                            "answer": None,
                            "token_count": tokens,
                            "mean_logprob": (
                                None if logprob is None or not tokens else float(logprob) / tokens
                            ),
                            "finish_reason": finish,
                            "parse_status": (
                                "truncated"
                                if finish == "length"
                                else "unparsed"
                                if finish == "stop"
                                else "incomplete"
                            ),
                        }
                    )
                rows.append(
                    {
                        "schema_version": 2,
                        "uid": source["unique_id"],
                        "question": source["problem"],
                        "gold": "math500-gold-unused",
                        "samples": samples,
                        "prompt_tokens": len(request.prompt_token_ids),
                        "metadata": {
                            "dataset": protocol["dataset"]["id"],
                            "dataset_revision": protocol["dataset"]["revision"],
                            "dataset_uid": source["unique_id"],
                            "source_index": index,
                            "output_split": name,
                            "gold_latex": source["answer"],
                            "subject": source["subject"],
                            "level": source["level"],
                            "model": model_info["id"],
                            "model_revision": model_info["revision"],
                        },
                    }
                )
        elapsed = time.perf_counter() - split_started
        bank = encode(rows)
        outputs[name] = bank
        completion_tokens = sum(sample["token_count"] for row in rows for sample in row["samples"])
        metrics[name] = {
            "records": len(rows),
            "samples": sum(len(row["samples"]) for row in rows),
            "completion_tokens": completion_tokens,
            "generation_seconds": elapsed,
            "completion_tokens_per_second": completion_tokens / elapsed,
            "compressed_bytes": len(bank),
            "raw_bank_sha256": hashlib.sha256(bank).hexdigest(),
            "finish_reasons": {
                reason: sum(
                    sample["finish_reason"] == reason for row in rows for sample in row["samples"]
                )
                for reason in ("stop", "length", "unknown")
            },
        }
    cloud_seconds = time.perf_counter() - started
    return {
        "banks": outputs,
        "manifest": {
            "protocol_sha256": protocol_sha256,
            "generation_commit": generation_commit,
            "pilot": pilot,
            "dataset": protocol["dataset"],
            "model": model_info,
            "sampling": config,
            "runtime": {
                "image": IMAGE,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_type": "L4",
                "vllm": importlib.metadata.version("vllm"),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cpu_cores": 2,
                "memory_gib": 8,
            },
            "startup_seconds": startup_seconds,
            "cloud_seconds": cloud_seconds,
            "cloud_minutes": cloud_seconds / 60,
            "cloud_cost_estimate_usd": cloud_seconds
            / 3600
            * (0.7992 + 2 * 0.047160 + 8 * 0.007992),
            "cost_estimate_basis": (
                "L4, two CPU cores and 8 GiB at published hourly rates; "
                "excludes image build and client overhead"
            ),
            "splits": metrics,
        },
    }


@app.local_entrypoint()
def main(pilot: bool = False, output_dir: str = "benchmarks/math500") -> None:
    raw_protocol = PROTOCOL.read_bytes()
    protocol = json.loads(raw_protocol)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    destination = ROOT / output_dir
    names = ["pilot"] if pilot else list(protocol["splits"])
    if any((destination / f"{name}.jsonl.gz").exists() for name in names):
        raise FileExistsError("Refusing to redraw an existing response bank")
    result = collect.remote(protocol, hashlib.sha256(raw_protocol).hexdigest(), commit, pilot)
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in result["banks"].items():
        (destination / f"{name}.jsonl.gz").write_bytes(payload)
    manifest_name = "pilot-manifest.json" if pilot else "generation-manifest.json"
    (destination / manifest_name).write_text(json.dumps(result["manifest"], indent=2) + "\n")
    print(json.dumps(result["manifest"], indent=2))
