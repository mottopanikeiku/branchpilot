from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import modal

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET = "openai/grade-school-math"
DATASET_CONFIG = "main"
DATASET_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{DATASET_REVISION}/grade_school_math/data"
)
DATASET_TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
DATASET_TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.10.2@"
    "sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db"
)
RUNTIME_OVERLAY = "ln -sf /usr/bin/python3 /usr/bin/python"
SYSTEM_PROMPT = (
    "Solve the arithmetic word problem carefully. Show concise reasoning, then put only the "
    "final numeric value after '####'."
)
PARSER = "branchpilot.numeric-complete-v3"
COMPLETED_OUTPUT_FALLBACK = "last-numeric-only-when-finish-reason-is-stop"
DATASET_SELECTION = "seeded-shuffle-with-recorded-source-indices"
TEST_SPLIT = "complete-official-test"

app = modal.App("branchpilot-rollouts")
image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .run_commands(RUNTIME_OVERLAY)
    .entrypoint([])
    .add_local_python_source("branchpilot")
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return _sha256_bytes(encoded)


def _encode_jsonl(records: list[dict[str, Any]]) -> bytes:
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    ).encode()
    return gzip.compress(payload, compresslevel=6, mtime=0)


def _download_dataset_split(split: str, expected_sha256: str) -> tuple[list[dict[str, str]], dict]:
    url = f"{DATASET_BASE_URL}/{split}.jsonl"
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - hash-pinned HTTPS
        payload = response.read()
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"GSM8K {split} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {"question", "answer"}:
            raise ValueError(f"invalid GSM8K {split} row at line {line_number}")
        question, answer = value["question"], value["answer"]
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"non-string GSM8K {split} row at line {line_number}")
        rows.append({"question": question, "answer": answer})
    return rows, {
        "url": url,
        "bytes": len(payload),
        "sha256": actual_sha256,
        "records": len(rows),
    }


def _snapshot_files(root: Path) -> list[dict[str, str | int]]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"model snapshot contains no files: {root}")
    return files


@app.function(
    image=image,
    gpu="L4",
    timeout=3 * 60 * 60,
    scaledown_window=5 * 60,
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
    software_provenance: dict[str, Any] | None = None,
    protocol_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    import importlib.metadata
    import time

    import torch
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    from branchpilot.answers import extract_answer

    if min(train_size, validation_size, test_size, max_samples, max_tokens) < 1:
        raise ValueError("split sizes, max_samples, and max_tokens must be positive")
    if software_provenance is None or protocol_metadata is None:
        raise ValueError("canonical collection requires software and protocol provenance")

    train_dataset, train_source = _download_dataset_split("train", DATASET_TRAIN_SHA256)
    test_dataset, test_source = _download_dataset_split("test", DATASET_TEST_SHA256)
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

    cache_path = tempfile.mkdtemp(prefix="branchpilot-huggingface-")
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            cache_dir=cache_path,
        )
    )
    model_files = _snapshot_files(snapshot)
    model = LLM(
        model=str(snapshot),
        tokenizer=str(snapshot),
        download_dir=cache_path,
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
        dataset: list[dict[str, str]],
        dataset_split: str,
        output_split: str,
        indices: list[int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        selected = [dataset[index] for index in indices]
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
            "incomplete": 0,
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
                elif finish_reason != "stop":
                    answer = None
                    parse_status = "incomplete"
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
    packages = sorted(
        {
            (distribution.metadata.get("Name") or "unknown", distribution.version)
            for distribution in importlib.metadata.distributions()
        }
    )
    package_inventory = [{"name": name, "version": version} for name, version in packages]
    manifest = {
        "manifest_schema": 3,
        "data_schema": 2,
        "protocol": protocol_metadata,
        "software": software_provenance,
        "dataset": {
            "id": DATASET,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "selection": DATASET_SELECTION,
            "test_split": TEST_SPLIT,
            "source_files": {"train": train_source, "test": test_source},
        },
        "model": {
            "id": model_id,
            "revision": model_revision,
            "dtype": "float16",
            "snapshot_files": model_files,
            "snapshot_sha256": _canonical_hash(model_files),
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
            "sha256": _sha256_bytes(SYSTEM_PROMPT.encode()),
            "parser": PARSER,
            "completed_output_fallback": COMPLETED_OUTPUT_FALLBACK,
            "truncated_outputs_vote": False,
        },
        "runtime": {
            "image": VLLM_IMAGE,
            "overlay": RUNTIME_OVERLAY,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "packages": package_inventory,
            "packages_sha256": _canonical_hash(package_inventory),
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


def _software_provenance(repository: Path, version: str) -> dict[str, Any]:
    from branchpilot.artifacts import sha256_file

    source_files = [
        repository / "modal_app.py",
        repository / "uv.lock",
        *sorted((repository / "src" / "branchpilot").rglob("*.py")),
    ]
    return {
        "branchpilot": version,
        "git": _git_provenance(repository),
        "source_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in source_files
            if path.is_file()
        },
    }


def _validate_collection_protocol(
    protocol: Any,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    max_samples: int,
    max_tokens: int,
    seed: int,
    model_id: str,
    model_revision: str,
) -> None:
    requirements = {
        "scope.dataset": DATASET,
        "scope.dataset_config": DATASET_CONFIG,
        "scope.dataset_revision": DATASET_REVISION,
        "scope.dataset_train_sha256": DATASET_TRAIN_SHA256,
        "scope.dataset_test_sha256": DATASET_TEST_SHA256,
        "scope.model": model_id,
        "scope.model_revision": model_revision,
        "scope.runtime_image": VLLM_IMAGE,
        "scope.runtime_overlay": RUNTIME_OVERLAY,
        "collection.train_records": train_size,
        "collection.validation_records": validation_size,
        "collection.test_records": test_size,
        "collection.test_split": TEST_SPLIT,
        "collection.selection": DATASET_SELECTION,
        "collection.samples_per_prompt": max_samples,
        "collection.max_completion_tokens": max_tokens,
        "collection.temperature": 0.7,
        "collection.top_p": 0.95,
        "collection.logprobs": 1,
        "collection.sampling_seed": seed,
        "collection.system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode()),
        "collection.parser": PARSER,
        "collection.completed_output_fallback": COMPLETED_OUTPUT_FALLBACK,
        "collection.truncated_outputs_vote": False,
    }
    for dotted_path, actual in requirements.items():
        protocol.require(dotted_path, actual)


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
    protocol: str = "",
    output_dir: str = "artifacts/gsm8k",
) -> None:
    from branchpilot import __version__
    from branchpilot.artifacts import atomic_write_text, sha256_file
    from branchpilot.integrity import profile_rollouts, validate_disjoint, validate_unique
    from branchpilot.provenance import Protocol, verify_manifest_artifact
    from branchpilot.schema import write_jsonl

    if not protocol:
        raise ValueError("canonical collection requires --protocol")
    frozen_protocol = Protocol.load(protocol)
    _validate_collection_protocol(
        frozen_protocol,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        max_samples=max_samples,
        max_tokens=max_tokens,
        seed=seed,
        model_id=model_id,
        model_revision=model_revision,
    )
    repository = Path(__file__).resolve().parent
    software = _software_provenance(repository, __version__)
    if software["git"]["dirty"] is not False:
        raise ValueError("canonical collection requires a clean Git worktree")
    protocol_metadata = frozen_protocol.metadata()
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"canonical output directory must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = collect_rollouts.remote(
        train_size,
        validation_size,
        test_size,
        max_samples,
        max_tokens,
        seed,
        model_id,
        model_revision,
        software,
        protocol_metadata,
    )
    if _software_provenance(repository, __version__) != software:
        raise RuntimeError("source or Git revision changed during canonical generation")
    if Protocol.load(protocol).metadata() != protocol_metadata:
        raise RuntimeError("frozen protocol changed during canonical generation")

    train = _decode_jsonl(payload["train_jsonl_gzip"])
    validation = _decode_jsonl(payload["validation_jsonl_gzip"])
    test = _decode_jsonl(payload["test_jsonl_gzip"])
    validate_unique(train)
    validate_unique(validation)
    validate_unique(test)
    validate_disjoint(train, validation)
    validate_disjoint(train, test)
    validate_disjoint(validation, test)

    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".staging",
        )
    )
    published = False
    try:
        artifacts = {
            "train": staging / "train.jsonl",
            "validation": staging / "validation.jsonl",
            "test": staging / "test.jsonl",
        }
        write_jsonl(artifacts["train"], train)
        write_jsonl(artifacts["validation"], validation)
        write_jsonl(artifacts["test"], test)
        manifest = payload["manifest"]
        manifest["profiles"] = {
            "train": profile_rollouts(train).to_dict(),
            "validation": profile_rollouts(validation).to_dict(),
            "test": profile_rollouts(test).to_dict(),
        }
        manifest["artifacts"] = {
            split: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for split, path in artifacts.items()
        }
        manifest_path = staging / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        for split, path in artifacts.items():
            verify_manifest_artifact(manifest_path, path, split, frozen_protocol)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"canonical output path appeared during generation: {destination}")
        os.replace(staging, destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    print(f"Wrote hash-bound GPU rollouts, validation split, and manifest to {destination}")
