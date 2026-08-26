from __future__ import annotations

import hashlib
import json
import statistics
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from branchpilot.schema import Rollout


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).split())


def _prompt_payload(rollout: Rollout) -> dict[str, str]:
    return {"question": _normalized_question(rollout.question)}


def _rollout_payload(rollout: Rollout) -> dict[str, Any]:
    return {
        **_prompt_payload(rollout),
        "gold": rollout.gold,
        "prompt_tokens": rollout.prompt_tokens,
        "samples": [
            {
                "answer": sample.answer,
                "mean_logprob": sample.mean_logprob,
                "text": sample.text,
                "token_count": sample.token_count,
            }
            for sample in rollout.samples
        ],
    }


def prompt_fingerprint(rollout: Rollout) -> str:
    """Return the stable identity of the model-visible normalized question."""

    return _fingerprint(_prompt_payload(rollout))


def rollout_fingerprint(rollout: Rollout) -> str:
    """Return the stable identity of all evaluation-relevant rollout content."""

    return _fingerprint(_rollout_payload(rollout))


def _nonempty_records(records: Iterable[Rollout], *, name: str) -> tuple[Rollout, ...]:
    materialized = tuple(records)
    if not materialized:
        raise ValueError(f"{name} cannot be empty")
    return materialized


def dataset_fingerprint(records: Iterable[Rollout]) -> str:
    """Return an order-sensitive fingerprint for a non-empty rollout collection."""

    materialized = _nonempty_records(records, name="dataset")
    return _fingerprint([rollout_fingerprint(record) for record in materialized])


def _duplicate_messages(records: Sequence[Rollout]) -> list[str]:
    messages: list[str] = []
    seen_uids: dict[str, int] = {}
    seen_prompts: dict[str, int] = {}
    seen_rollouts: dict[str, int] = {}

    for index, record in enumerate(records):
        identities = (
            ("UID", record.uid, seen_uids),
            ("prompt fingerprint", prompt_fingerprint(record), seen_prompts),
            ("rollout fingerprint", rollout_fingerprint(record), seen_rollouts),
        )
        for label, identity, seen in identities:
            previous = seen.get(identity)
            if previous is None:
                seen[identity] = index
                continue
            previous_uid = records[previous].uid
            messages.append(
                f"duplicate {label} {identity!r} at records {previous} "
                f"(UID {previous_uid!r}) and {index} (UID {record.uid!r})"
            )
    return messages


def validate_unique(records: Iterable[Rollout]) -> None:
    """Reject duplicate UIDs, prompts, or rollout content within a dataset."""

    materialized = _nonempty_records(records, name="dataset")
    messages = _duplicate_messages(materialized)
    if messages:
        raise ValueError("dataset integrity validation failed:\n" + "\n".join(messages))


def validate_disjoint(left: Iterable[Rollout], right: Iterable[Rollout]) -> None:
    """Reject UID, prompt, or rollout-content overlap between two datasets."""

    left_records = _nonempty_records(left, name="left dataset")
    right_records = _nonempty_records(right, name="right dataset")
    messages: list[str] = []

    identity_functions = (
        ("UID", lambda record: record.uid),
        ("prompt fingerprint", prompt_fingerprint),
        ("rollout fingerprint", rollout_fingerprint),
    )
    for label, identity_of in identity_functions:
        left_positions: dict[str, tuple[int, Rollout]] = {}
        for index, record in enumerate(left_records):
            left_positions.setdefault(identity_of(record), (index, record))
        reported: set[str] = set()
        for right_index, right_record in enumerate(right_records):
            identity = identity_of(right_record)
            match = left_positions.get(identity)
            if match is None or identity in reported:
                continue
            reported.add(identity)
            left_index, left_record = match
            messages.append(
                f"overlapping {label} {identity!r}: left record {left_index} "
                f"(UID {left_record.uid!r}) and right record {right_index} "
                f"(UID {right_record.uid!r})"
            )

    if messages:
        raise ValueError("datasets are not disjoint:\n" + "\n".join(messages))


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    record_count: int
    sample_count: int
    horizon_min: int
    horizon_median: float
    horizon_max: int
    prompt_uniqueness: float
    parse_rate: float
    logprob_coverage: float
    completion_token_total: int
    dataset_fingerprint: str
    unique_uid_count: int
    unique_prompt_count: int
    unique_rollout_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_rollouts(records: Iterable[Rollout]) -> DatasetProfile:
    """Summarize dataset shape, coverage, and provenance identities."""

    materialized = _nonempty_records(records, name="dataset")
    horizons = [len(record.samples) for record in materialized]
    samples = [sample for record in materialized for sample in record.samples]
    record_count = len(materialized)
    sample_count = len(samples)
    unique_prompt_count = len({prompt_fingerprint(record) for record in materialized})

    return DatasetProfile(
        record_count=record_count,
        sample_count=sample_count,
        horizon_min=min(horizons),
        horizon_median=float(statistics.median(horizons)),
        horizon_max=max(horizons),
        prompt_uniqueness=unique_prompt_count / record_count,
        parse_rate=sum(sample.answer is not None for sample in samples) / sample_count,
        logprob_coverage=sum(sample.mean_logprob is not None for sample in samples) / sample_count,
        completion_token_total=sum(sample.token_count for sample in samples),
        dataset_fingerprint=dataset_fingerprint(materialized),
        unique_uid_count=len({record.uid for record in materialized}),
        unique_prompt_count=unique_prompt_count,
        unique_rollout_count=len({rollout_fingerprint(record) for record in materialized}),
    )
