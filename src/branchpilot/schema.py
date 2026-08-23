from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Sample:
    text: str
    answer: str | None
    token_count: int
    mean_logprob: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, (str, type(None))):
            raise TypeError("sample answer must be a string or None")
        if self.token_count < 0:
            raise ValueError("sample token_count cannot be negative")
        if self.mean_logprob is not None and not math.isfinite(self.mean_logprob):
            raise ValueError("sample mean_logprob must be finite")


    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Sample:
        return cls(
            text=str(value["text"]),
            answer=(None if value.get("answer") is None else str(value["answer"])),
            token_count=int(value["token_count"]),
            mean_logprob=(
                None if value.get("mean_logprob") is None else float(value["mean_logprob"])
            ),
        )


@dataclass(frozen=True, slots=True)
class Rollout:
    uid: str
    question: str
    gold: str
    samples: tuple[Sample, ...]
    prompt_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("a rollout needs at least one sample")
        if not self.gold:
            raise ValueError("gold answer cannot be empty")

        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Rollout:
        version = int(value.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {version}")
        return cls(
            uid=str(value["uid"]),
            question=str(value["question"]),
            gold=str(value["gold"]),
            samples=tuple(Sample.from_dict(item) for item in value["samples"]),
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            metadata=dict(value.get("metadata", {})),
        )


def read_jsonl(path: str | Path) -> list[Rollout]:
    records: list[Rollout] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(Rollout.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid rollout at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"no rollouts found in {path}")
    return records


def write_jsonl(path: str | Path, records: Iterable[Rollout]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[Rollout]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Rollout.from_dict(json.loads(line))
