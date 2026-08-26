from __future__ import annotations

import io
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from branchpilot.artifacts import atomic_text_writer

SCHEMA_VERSION = 2
_PARSE_STATUSES = frozenset(
    {"incomplete", "parsed", "parsed_explicit", "parsed_fallback", "unparsed", "truncated"}
)


@dataclass(frozen=True, slots=True)
class Sample:
    text: str
    answer: str | None
    token_count: int
    mean_logprob: float | None = None
    finish_reason: str | None = None
    parse_status: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("sample text must be a string")
        if not isinstance(self.answer, (str, type(None))):
            raise TypeError("sample answer must be a string or None")
        if not isinstance(self.token_count, int) or isinstance(self.token_count, bool):
            raise TypeError("sample token_count must be an integer")
        if self.token_count < 0:
            raise ValueError("sample token_count cannot be negative")
        if self.mean_logprob is not None:
            if isinstance(self.mean_logprob, bool) or not isinstance(
                self.mean_logprob, (int, float)
            ):
                raise TypeError("sample mean_logprob must be a real number or None")
            if not math.isfinite(self.mean_logprob):
                raise ValueError("sample mean_logprob must be finite")
        if not isinstance(self.finish_reason, (str, type(None))):
            raise TypeError("sample finish_reason must be a string or None")
        if self.parse_status is not None and self.parse_status not in _PARSE_STATUSES:
            expected = ", ".join(sorted(_PARSE_STATUSES))
            raise ValueError(f"sample parse_status must be one of: {expected}")
        if (
            self.parse_status is not None
            and self.parse_status.startswith("parsed")
            and not self.answer
        ):
            raise ValueError("a parsed sample must have a non-empty answer")
        if self.parse_status in {"incomplete", "truncated", "unparsed"} and self.answer is not None:
            raise ValueError(f"a {self.parse_status} sample cannot have an answer")
        if self.finish_reason == "length" and self.parse_status not in {None, "truncated"}:
            raise ValueError("a length-finished sample must be marked truncated")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Sample:
        return cls(
            text=str(value["text"]),
            answer=(None if value.get("answer") is None else str(value["answer"])),
            token_count=int(value["token_count"]),
            mean_logprob=(
                None if value.get("mean_logprob") is None else float(value["mean_logprob"])
            ),
            finish_reason=(
                None if value.get("finish_reason") is None else str(value["finish_reason"])
            ),
            parse_status=(
                None if value.get("parse_status") is None else str(value["parse_status"])
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
        if not isinstance(self.uid, str) or not self.uid.strip():
            raise ValueError("rollout uid cannot be empty")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("rollout question cannot be empty")
        if not isinstance(self.gold, str) or not self.gold:
            raise ValueError("gold answer cannot be empty")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("a rollout needs a non-empty sample tuple")
        if any(not isinstance(sample, Sample) for sample in self.samples):
            raise TypeError("rollout samples must contain only Sample values")
        if not isinstance(self.prompt_tokens, int) or isinstance(self.prompt_tokens, bool):
            raise TypeError("prompt_tokens must be an integer")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens cannot be negative")
        if not isinstance(self.metadata, dict):
            raise TypeError("rollout metadata must be a dictionary")
        try:
            json.dumps(self.metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rollout metadata must be finite JSON data: {exc}") from exc

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


def _parse_jsonl(lines: Iterable[str], source: str) -> Iterator[Rollout]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("record must be a JSON object")
            yield Rollout.from_dict(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid rollout at {source}:{line_number}: {exc}") from exc


def read_jsonl_bytes(payload: bytes, source: str = "<bytes>") -> list[Rollout]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 rollout data at {source}") from exc
    records = list(_parse_jsonl(io.StringIO(text), source))
    if not records:
        raise ValueError(f"no rollouts found in {source}")
    return records


def read_jsonl(path: str | Path) -> list[Rollout]:
    records = list(iter_jsonl(path))
    if not records:
        raise ValueError(f"no rollouts found in {path}")
    return records


def write_jsonl(path: str | Path, records: Iterable[Rollout]) -> None:
    with atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[Rollout]:
    with Path(path).open(encoding="utf-8") as handle:
        yield from _parse_jsonl(handle, str(path))
