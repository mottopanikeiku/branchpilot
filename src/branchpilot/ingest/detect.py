"""Format detection that refuses to guess.

:func:`detect` reads the head of a log and counts, per format, how many of those records
its probe recognizes. The format with the strictly highest count wins. A tie between two
or more formats raises :class:`AmbiguousFormatError` naming every tied candidate, and no
recognized record at all raises :class:`UnknownFormatError`. There is no default and no
tie-break, because a mis-detected format silently produces a plausible-looking but wrong
audit.

Counting rather than requiring unanimity is deliberate: real exports interleave rows a
format's probe cannot recognize -- an error row with no usage block, an embedding call --
and demanding that every record match would refuse ordinary production logs. A row the
winning reader cannot parse still raises :class:`MalformedRecordError` with its index, so
a genuinely mixed file fails loudly at read time rather than being silently half-read.

``generic-jsonl`` is deliberately not probed: it exists only for explicitly mapped logs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from branchpilot.ingest.formats import (
    EmptySourceError,
    IngestError,
    SourceNotFoundError,
    iter_json_records,
)

__all__ = [
    "DETECT_RECORDS",
    "DETECTABLE_FORMAT_IDS",
    "AmbiguousFormatError",
    "UnknownFormatError",
    "detect",
]

DETECT_RECORDS = 16
"""Number of leading non-blank records inspected before deciding."""


class AmbiguousFormatError(IngestError):
    """Two or more formats matched every inspected record."""

    def __init__(self, candidates: Sequence[str], source: str) -> None:
        self.candidates = tuple(candidates)
        listed = ", ".join(self.candidates)
        super().__init__(
            f"{source} matches more than one log format: {listed}; "
            f"fix: pass format='<id>' to read_requests (one of: {listed}) instead of relying "
            "on auto-detection"
        )


class UnknownFormatError(IngestError):
    """No known format matched, or an unsupported format id was requested."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _probe_openai(record: Mapping[str, Any]) -> bool:
    request = record.get("request")
    response = record.get("response")
    if not _is_object(request) or not _is_object(response):
        return False
    if "choices" not in response and "usage" not in response:
        return False
    if _is_list(response.get("content")) or "input_tokens" in (response.get("usage") or {}):
        return False
    return "messages" in request or "input" in request or "prompt" in request


def _probe_anthropic(record: Mapping[str, Any]) -> bool:
    request = record.get("request")
    response = record.get("response")
    if not _is_object(request) or not _is_object(response):
        return False
    if "messages" not in request:
        return False
    usage = response.get("usage")
    if _is_object(usage) and "input_tokens" in usage:
        return True
    return _is_list(response.get("content")) and "stop_reason" in response


def _probe_litellm(record: Mapping[str, Any]) -> bool:
    return (
        "request_id" in record
        and "call_type" in record
        and "startTime" in record
        and "request_body" not in record
    )


def _probe_helicone(record: Mapping[str, Any]) -> bool:
    return (
        "request_id" in record
        and "request_created_at" in record
        and _is_object(record.get("request_body"))
        and "response_body" in record
    )


def _probe_openrouter(record: Mapping[str, Any]) -> bool:
    if "id" not in record or "created_at" not in record or "provider_name" not in record:
        return False
    return any(
        key in record
        for key in (
            "native_tokens_completion",
            "native_tokens_prompt",
            "tokens_completion",
            "tokens_prompt",
        )
    )


_PROBES: Mapping[str, Any] = {
    "anthropic-jsonl": _probe_anthropic,
    "helicone-export": _probe_helicone,
    "litellm-jsonl": _probe_litellm,
    "openai-jsonl": _probe_openai,
    "openrouter-export": _probe_openrouter,
}
DETECTABLE_FORMAT_IDS = tuple(sorted(_PROBES))


def head_records(path: str | Path, limit: int = DETECT_RECORDS) -> list[Mapping[str, Any]]:
    """Return up to ``limit`` leading non-blank JSON objects from ``path``."""
    source = require_file(path)
    records: list[Mapping[str, Any]] = []
    with source.open("rb") as handle:
        for _, record in iter_json_records(handle):
            if record is None:
                continue
            records.append(record)
            if len(records) >= limit:
                break
    return records


def require_file(path: str | Path) -> Path:
    """Return ``path`` as a :class:`Path`, refusing anything that is not a readable file."""
    source = Path(path)
    if not source.is_file():
        what = "is a directory" if source.is_dir() else "does not exist"
        raise SourceNotFoundError(
            f"log source {str(source)!r} {what}; "
            "fix: pass the path of a newline-delimited JSON log file"
        )
    return source


def detect(path: str | Path) -> str:
    """Return the one format that recognizes the most head records, or raise."""
    source = require_file(path)
    records = head_records(source)
    if not records:
        raise EmptySourceError(
            f"log source {str(source)!r} contains no JSON records; "
            "fix: point at a newline-delimited JSON log with at least one record"
        )
    matches = {
        format_id: sum(1 for record in records if probe(record))
        for format_id, probe in _PROBES.items()
    }
    best = max(matches.values())
    if best == 0:
        known = ", ".join(DETECTABLE_FORMAT_IDS)
        raise UnknownFormatError(
            f"log source {str(source)!r} does not match any known format across its first "
            f"{len(records)} record(s); fix: pass format='<id>' explicitly (auto-detectable: "
            f"{known}), or use format='generic-jsonl' with a mapping describing your fields"
        )
    candidates = sorted(format_id for format_id, count in matches.items() if count == best)
    if len(candidates) > 1:
        raise AmbiguousFormatError(candidates, f"log source {str(source)!r}")
    return candidates[0]
