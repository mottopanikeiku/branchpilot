"""Text-free ingestion of third-party LLM traffic logs.

The audit runs on production logs without a data-handling review because prompt and
completion text never survives a read. Text is folded into a truncated SHA-256 digest
(:mod:`branchpilot.ingest.formats` documents the exact framing) and dropped; no record,
report, log line, or error message ever carries it.

Reading a log
-------------
:func:`read_requests` returns a :class:`RequestStream`, which *is* an
``Iterator[RequestRecord]`` and additionally carries the ingest report::

    stream = read_requests("traffic.jsonl")
    for record in stream:
        ...
    report = stream.report()
    print(report.parsed, report.skipped, dict(report.reasons))

This is the one report API: the counts live on the object you are already holding, so a
caller cannot iterate a log and then find the parsed/skipped counts unreachable.
:meth:`RequestStream.report` is valid at any point -- it returns the counts accumulated so
far, and the final counts once :attr:`RequestStream.exhausted` is true.
:func:`read_requests_report` is the shorthand for "consume everything, give me only the
counts".

Format handling is fail-loud. ``format="auto"`` calls :func:`detect`, which raises
:class:`AmbiguousFormatError` naming every candidate rather than guessing. A structurally
invalid record raises :class:`MalformedRecordError` naming its zero-based index and the
offending field. A record that is valid but carries no auditable chat request (an
embedding call, a row exported without message content) is counted as skipped under a
closed vocabulary of reasons, and those counts are always available on the report.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from branchpilot.ingest.detect import (
    DETECTABLE_FORMAT_IDS,
    AmbiguousFormatError,
    UnknownFormatError,
    detect,
    require_file,
)
from branchpilot.ingest.formats import (
    EMPTY_TEXT_HASH,
    FORMAT_IDS,
    MAX_RECORD_BYTES,
    RECORD_READERS,
    SKIP_REASONS,
    STATUSES,
    EmptySourceError,
    IngestError,
    IngestReport,
    MalformedRecordError,
    MappingError,
    RecordReader,
    RequestRecord,
    Skipped,
    SourceNotFoundError,
    iter_json_records,
    make_generic_reader,
    read_anthropic_jsonl,
    read_helicone_export,
    read_litellm_jsonl,
    read_openai_jsonl,
    read_openrouter_export,
)

__all__ = [
    "DETECTABLE_FORMAT_IDS",
    "EMPTY_TEXT_HASH",
    "FORMAT_IDS",
    "MAX_RECORD_BYTES",
    "SKIP_REASONS",
    "STATUSES",
    "AmbiguousFormatError",
    "EmptySourceError",
    "IngestError",
    "IngestReport",
    "MalformedRecordError",
    "MappingError",
    "RequestRecord",
    "RequestStream",
    "Skipped",
    "SourceNotFoundError",
    "UnknownFormatError",
    "detect",
    "read_anthropic_jsonl",
    "read_helicone_export",
    "read_litellm_jsonl",
    "read_openai_jsonl",
    "read_openrouter_export",
    "read_requests",
    "read_requests_report",
]

_GENERIC = "generic-jsonl"


class RequestStream(Iterator[RequestRecord]):
    """A single-pass, streaming iterator over a log that carries its own ingest report.

    Peak memory is independent of file size: one line is decoded at a time, message text
    is folded straight into a hasher, and records are never accumulated internally.
    """

    __slots__ = ("_exhausted", "_format", "_iter", "_parsed", "_path", "_reader", "_reasons")

    def __init__(
        self,
        path: str | Path,
        *,
        format: str = "auto",
        mapping: Mapping[str, Any] | None = None,
    ) -> None:
        self._path = require_file(path)
        self._format = _resolve_format(self._path, format)
        if self._format == _GENERIC:
            self._reader: RecordReader = make_generic_reader(mapping)
        else:
            if mapping is not None:
                raise MappingError(
                    f"a field mapping is only used by format {_GENERIC!r}, but format "
                    f"{self._format!r} was selected; fix: drop the mapping argument, or pass "
                    f"format={_GENERIC!r} to read the log through your mapping"
                )
            self._reader = RECORD_READERS[self._format]
        self._parsed = 0
        self._reasons: dict[str, int] = {}
        self._exhausted = False
        self._iter = self._run()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def format(self) -> str:
        """The resolved format id, after auto-detection."""
        return self._format

    @property
    def exhausted(self) -> bool:
        """True once the source has been read to the end."""
        return self._exhausted

    def __iter__(self) -> RequestStream:
        return self

    def __next__(self) -> RequestRecord:
        return next(self._iter)

    def __repr__(self) -> str:
        return (
            f"RequestStream(path={str(self._path)!r}, format={self._format!r}, "
            f"parsed={self._parsed}, exhausted={self._exhausted})"
        )

    def report(self) -> IngestReport:
        """Counts accumulated so far; the final report once :attr:`exhausted` is true."""
        return IngestReport(
            parsed=self._parsed,
            skipped=sum(self._reasons.values()),
            reasons=MappingProxyType(dict(self._reasons)),
        )

    def _run(self) -> Iterator[RequestRecord]:
        reader = self._reader
        reasons = self._reasons
        with self._path.open("rb") as handle:
            for index, record in iter_json_records(handle):
                if record is None:
                    reasons["blank_line"] = reasons.get("blank_line", 0) + 1
                    continue
                outcome = reader(record, index)
                if isinstance(outcome, Skipped):
                    reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
                    continue
                self._parsed += 1
                yield outcome
        self._exhausted = True


def _resolve_format(source: Path, requested: str) -> str:
    if not isinstance(requested, str) or not requested:
        raise UnknownFormatError(
            f"format must be a format id or 'auto'; fix: pass one of: auto, {', '.join(FORMAT_IDS)}"
        )
    if requested == "auto":
        return detect(source)
    if requested not in FORMAT_IDS:
        raise UnknownFormatError(
            f"unsupported log format {requested!r}; fix: pass one of: auto, {', '.join(FORMAT_IDS)}"
        )
    return requested


def read_requests(
    path: str | Path,
    *,
    format: str = "auto",
    mapping: Mapping[str, Any] | None = None,
) -> RequestStream:
    """Stream :class:`RequestRecord` values from a log, with text hashed and discarded.

    The returned :class:`RequestStream` is an ``Iterator[RequestRecord]`` that also
    exposes :meth:`RequestStream.report`. Source and format problems are raised here, up
    front; per-record problems are raised while iterating.
    """
    return RequestStream(path, format=format, mapping=mapping)


def read_requests_report(
    path: str | Path,
    *,
    format: str = "auto",
    mapping: Mapping[str, Any] | None = None,
) -> IngestReport:
    """Consume a log and return only its :class:`IngestReport`."""
    stream = read_requests(path, format=format, mapping=mapping)
    for _ in stream:
        pass
    return stream.report()
