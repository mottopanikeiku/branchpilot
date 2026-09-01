"""Per-format record readers and the text-free :class:`RequestRecord` value type.

Every reader here takes one already-parsed JSON object plus its zero-based record index
and returns either a :class:`RequestRecord` or a :class:`Skipped` marker. No reader ever
retains, returns, logs, or embeds prompt or completion text: text is folded into a
truncated SHA-256 digest and dropped before the reader returns.

Canonical hash framing
----------------------
``messages_hash`` and ``system_prefix_hash`` are the first 16 bytes of a SHA-256 digest,
rendered as 32 lowercase hex characters. The digest is fed a length-prefixed, unambiguous
encoding of the request messages::

    frame(s)  := ascii(len(utf8(s))) || 0x00 || utf8(s)
    message   := frame(role) || frame(str(part_count)) || (frame(kind) || frame(text))*
    digest    := sha256(message_0 || message_1 || ...).digest()[:16].hex()

``role`` is the lowercased message role. Each content part contributes its ``kind``
(``"text"`` for plain string content, otherwise the block ``type``) and its text, which is
empty for non-text blocks so image bytes and tool payloads can never enter the digest.
String content is a single ``("text", content)`` part; ``None`` content contributes zero
parts. Because the framing is defined over roles and text only, the same conversation
logged by two different providers yields the same ``messages_hash``.

``system_prefix_hash`` covers the contiguous run of leading ``system``/``developer``
messages under the same framing. When a request has no system prefix the value is
:data:`EMPTY_TEXT_HASH` -- the digest of the empty *byte string*, not of a framed empty
message -- so "no prefix" is a distinct constant that cannot collide with a real prefix.

``system_prefix_chars`` is the exact number of characters of *content text* in that same
contiguous system prefix, summed over every text part of every prefix message. It is
measured while the text is in the hasher's hands and before the text is dropped, so it
costs no extra pass and retains nothing: a count is not text, and no tokenizer is
involved. Roles, framing bytes, and non-text blocks contribute nothing. A request with no
system prefix has ``system_prefix_chars == 0``.

Volatile-prefix detection is what this count exists for: a cluster whose prefix *length*
is stable while its ``system_prefix_hash`` churns is a prefix with an injected timestamp,
uuid, or per-user string in it, silently defeating the provider's prefix cache.

Token normalization
-------------------
``prompt_tokens`` is always the full billed prompt and ``cached_prompt_tokens`` is always
the cached subset of it (enforced: ``cached_prompt_tokens <= prompt_tokens``). Anthropic
reports ``input_tokens`` *excluding* cache traffic, so its reader sums ``input_tokens +
cache_read_input_tokens + cache_creation_input_tokens``. ``cached_prompt_tokens`` is
``None`` only when the source omits the field entirely; a source-reported zero stays zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "EMPTY_TEXT_HASH",
    "FORMAT_IDS",
    "MAX_RECORD_BYTES",
    "RECORD_READERS",
    "SKIP_REASONS",
    "STATUSES",
    "EmptySourceError",
    "IngestError",
    "IngestReport",
    "MalformedRecordError",
    "MappingError",
    "RequestRecord",
    "Skipped",
    "SourceNotFoundError",
    "iter_json_records",
    "make_generic_reader",
    "read_anthropic_jsonl",
    "read_helicone_export",
    "read_litellm_jsonl",
    "read_openai_jsonl",
    "read_openrouter_export",
]

MAX_RECORD_BYTES = 8 * 1024 * 1024
"""Hard cap on one record's byte length, so peak memory stays bounded on any input."""

_CHUNK_BYTES = 262_144
_HASH_BYTES = 16
_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")

EMPTY_TEXT_HASH = hashlib.sha256(b"").digest()[:_HASH_BYTES].hex()

STATUSES = frozenset({"ok", "error", "timeout", "rate_limited", "cancelled", "unknown"})
SKIP_REASONS = frozenset(
    {
        "blank_line",
        "missing_message_content",
        "missing_usage",
        "non_chat_call_type",
        "non_chat_request",
        "non_chat_request_body",
    }
)
FORMAT_IDS = (
    "openai-jsonl",
    "anthropic-jsonl",
    "litellm-jsonl",
    "helicone-export",
    "openrouter-export",
    "generic-jsonl",
)

_SYSTEM_ROLES = frozenset({"developer", "system"})
_MIN_TIMESTAMP = 1_000_000_000.0
_MAX_TIMESTAMP = 4_000_000_000.0
_LITELLM_CHAT_CALLS = frozenset({"acompletion", "chat", "completion", "streaming_completion"})
_MISSING: Any = object()

_TEXT_FIX = (
    "export the request messages with the log, or read the log with format='generic-jsonl' "
    "and a mapping that points at the message array"
)


class IngestError(ValueError):
    """Base class for every public ingest failure. Never carries prompt text."""


class SourceNotFoundError(IngestError):
    """The log path does not exist or is not a regular file."""


class EmptySourceError(IngestError):
    """The log contains no JSON records."""


class MappingError(IngestError):
    """A ``generic-jsonl`` field mapping is missing, incomplete, or unusable."""


class MalformedRecordError(IngestError):
    """A structurally invalid record, named by zero-based index and offending field."""

    def __init__(self, index: int, field: str, problem: str, fix: str) -> None:
        self.index = index
        self.field = field
        super().__init__(
            f"malformed record {index} (line {index + 1}): field {field!r} {problem}; fix: {fix}"
        )


@dataclass(frozen=True, slots=True)
class Skipped:
    """A structurally valid record that carries no auditable chat request."""

    reason: str

    def __post_init__(self) -> None:
        if self.reason not in SKIP_REASONS:
            expected = ", ".join(sorted(SKIP_REASONS))
            raise IngestError(
                f"unknown skip reason {self.reason!r}; "
                f"fix: use one of the closed reason vocabulary: {expected}"
            )


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One logged LLM request, with all prompt and completion text already discarded."""

    id: str
    timestamp: float
    model: str
    provider: str
    messages_hash: str
    system_prefix_hash: str
    prompt_tokens: int
    cached_prompt_tokens: int | None
    completion_tokens: int
    latency_ms: float | None
    status: str
    group_key: str | None
    raw_index: int
    # Declared last with a default so hand-built records stay valid. Every reader in this
    # module populates it explicitly; the default is never a data path.
    system_prefix_chars: int = 0

    def __post_init__(self) -> None:
        for name in ("id", "model", "provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise IngestError(
                    f"RequestRecord {name} must be a non-empty string; "
                    f"fix: map a non-empty {name} for every record"
                )
        if self.provider != self.provider.lower():
            raise IngestError(
                f"RequestRecord provider {self.provider!r} must be lowercase; "
                "fix: normalize provider names with str.lower() so that 'OpenAI' and "
                "'openai' cannot split into two providers"
            )
        _check_timestamp(self.timestamp)
        for name in ("messages_hash", "system_prefix_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX32.match(value):
                raise IngestError(
                    f"RequestRecord {name} must be 32 lowercase hex characters; "
                    "fix: build it with sha256(...).digest()[:16].hex()"
                )
        for name in ("prompt_tokens", "completion_tokens", "raw_index", "system_prefix_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise IngestError(
                    f"RequestRecord {name} must be an integer; fix: convert {name} to int"
                )
            if value < 0:
                raise IngestError(
                    f"RequestRecord {name} cannot be negative; "
                    f"fix: drop records whose {name} is negative instead of forwarding them"
                )
        _check_cached_tokens(self.cached_prompt_tokens, self.prompt_tokens)
        _check_latency(self.latency_ms)
        if self.status not in STATUSES:
            expected = ", ".join(sorted(STATUSES))
            raise IngestError(
                f"RequestRecord status {self.status!r} is not a known status; "
                f"fix: map the source outcome to one of: {expected}"
            )
        if self.group_key is not None and (
            not isinstance(self.group_key, str) or not self.group_key.strip()
        ):
            raise IngestError(
                "RequestRecord group_key must be a non-empty string or None; "
                "fix: use None when the log has no grouping field"
            )


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Parsed and skipped counts plus a closed-vocabulary reason histogram."""

    parsed: int
    skipped: int
    reasons: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in ("parsed", "skipped"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise IngestError(
                    f"IngestReport {name} must be an integer; fix: pass an int count for {name}"
                )
            if value < 0:
                raise IngestError(
                    f"IngestReport {name} cannot be negative; fix: pass a non-negative {name}"
                )
        if not isinstance(self.reasons, Mapping):
            raise IngestError(
                "IngestReport reasons must be a mapping; fix: pass a reason -> count mapping"
            )
        unknown = sorted(set(self.reasons) - SKIP_REASONS)
        if unknown:
            expected = ", ".join(sorted(SKIP_REASONS))
            raise IngestError(
                f"IngestReport reasons has unknown keys: {', '.join(unknown)}; "
                f"fix: use only the closed reason vocabulary: {expected}"
            )
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in self.reasons.values()
        ):
            raise IngestError(
                "IngestReport reason counts must be positive integers; "
                "fix: omit reasons that never occurred instead of recording a zero"
            )
        total = sum(self.reasons.values())
        if total != self.skipped:
            raise IngestError(
                f"IngestReport reason counts sum to {total} but skipped is {self.skipped}; "
                "fix: record exactly one reason for every skipped record"
            )

    def total(self) -> int:
        """Records seen, parsed plus skipped."""
        return self.parsed + self.skipped


def _check_timestamp(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IngestError(
            "RequestRecord timestamp must be a real number of unix epoch seconds; "
            "fix: convert the log's time field to epoch seconds"
        )
    if not math.isfinite(value):
        raise IngestError(
            "RequestRecord timestamp must be finite; "
            "fix: convert the log's time field to epoch seconds"
        )
    if value <= 0:
        raise IngestError(
            "RequestRecord timestamp must be a positive unix epoch value; fix: a zero or "
            "negative timestamp means the log's time field was not parsed -- point the "
            "reader at the correct field"
        )


def _check_cached_tokens(cached: object, prompt_tokens: int) -> None:
    if cached is None:
        return
    if isinstance(cached, bool) or not isinstance(cached, int):
        raise IngestError(
            "RequestRecord cached_prompt_tokens must be an integer or None; "
            "fix: use None when the source does not report cached prompt tokens"
        )
    if cached < 0:
        raise IngestError(
            "RequestRecord cached_prompt_tokens cannot be negative; "
            "fix: use None when the source does not report cached prompt tokens"
        )
    if cached > prompt_tokens:
        raise IngestError(
            f"RequestRecord cached_prompt_tokens ({cached}) exceeds prompt_tokens "
            f"({prompt_tokens}); fix: prompt_tokens must be the full billed prompt "
            "including its cached subset"
        )


def _check_latency(latency: object) -> None:
    if latency is None:
        return
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise IngestError(
            "RequestRecord latency_ms must be a real number or None; "
            "fix: use None when the source does not report latency"
        )
    if not math.isfinite(latency):
        raise IngestError(
            "RequestRecord latency_ms must be finite; "
            "fix: use None when the source does not report latency"
        )
    if latency < 0:
        raise IngestError(
            "RequestRecord latency_ms cannot be negative; "
            "fix: check that the log's end timestamp follows its start timestamp"
        )


# --------------------------------------------------------------------------------------
# line and record framing
# --------------------------------------------------------------------------------------


def _iter_lines(handle: Any, limit: int = MAX_RECORD_BYTES) -> Iterator[bytes]:
    """Yield newline-delimited byte records, refusing any record longer than ``limit``.

    Reads fixed-size chunks instead of using line iteration so that a single pathological
    line cannot pull an entire file into memory before the guard fires.
    """
    buffer = bytearray()
    index = 0
    while True:
        chunk = handle.read(_CHUNK_BYTES)
        if not chunk:
            break
        buffer += chunk
        start = 0
        while True:
            newline = buffer.find(b"\n", start)
            if newline < 0:
                break
            yield bytes(buffer[start:newline])
            index += 1
            start = newline + 1
        if start:
            del buffer[:start]
        if len(buffer) > limit:
            raise MalformedRecordError(
                index,
                "<line>",
                f"exceeds the {limit} byte per-record limit",
                "write one JSON object per line -- re-export the log as newline-delimited JSON",
            )
    if buffer:
        yield bytes(buffer)


def iter_json_records(handle: Any) -> Iterator[tuple[int, Mapping[str, Any] | None]]:
    """Stream ``(index, object)`` pairs from a binary handle; blank lines yield ``None``.

    ``index`` is zero-based over every line, so the source line number is ``index + 1``.
    """
    for index, raw in enumerate(_iter_lines(handle)):
        stripped = raw.strip()
        if not stripped:
            yield index, None
            continue
        try:
            text = stripped.decode("utf-8")
        except UnicodeDecodeError:
            raise MalformedRecordError(
                index, "<line>", "is not valid UTF-8", "re-export the log with UTF-8 encoding"
            ) from None
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            # Only exc.msg (a fixed template) and the column are reused: exc.doc holds the
            # raw line, so this exception is never chained, stored, or stringified.
            problem = f"is not valid JSON ({exc.msg} at column {exc.colno})"
            raise MalformedRecordError(
                index, "<line>", problem, "one complete JSON object per line, no trailing commas"
            ) from None
        del text
        if not isinstance(value, dict):
            raise MalformedRecordError(
                index,
                "<line>",
                f"is a JSON {type(value).__name__}, not a JSON object",
                "write one JSON object per line -- if the export is a single JSON array, "
                "convert it to newline-delimited JSON",
            )
        yield index, value


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------


def _frame(hasher: Any, value: str) -> None:
    payload = value.encode("utf-8")
    hasher.update(str(len(payload)).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(payload)


def _content_parts(content: Any, index: int, field: str) -> tuple[tuple[str, str], ...]:
    if content is None or content is _MISSING:
        return ()
    if isinstance(content, str):
        return (("text", content),)
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[tuple[str, str]] = []
        for position, block in enumerate(content):
            if isinstance(block, str):
                parts.append(("text", block))
                continue
            if not isinstance(block, Mapping):
                raise MalformedRecordError(
                    index,
                    f"{field}[{position}]",
                    f"is a {type(block).__name__}, not a content block object or a string",
                    "each content block must be a JSON object carrying a 'type' field",
                )
            kind = block.get("type", "text")
            if not isinstance(kind, str) or not kind:
                raise MalformedRecordError(
                    index,
                    f"{field}[{position}].type",
                    "is not a non-empty string",
                    "set each content block's 'type' to a string such as 'text'",
                )
            if kind != "text":
                parts.append((kind, ""))
                continue
            text = block.get("text")
            if not isinstance(text, str):
                raise MalformedRecordError(
                    index,
                    f"{field}[{position}].text",
                    "is not a string",
                    "a text content block must carry its text in a string 'text' field",
                )
            parts.append(("text", text))
        return tuple(parts)
    raise MalformedRecordError(
        index,
        field,
        f"is a {type(content).__name__}, not a string or a list of content blocks",
        "message content must be a string or a list of content blocks",
    )


def _frame_parts(hasher: Any, role: str, parts: Sequence[tuple[str, str]]) -> None:
    _frame(hasher, role)
    _frame(hasher, str(len(parts)))
    for kind, text in parts:
        _frame(hasher, kind)
        _frame(hasher, text)


def _update_message(hasher: Any, role: str, content: Any, index: int, field: str) -> None:
    _frame_parts(hasher, role, _content_parts(content, index, field))


def _message_triples(messages: Any, index: int, field: str) -> list[tuple[str, Any, str]]:
    """Return ``(role, content, field_path)`` triples, validating the message array shape."""
    return _generic_triples(messages, index, field, "role", "content")


def _generic_triples(
    messages: Any, index: int, field: str, role_key: str, content_key: str
) -> list[tuple[str, Any, str]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise MalformedRecordError(
            index,
            field,
            f"is a {type(messages).__name__}, not a list of messages",
            f"export the chat messages as a JSON array of "
            f"{{'{role_key}', '{content_key}'}} objects",
        )
    if not messages:
        raise MalformedRecordError(
            index, field, "is an empty list", "a chat request must carry at least one message"
        )
    triples: list[tuple[str, Any, str]] = []
    for position, message in enumerate(messages):
        path = f"{field}[{position}]"
        if not isinstance(message, Mapping):
            raise MalformedRecordError(
                index,
                path,
                f"is a {type(message).__name__}, not a message object",
                f"each message must be a JSON object with '{role_key}' and '{content_key}'",
            )
        role = message.get(role_key)
        if not isinstance(role, str) or not role.strip():
            raise MalformedRecordError(
                index,
                f"{path}.{role_key}",
                "is not a non-empty string",
                f"set each message's '{role_key}' to a string such as 'system' or 'user'",
            )
        triples.append((role.strip().lower(), message.get(content_key), f"{path}.{content_key}"))
    return triples


def _hash_messages(triples: Sequence[tuple[str, Any, str]], index: int) -> tuple[str, str, int]:
    """Hash a message list into ``(messages_hash, system_prefix_hash, system_prefix_chars)``.

    Text is consumed by the hashers and measured in place; only digests and a character
    count are returned, never a copy of the text.
    """
    full = hashlib.sha256()
    prefix = hashlib.sha256()
    prefix_messages = 0
    prefix_chars = 0
    in_prefix = True
    for role, content, path in triples:
        parts = _content_parts(content, index, path)
        _frame_parts(full, role, parts)
        if in_prefix and role in _SYSTEM_ROLES:
            _frame_parts(prefix, role, parts)
            prefix_messages += 1
            prefix_chars += sum(len(text) for _, text in parts)
        else:
            in_prefix = False
    system_prefix_hash = prefix.digest()[:_HASH_BYTES].hex() if prefix_messages else EMPTY_TEXT_HASH
    return full.digest()[:_HASH_BYTES].hex(), system_prefix_hash, prefix_chars


def _discard_completion(content: Any, index: int, field: str) -> None:
    """Hash completion text and drop the digest.

    The record schema has no completion digest field, so the digest is deliberately
    discarded: reading and discarding is strictly stronger than retaining a hash. The
    content shape is still validated, so a malformed completion is reported rather than
    silently ignored.
    """
    hasher = hashlib.sha256()
    _update_message(hasher, "assistant", content, index, field)
    hasher.digest()


# --------------------------------------------------------------------------------------
# field access helpers -- all value-based so error messages name the real source path
# --------------------------------------------------------------------------------------


def _dig(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        if (
            part.isdigit()
            and isinstance(current, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
        ):
            position = int(part)
            if position >= len(current):
                return _MISSING
            current = current[position]
            continue
        return _MISSING
    return current


def _pick(obj: Any, *paths: str) -> tuple[str, Any]:
    """Return ``(path, value)`` for the first present path, or ``(paths[-1], _MISSING)``."""
    for path in paths:
        value = _dig(obj, path)
        if value is not _MISSING and value is not None:
            return path, value
    return paths[-1], _MISSING


def _as_mapping(value: Any, path: str, index: int, fix: str) -> Mapping[str, Any]:
    if value is _MISSING:
        raise MalformedRecordError(index, path, "is missing", fix)
    if not isinstance(value, Mapping):
        raise MalformedRecordError(
            index, path, f"is a {type(value).__name__}, not a JSON object", fix
        )
    return value


def _as_text(value: Any, path: str, index: int, fix: str) -> str:
    if value is _MISSING:
        raise MalformedRecordError(index, path, "is missing", fix)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise MalformedRecordError(
        index, path, f"is a {type(value).__name__}, not a non-empty string", fix
    )


def _as_optional_text(value: Any, path: str, index: int, fix: str) -> str | None:
    if value is _MISSING or value is None:
        return None
    return _as_text(value, path, index, fix)


def _as_int(value: Any, path: str, index: int, fix: str) -> int:
    if value is _MISSING:
        raise MalformedRecordError(index, path, "is missing", fix)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedRecordError(
            index, path, f"is a {type(value).__name__}, not an integer token count", fix
        )
    if value < 0:
        raise MalformedRecordError(index, path, "is a negative token count", fix)
    return value


def _as_optional_int(value: Any, path: str, index: int, fix: str) -> int | None:
    if value is _MISSING or value is None:
        return None
    return _as_int(value, path, index, fix)


_LATENCY_FIX = "report latency in milliseconds as a finite non-negative number, or omit the field"


def _as_optional_latency(value: Any, path: str, index: int) -> float | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedRecordError(
            index,
            path,
            f"is a {type(value).__name__}, not a latency in milliseconds",
            _LATENCY_FIX,
        )
    if not math.isfinite(value) or value < 0:
        raise MalformedRecordError(
            index, path, "is not a finite non-negative latency", _LATENCY_FIX
        )
    return float(value)


_TIMESTAMP_FIX = (
    "use unix epoch seconds (not milliseconds) or an ISO-8601 timestamp with an explicit "
    "UTC offset, for example 2026-04-01T09:00:00Z"
)


def _as_timestamp(value: Any, path: str, index: int) -> float:
    if value is _MISSING or value is None:
        raise MalformedRecordError(index, path, "is missing", _TIMESTAMP_FIX)
    if isinstance(value, bool):
        raise MalformedRecordError(index, path, "is a boolean, not a timestamp", _TIMESTAMP_FIX)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds):
            raise MalformedRecordError(index, path, "is not a finite timestamp", _TIMESTAMP_FIX)
        return _check_epoch_range(seconds, path, index, "is")
    if isinstance(value, str):
        return _parse_iso8601(value, path, index)
    raise MalformedRecordError(
        index, path, f"is a {type(value).__name__}, not a timestamp", _TIMESTAMP_FIX
    )


def _check_epoch_range(seconds: float, path: str, index: int, verb: str) -> float:
    if not _MIN_TIMESTAMP <= seconds <= _MAX_TIMESTAMP:
        raise MalformedRecordError(
            index,
            path,
            f"{verb} {seconds:.0f}, outside the plausible epoch-seconds range "
            f"[{_MIN_TIMESTAMP:.0f}, {_MAX_TIMESTAMP:.0f}]",
            _TIMESTAMP_FIX,
        )
    return seconds


def _parse_iso8601(value: str, path: str, index: int) -> float:
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = f"{text[:-1]}+00:00"
    head, dot, tail = text.partition(".")
    if dot:
        digits = ""
        for char in tail:
            if not char.isdigit():
                break
            digits += char
        remainder = tail[len(digits) :]
        # datetime.fromisoformat on 3.10 accepts exactly 3 or 6 fractional digits.
        text = f"{head}.{digits[:6]:0<6s}{remainder}" if digits else f"{head}{remainder}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise MalformedRecordError(
            index, path, "is not a parseable ISO-8601 timestamp", _TIMESTAMP_FIX
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _check_epoch_range(parsed.timestamp(), path, index, "resolves to")


_STATUS_FIX = f"map the source outcome to one of: {', '.join(sorted(STATUSES))}"
_HTTP_STATUS: Mapping[int, str] = {
    408: "timeout",
    429: "rate_limited",
    499: "cancelled",
    504: "timeout",
}
_TEXT_STATUS: Mapping[str, str] = {
    "canceled": "cancelled",
    "cancelled": "cancelled",
    "error": "error",
    "failed": "error",
    "failure": "error",
    "ok": "ok",
    "rate_limited": "rate_limited",
    "succeeded": "ok",
    "success": "ok",
    "timeout": "timeout",
    "unknown": "unknown",
}


def _as_status(value: Any, path: str, index: int, default: str) -> str:
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        raise MalformedRecordError(index, path, "is a boolean, not a status", _STATUS_FIX)
    if isinstance(value, int):
        if value in (0, 200):
            return "ok"
        mapped = _HTTP_STATUS.get(value)
        if mapped is not None:
            return mapped
        if 400 <= value < 600:
            return "error"
        raise MalformedRecordError(
            index, path, f"is {value}, not a recognized HTTP status code", _STATUS_FIX
        )
    if isinstance(value, str):
        mapped = _TEXT_STATUS.get(value.strip().lower())
        if mapped is not None:
            return mapped
        raise MalformedRecordError(index, path, "is not a recognized outcome", _STATUS_FIX)
    raise MalformedRecordError(
        index, path, f"is a {type(value).__name__}, not a status", _STATUS_FIX
    )


# --------------------------------------------------------------------------------------
# format readers
# --------------------------------------------------------------------------------------

RecordReader = Callable[[Mapping[str, Any], int], "RequestRecord | Skipped"]


def read_openai_jsonl(obj: Mapping[str, Any], index: int) -> RequestRecord | Skipped:
    """Read an OpenAI-shaped proxy log line: ``{"request": {...}, "response": {...}}``."""
    request = _as_mapping(
        _dig(obj, "request"),
        "request",
        index,
        "each line needs a 'request' object holding the outbound payload",
    )
    response = _as_mapping(
        _dig(obj, "response"),
        "response",
        index,
        "each line needs a 'response' object holding the upstream payload",
    )
    messages = _dig(request, "messages")
    if messages is _MISSING or messages is None:
        if any(key in request for key in ("input", "prompt")):
            return Skipped("non_chat_request")
        raise MalformedRecordError(index, "request.messages", "is missing", _TEXT_FIX)
    raw_usage = _dig(response, "usage")
    if raw_usage is _MISSING or raw_usage is None:
        return Skipped("missing_usage")
    token_fix = "report integer 'prompt_tokens' and 'completion_tokens' in 'response.usage'"
    usage = _as_mapping(raw_usage, "response.usage", index, token_fix)
    messages_hash, system_prefix_hash, system_prefix_chars = _hash_messages(
        _message_triples(messages, index, "request.messages"), index
    )
    _discard_completion(
        _dig(response, "choices.0.message.content"), index, "response.choices[0].message.content"
    )
    id_path, id_value = _pick(obj, "response.id", "id")
    time_path, time_value = _pick(obj, "response.created", "timestamp")
    model_path, model_value = _pick(obj, "response.model", "request.model")
    return RequestRecord(
        id=_as_text(
            id_value,
            id_path,
            index,
            "give every line a request id in 'response.id' or a top-level 'id'",
        ),
        timestamp=_as_timestamp(time_value, time_path, index),
        model=_as_text(
            model_value,
            model_path,
            index,
            "record the served model in 'response.model' or 'request.model'",
        ),
        provider=(
            _as_optional_text(_dig(obj, "provider"), "provider", index, "provider must be a string")
            or "openai"
        ).lower(),
        messages_hash=messages_hash,
        system_prefix_hash=system_prefix_hash,
        system_prefix_chars=system_prefix_chars,
        prompt_tokens=_as_int(
            _dig(usage, "prompt_tokens"), "response.usage.prompt_tokens", index, token_fix
        ),
        cached_prompt_tokens=_as_optional_int(
            _dig(usage, "prompt_tokens_details.cached_tokens"),
            "response.usage.prompt_tokens_details.cached_tokens",
            index,
            token_fix,
        ),
        completion_tokens=_as_int(
            _dig(usage, "completion_tokens"), "response.usage.completion_tokens", index, token_fix
        ),
        latency_ms=_as_optional_latency(_dig(obj, "latency_ms"), "latency_ms", index),
        status=_as_status(_dig(obj, "status_code"), "status_code", index, "ok"),
        group_key=_as_optional_text(
            _dig(request, "metadata.group"),
            "request.metadata.group",
            index,
            "'request.metadata.group' must be a string",
        ),
        raw_index=index,
    )


def read_anthropic_jsonl(obj: Mapping[str, Any], index: int) -> RequestRecord | Skipped:
    """Read an Anthropic Messages-shaped log line: ``{"request": {...}, "response": {...}}``."""
    request = _as_mapping(
        _dig(obj, "request"),
        "request",
        index,
        "each line needs a 'request' object holding the outbound payload",
    )
    response = _as_mapping(
        _dig(obj, "response"),
        "response",
        index,
        "each line needs a 'response' object holding the upstream payload",
    )
    messages = _dig(request, "messages")
    if messages is _MISSING or messages is None:
        raise MalformedRecordError(index, "request.messages", "is missing", _TEXT_FIX)
    raw_usage = _dig(response, "usage")
    if raw_usage is _MISSING or raw_usage is None:
        return Skipped("missing_usage")
    token_fix = (
        "report integer 'input_tokens' and 'output_tokens' in 'response.usage'; the cache "
        "fields are optional and are added to the prompt total"
    )
    usage = _as_mapping(raw_usage, "response.usage", index, token_fix)
    triples: list[tuple[str, Any, str]] = []
    system = _dig(request, "system")
    if system is not _MISSING and system not in (None, "", []):
        triples.append(("system", system, "request.system"))
    triples.extend(_message_triples(messages, index, "request.messages"))
    messages_hash, system_prefix_hash, system_prefix_chars = _hash_messages(triples, index)
    _discard_completion(_dig(response, "content"), index, "response.content")
    input_tokens = _as_int(
        _dig(usage, "input_tokens"), "response.usage.input_tokens", index, token_fix
    )
    cache_read = _as_optional_int(
        _dig(usage, "cache_read_input_tokens"),
        "response.usage.cache_read_input_tokens",
        index,
        token_fix,
    )
    cache_creation = _as_optional_int(
        _dig(usage, "cache_creation_input_tokens"),
        "response.usage.cache_creation_input_tokens",
        index,
        token_fix,
    )
    id_path, id_value = _pick(obj, "response.id", "id")
    model_path, model_value = _pick(obj, "response.model", "request.model")
    status = (
        "error"
        if _dig(response, "type") == "error"
        else _as_status(_dig(obj, "status_code"), "status_code", index, "ok")
    )
    return RequestRecord(
        id=_as_text(
            id_value,
            id_path,
            index,
            "give every line a message id in 'response.id' or a top-level 'id'",
        ),
        timestamp=_as_timestamp(_dig(obj, "timestamp"), "timestamp", index),
        model=_as_text(
            model_value,
            model_path,
            index,
            "record the served model in 'response.model' or 'request.model'",
        ),
        provider=(
            _as_optional_text(_dig(obj, "provider"), "provider", index, "provider must be a string")
            or "anthropic"
        ).lower(),
        messages_hash=messages_hash,
        system_prefix_hash=system_prefix_hash,
        system_prefix_chars=system_prefix_chars,
        prompt_tokens=input_tokens + (cache_read or 0) + (cache_creation or 0),
        cached_prompt_tokens=cache_read,
        completion_tokens=_as_int(
            _dig(usage, "output_tokens"), "response.usage.output_tokens", index, token_fix
        ),
        latency_ms=_as_optional_latency(_dig(obj, "latency_ms"), "latency_ms", index),
        status=status,
        group_key=_as_optional_text(
            _dig(request, "metadata.group"),
            "request.metadata.group",
            index,
            "'request.metadata.group' must be a string",
        ),
        raw_index=index,
    )


def read_litellm_jsonl(obj: Mapping[str, Any], index: int) -> RequestRecord | Skipped:
    """Read a LiteLLM spend-log row (flat record with ``request_id`` and ``call_type``)."""
    call_type = _as_text(
        _dig(obj, "call_type"),
        "call_type",
        index,
        "LiteLLM spend logs carry a 'call_type' string on every row",
    )
    if call_type.lower() not in _LITELLM_CHAT_CALLS:
        return Skipped("non_chat_call_type")
    messages = _dig(obj, "messages")
    if messages is _MISSING or messages is None:
        return Skipped("missing_message_content")
    raw_prompt_tokens = _dig(obj, "prompt_tokens")
    if raw_prompt_tokens is _MISSING or raw_prompt_tokens is None:
        return Skipped("missing_usage")
    messages_hash, system_prefix_hash, system_prefix_chars = _hash_messages(
        _message_triples(messages, index, "messages"), index
    )
    _discard_completion(
        _dig(obj, "response.choices.0.message.content"),
        index,
        "response.choices[0].message.content",
    )
    token_fix = "report integer 'prompt_tokens' and 'completion_tokens' on every row"
    start = _as_timestamp(_dig(obj, "startTime"), "startTime", index)
    raw_end = _dig(obj, "endTime")
    latency_ms: float | None = None
    if raw_end is not _MISSING and raw_end is not None:
        end = _as_timestamp(raw_end, "endTime", index)
        if end < start:
            raise MalformedRecordError(
                index, "endTime", "precedes 'startTime'", "record 'endTime' at or after 'startTime'"
            )
        # Source timestamps carry at most microsecond resolution, so the difference is
        # quantized to it -- subtracting two large epoch floats otherwise leaves ~1e-4 ms
        # of representation noise in the reported latency.
        latency_ms = round((end - start) * 1_000_000.0) / 1000.0
    return RequestRecord(
        id=_as_text(
            _dig(obj, "request_id"), "request_id", index, "LiteLLM rows carry a 'request_id' string"
        ),
        timestamp=start,
        model=_as_text(_dig(obj, "model"), "model", index, "record the served model in 'model'"),
        provider=_as_text(
            _dig(obj, "custom_llm_provider"),
            "custom_llm_provider",
            index,
            "LiteLLM records the upstream in 'custom_llm_provider'; include it in the export",
        ).lower(),
        messages_hash=messages_hash,
        system_prefix_hash=system_prefix_hash,
        system_prefix_chars=system_prefix_chars,
        prompt_tokens=_as_int(raw_prompt_tokens, "prompt_tokens", index, token_fix),
        cached_prompt_tokens=_as_optional_int(
            _dig(obj, "cache_read_input_tokens"), "cache_read_input_tokens", index, token_fix
        ),
        completion_tokens=_as_int(
            _dig(obj, "completion_tokens"), "completion_tokens", index, token_fix
        ),
        latency_ms=latency_ms,
        status=_as_status(_dig(obj, "status"), "status", index, "ok"),
        group_key=_as_optional_text(
            _dig(obj, "metadata.user_api_key_alias"),
            "metadata.user_api_key_alias",
            index,
            "'metadata.user_api_key_alias' must be a string",
        ),
        raw_index=index,
    )


def read_helicone_export(obj: Mapping[str, Any], index: int) -> RequestRecord | Skipped:
    """Read a Helicone request-export row (``request_body`` / ``response_body`` columns)."""
    request_body = _as_mapping(
        _dig(obj, "request_body"),
        "request_body",
        index,
        "Helicone exports carry the raw request under 'request_body'",
    )
    response_body = _as_mapping(
        _dig(obj, "response_body"),
        "response_body",
        index,
        "Helicone exports carry the raw response under 'response_body'",
    )
    messages = _dig(request_body, "messages")
    if messages is _MISSING or messages is None:
        if any(key in request_body for key in ("input", "prompt")):
            return Skipped("non_chat_request_body")
        raise MalformedRecordError(index, "request_body.messages", "is missing", _TEXT_FIX)
    token_fix = (
        "report integer 'prompt_tokens' and 'completion_tokens' columns, or keep the usage "
        "block inside 'response_body'"
    )
    # The export's own token columns win over the stored response body when both exist.
    usage_path = "prompt_tokens" if "prompt_tokens" in obj else "response_body.usage"
    raw_usage = obj if usage_path == "prompt_tokens" else _dig(response_body, "usage")
    if raw_usage is _MISSING or raw_usage is None:
        return Skipped("missing_usage")
    usage = _as_mapping(raw_usage, usage_path, index, token_fix)
    prefix = "" if usage_path == "prompt_tokens" else "response_body.usage."
    messages_hash, system_prefix_hash, system_prefix_chars = _hash_messages(
        _message_triples(messages, index, "request_body.messages"), index
    )
    _discard_completion(
        _dig(response_body, "choices.0.message.content"),
        index,
        "response_body.choices[0].message.content",
    )
    return RequestRecord(
        id=_as_text(
            _dig(obj, "request_id"),
            "request_id",
            index,
            "Helicone exports carry a 'request_id' column",
        ),
        timestamp=_as_timestamp(_dig(obj, "request_created_at"), "request_created_at", index),
        model=_as_text(
            _dig(obj, "model"), "model", index, "Helicone exports carry a 'model' column"
        ),
        provider=_as_text(
            _dig(obj, "provider"), "provider", index, "Helicone exports carry a 'provider' column"
        ).lower(),
        messages_hash=messages_hash,
        system_prefix_hash=system_prefix_hash,
        system_prefix_chars=system_prefix_chars,
        prompt_tokens=_as_int(
            _dig(usage, "prompt_tokens"), f"{prefix}prompt_tokens", index, token_fix
        ),
        cached_prompt_tokens=_as_optional_int(
            _dig(response_body, "usage.prompt_tokens_details.cached_tokens"),
            "response_body.usage.prompt_tokens_details.cached_tokens",
            index,
            token_fix,
        ),
        completion_tokens=_as_int(
            _dig(usage, "completion_tokens"), f"{prefix}completion_tokens", index, token_fix
        ),
        latency_ms=_as_optional_latency(_dig(obj, "latency"), "latency", index),
        status=_as_status(_dig(obj, "status"), "status", index, "ok"),
        group_key=_as_optional_text(
            _dig(obj, "properties.group"),
            "properties.group",
            index,
            "'properties.group' must be a string",
        ),
        raw_index=index,
    )


def read_openrouter_export(obj: Mapping[str, Any], index: int) -> RequestRecord | Skipped:
    """Read an OpenRouter generation-export row.

    OpenRouter only includes message content when prompt logging is enabled. Rows without
    it are skipped as ``missing_message_content`` rather than hashed as an empty prompt,
    which would silently merge every request into a single prefix cluster.
    """
    messages = _dig(obj, "messages")
    if messages is _MISSING or messages is None:
        return Skipped("missing_message_content")
    messages_hash, system_prefix_hash, system_prefix_chars = _hash_messages(
        _message_triples(messages, index, "messages"), index
    )
    _discard_completion(_dig(obj, "completion"), index, "completion")
    token_fix = (
        "report integer 'native_tokens_prompt'/'native_tokens_completion' (preferred) or "
        "'tokens_prompt'/'tokens_completion'"
    )
    prompt_path, prompt_value = _pick(obj, "native_tokens_prompt", "tokens_prompt")
    completion_path, completion_value = _pick(obj, "native_tokens_completion", "tokens_completion")
    if _dig(obj, "cancelled") is True:
        status = "cancelled"
    elif _dig(obj, "error") not in (_MISSING, None):
        status = "error"
    else:
        status = _as_status(_dig(obj, "status"), "status", index, "ok")
    return RequestRecord(
        id=_as_text(_dig(obj, "id"), "id", index, "OpenRouter exports carry a generation 'id'"),
        timestamp=_as_timestamp(_dig(obj, "created_at"), "created_at", index),
        model=_as_text(
            _dig(obj, "model"), "model", index, "OpenRouter exports carry a 'model' column"
        ),
        provider=_as_text(
            _dig(obj, "provider_name"),
            "provider_name",
            index,
            "OpenRouter records the serving upstream in 'provider_name'",
        ).lower(),
        messages_hash=messages_hash,
        system_prefix_hash=system_prefix_hash,
        system_prefix_chars=system_prefix_chars,
        prompt_tokens=_as_int(prompt_value, prompt_path, index, token_fix),
        cached_prompt_tokens=_as_optional_int(
            _dig(obj, "native_tokens_cached"), "native_tokens_cached", index, token_fix
        ),
        completion_tokens=_as_int(completion_value, completion_path, index, token_fix),
        latency_ms=_as_optional_latency(_dig(obj, "latency"), "latency", index),
        status=status,
        group_key=_as_optional_text(
            _dig(obj, "app_id"), "app_id", index, "'app_id' must be a string or an integer"
        ),
        raw_index=index,
    )


# --------------------------------------------------------------------------------------
# generic-jsonl
# --------------------------------------------------------------------------------------

_GENERIC_REQUIRED = (
    "id",
    "timestamp",
    "model",
    "provider",
    "messages",
    "prompt_tokens",
    "completion_tokens",
)
_GENERIC_OPTIONAL = (
    "cached_prompt_tokens",
    "completion",
    "group_key",
    "latency_ms",
    "message_content_key",
    "message_role_key",
    "status",
)
_GENERIC_KEYS = frozenset(_GENERIC_REQUIRED + _GENERIC_OPTIONAL)
_GENERIC_KEY_NAMES = frozenset({"message_content_key", "message_role_key"})


def make_generic_reader(mapping: Mapping[str, Any] | None) -> RecordReader:
    """Build a ``generic-jsonl`` reader from an explicit field mapping.

    Mapping values are dotted paths into each record (``"usage.prompt_tokens"``; numeric
    path parts index into arrays), or ``{"const": value}`` for a literal.
    ``message_role_key`` and ``message_content_key`` are literal key names looked up
    inside each message object and default to ``"role"`` and ``"content"``.
    """
    if mapping is None:
        required = ", ".join(_GENERIC_REQUIRED)
        raise MappingError(
            "format 'generic-jsonl' cannot read a log without a field mapping; "
            f"fix: pass mapping={{...}} giving a dotted path for each of: {required}"
        )
    if not isinstance(mapping, Mapping):
        raise MappingError(
            f"mapping must be a dict, not {type(mapping).__name__}; "
            "fix: pass mapping={'id': 'trace_id', ...}"
        )
    unknown = sorted(set(mapping) - _GENERIC_KEYS)
    if unknown:
        allowed = ", ".join(sorted(_GENERIC_KEYS))
        raise MappingError(
            f"mapping has unknown keys: {', '.join(unknown)}; "
            f"fix: use only these mapping keys: {allowed}"
        )
    missing = [key for key in _GENERIC_REQUIRED if key not in mapping]
    if missing:
        raise MappingError(
            f"mapping is missing required keys: {', '.join(missing)}; "
            "fix: add a dotted path (or {'const': value}) for each missing key"
        )
    paths: dict[str, str] = {}
    literals: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            if set(value) != {"const"}:
                raise MappingError(
                    f"mapping value for {key!r} must be a dotted path string or "
                    "{'const': value}; fix: replace it with one of those two forms"
                )
            literals[key] = value["const"]
            continue
        if not isinstance(value, str) or not value.strip():
            raise MappingError(
                f"mapping value for {key!r} must be a non-empty string; "
                f"fix: set mapping[{key!r}] to a path such as 'usage.prompt_tokens'"
            )
        if key in _GENERIC_KEY_NAMES:
            literals[key] = value
        else:
            paths[key] = value.strip()
    role_key = literals.get("message_role_key", "role")
    content_key = literals.get("message_content_key", "content")
    if not isinstance(role_key, str) or not role_key:
        raise MappingError(
            "mapping value for 'message_role_key' must be a non-empty key name; "
            "fix: set it to the message field holding the role, for example 'role'"
        )
    if not isinstance(content_key, str) or not content_key:
        raise MappingError(
            "mapping value for 'message_content_key' must be a non-empty key name; "
            "fix: set it to the message field holding the text, for example 'content'"
        )
    resolved_paths: Mapping[str, str] = dict(paths)
    resolved_literals: Mapping[str, Any] = dict(literals)

    def read_generic_jsonl(obj: Mapping[str, Any], index: int) -> RequestRecord | Skipped:
        return _read_generic(obj, index, resolved_paths, resolved_literals, role_key, content_key)

    return read_generic_jsonl


def _read_generic(
    obj: Mapping[str, Any],
    index: int,
    paths: Mapping[str, str],
    literals: Mapping[str, Any],
    role_key: str,
    content_key: str,
) -> RequestRecord | Skipped:
    def fetch(key: str) -> tuple[str, Any]:
        if key in literals:
            return f"mapping[{key!r}].const", literals[key]
        path = paths.get(key)
        if path is None:
            return f"mapping[{key!r}]", _MISSING
        return path, _dig(obj, path)

    def fix(key: str, what: str) -> str:
        return f"point mapping[{key!r}] at {what}"

    messages_path, messages = fetch("messages")
    if messages is _MISSING or messages is None:
        raise MalformedRecordError(
            index, messages_path, "is missing", fix("messages", "the message array")
        )
    messages_hash, system_prefix_hash, system_prefix_chars = _hash_messages(
        _generic_triples(messages, index, messages_path, role_key, content_key), index
    )
    completion_path, completion = fetch("completion")
    _discard_completion(completion, index, completion_path)

    id_path, id_value = fetch("id")
    time_path, time_value = fetch("timestamp")
    model_path, model_value = fetch("model")
    provider_path, provider_value = fetch("provider")
    prompt_path, prompt_value = fetch("prompt_tokens")
    completion_tokens_path, completion_tokens_value = fetch("completion_tokens")
    cached_path, cached_value = fetch("cached_prompt_tokens")
    latency_path, latency_value = fetch("latency_ms")
    status_path, status_value = fetch("status")
    group_path, group_value = fetch("group_key")
    return RequestRecord(
        id=_as_text(id_value, id_path, index, fix("id", "a non-empty request id")),
        timestamp=_as_timestamp(time_value, time_path, index),
        model=_as_text(model_value, model_path, index, fix("model", "a non-empty model name")),
        provider=_as_text(
            provider_value, provider_path, index, fix("provider", "a non-empty provider name")
        ).lower(),
        messages_hash=messages_hash,
        system_prefix_hash=system_prefix_hash,
        system_prefix_chars=system_prefix_chars,
        prompt_tokens=_as_int(
            prompt_value, prompt_path, index, fix("prompt_tokens", "an integer token count")
        ),
        cached_prompt_tokens=_as_optional_int(
            cached_value,
            cached_path,
            index,
            fix("cached_prompt_tokens", "an integer token count"),
        ),
        completion_tokens=_as_int(
            completion_tokens_value,
            completion_tokens_path,
            index,
            fix("completion_tokens", "an integer token count"),
        ),
        latency_ms=_as_optional_latency(latency_value, latency_path, index),
        # Nothing is assumed about an unmapped outcome: it stays 'unknown' rather than
        # being reported as a successful request.
        status=_as_status(status_value, status_path, index, "unknown"),
        group_key=_as_optional_text(
            group_value, group_path, index, fix("group_key", "a non-empty string")
        ),
        raw_index=index,
    )


RECORD_READERS: Mapping[str, RecordReader] = {
    "openai-jsonl": read_openai_jsonl,
    "anthropic-jsonl": read_anthropic_jsonl,
    "litellm-jsonl": read_litellm_jsonl,
    "helicone-export": read_helicone_export,
    "openrouter-export": read_openrouter_export,
}
