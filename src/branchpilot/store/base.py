from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

DIALECTS = ("sqlite", "postgres")
LEDGER_MODES = frozenset({"serve", "shadow"})
BATCH_STATUSES = frozenset(
    {"queued", "submitted", "in_progress", "completed", "failed", "cancelled"}
)
TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "cancelled"})
LEDGER_FIELDS = frozenset(
    {
        "record_id",
        "ts",
        "principal",
        "lever",
        "model",
        "mode",
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "baseline_cost",
        "request_hash",
        "attrs",
    }
)
CACHE_FIELDS = frozenset({"principal", "created_s", "payload"})
BATCH_FIELDS = frozenset(
    {
        "batch_id",
        "principal",
        "provider",
        "upstream_batch_id",
        "status",
        "attempts",
        "created_s",
        "updated_s",
        "state",
    }
)
BATCH_PATCH_FIELDS = frozenset({"status", "upstream_batch_id", "attempts", "updated_s", "state"})

# Keys that would carry prompt or completion text into durable storage. The store refuses them
# outright: hashing at the boundary is the only way "text is never retained" can be a property of
# the system rather than a habit of its callers.
FORBIDDEN_TEXT_KEYS = frozenset(
    {
        "answer",
        "choices",
        "completion",
        "completions",
        "content",
        "input",
        "message",
        "messages",
        "output",
        "prompt",
        "prompts",
        "query",
        "request_body",
        "response",
        "response_body",
        "system_prompt",
        "text",
    }
)

MAX_IDENTIFIER_CHARS = 128
MAX_JSON_CHARS = 65_536
MAX_JSON_DEPTH = 8
DECIMAL_TAG = "$decimal"

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class StoreError(RuntimeError):
    """A public, secret-free persistence error."""


def money_text(value: Any, *, field: str) -> str:
    """Render money for a text column. ``float`` is rejected outright, never coerced."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(
                f"{field} must be a finite Decimal, not {value}; "
                'fix: pass a finite amount such as Decimal("0.0000123456")'
            )
        if value < 0:
            raise ValueError(
                f"{field} must not be negative; fix: pass a non-negative Decimal amount"
            )
        return format(value, "f")
    if isinstance(value, str):
        raise TypeError(
            f"{field} must be a decimal.Decimal, not a string; "
            f'fix: wrap the amount, Decimal("{value}")'
        )
    raise TypeError(
        f"{field} must be a decimal.Decimal; float is rejected in cost paths because binary "
        'floats cannot represent prices exactly; fix: pass Decimal("0.0000123456")'
    )


def money_value(text: Any, *, field: str) -> Decimal:
    """Read money back out of a text column."""

    if not isinstance(text, str):
        raise StoreError(
            f"{field} was stored as {type(text).__name__} instead of text; "
            "fix: recreate the database with a current branchpilot so money columns are TEXT"
        )
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise StoreError(
            f"{field} holds {text!r}, which is not a decimal number; "
            "fix: repair or drop the affected row, then re-ingest it"
        ) from exc
    if not value.is_finite():
        raise StoreError(
            f"{field} holds the non-finite amount {text!r}; "
            "fix: repair or drop the affected row, then re-ingest it"
        )
    return value


def _reject_forbidden(value: Any, *, field: str, depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError(
            f"{field} nests deeper than {MAX_JSON_DEPTH} levels; "
            "fix: flatten the payload before storing it"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{field} keys must be strings, found {type(key).__name__}; "
                    "fix: convert the key to a string before storing it"
                )
            if key.lower() in FORBIDDEN_TEXT_KEYS:
                raise ValueError(
                    f"{field} must not contain the key {key!r}: prompt and completion text is "
                    "never retained; fix: hash the text and store the digest instead, e.g. "
                    "request_hash=hashlib.sha256(body).hexdigest()"
                )
            _reject_forbidden(item, field=f"{field}.{key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden(item, field=f"{field}[{index}]", depth=depth + 1)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"{field} must be a finite number, not {value}; "
            "fix: replace the value with a finite number or omit the field"
        )
    if not isinstance(value, (str, bool, int, float, Decimal, type(None))):
        raise TypeError(
            f"{field} must be JSON-encodable, found {type(value).__name__}; "
            "fix: convert the value to a string, number, Decimal, bool, list, or mapping"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {DECIMAL_TAG: money_text(value, field="decimal value")}
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def encode_json(value: Any, *, field: str) -> str:
    """Encode a metadata payload, preserving ``Decimal`` exactly and refusing retained text."""

    _reject_forbidden(value, field=field, depth=0)
    text = json.dumps(_jsonable(value), allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(text) > MAX_JSON_CHARS:
        raise ValueError(
            f"{field} encodes to {len(text)} characters, above the {MAX_JSON_CHARS} limit; "
            "fix: store a digest or an external reference instead of the full payload"
        )
    return text


def _decimal_hook(payload: dict[str, Any]) -> Any:
    if len(payload) == 1 and DECIMAL_TAG in payload:
        return money_value(payload[DECIMAL_TAG], field="stored decimal value")
    return payload


def decode_json(text: Any, *, field: str) -> Any:
    """Decode a metadata payload written by :func:`encode_json`."""

    if not isinstance(text, str):
        raise StoreError(
            f"{field} was stored as {type(text).__name__} instead of text; "
            "fix: recreate the database with a current branchpilot"
        )
    try:
        return json.loads(text, object_hook=_decimal_hook)
    except json.JSONDecodeError as exc:
        raise StoreError(
            f"{field} does not hold JSON; fix: repair or drop the affected row"
        ) from exc


def identifier(value: Any, *, field: str, max_chars: int = MAX_IDENTIFIER_CHARS) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a string, found {type(value).__name__}; "
            "fix: pass the identifier as a string"
        )
    if not value or value.strip() != value:
        raise ValueError(
            f"{field} must be a non-empty string without surrounding whitespace; "
            "fix: pass a trimmed, non-empty identifier"
        )
    if len(value) > max_chars:
        raise ValueError(
            f"{field} is {len(value)} characters, above the {max_chars} limit; "
            f"fix: shorten it to at most {max_chars} characters"
        )
    if any(character.isspace() or ord(character) < 0x20 for character in value):
        raise ValueError(
            f"{field} must not contain whitespace or control characters; "
            "fix: use a slug such as 'tenant-a' or a hex digest"
        )
    return value


def finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field} must be a real number, found {type(value).__name__}; "
            "fix: pass a float such as time.time()"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field} must be finite, not {number}; fix: pass a finite number of seconds"
        )
    if minimum is not None and number < minimum:
        raise ValueError(
            f"{field} must be at least {minimum}, not {number}; "
            f"fix: pass a value greater than or equal to {minimum}"
        )
    return number


def counter(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field} must be an integer, found {type(value).__name__}; "
            "fix: pass a whole token count"
        )
    if value < 0:
        raise ValueError(f"{field} must not be negative, got {value}; fix: pass a count >= 0")
    return value


def _unknown_keys(record: Mapping[str, Any], allowed: frozenset[str], *, field: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        expected = ", ".join(sorted(allowed))
        raise ValueError(
            f"{field} has unsupported key(s): {', '.join(unknown)}; "
            f"fix: move card-specific fields into the free-form payload, or use one of: {expected}"
        )


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One durable ledger row, money already rendered as text."""

    record_id: str
    ts: float
    principal: str
    lever: str
    model: str
    mode: str
    prompt_tokens: int
    completion_tokens: int
    cost_text: str
    baseline_cost_text: str
    request_hash: str
    attrs: str

    @property
    def params(self) -> tuple[Any, ...]:
        return (
            self.record_id,
            self.ts,
            self.principal,
            self.lever,
            self.model,
            self.mode,
            self.prompt_tokens,
            self.completion_tokens,
            self.cost_text,
            self.baseline_cost_text,
            self.request_hash,
            self.attrs,
        )


@dataclass(frozen=True, slots=True)
class CacheRow:
    key: str
    principal: str
    created_s: float
    expires_s: float
    payload: str

    @property
    def params(self) -> tuple[Any, ...]:
        return (self.key, self.principal, self.created_s, self.expires_s, self.payload)


@dataclass(frozen=True, slots=True)
class BatchRow:
    batch_id: str
    principal: str
    provider: str
    upstream_batch_id: str | None
    status: str
    pending: bool
    attempts: int
    created_s: float
    updated_s: float
    state: str

    @property
    def params(self) -> tuple[Any, ...]:
        return (
            self.batch_id,
            self.principal,
            self.provider,
            self.upstream_batch_id,
            self.status,
            self.pending,
            self.attempts,
            self.created_s,
            self.updated_s,
            self.state,
        )


def normalize_ledger_records(records: Any) -> tuple[LedgerRow, ...]:
    """Validate a ledger batch. Raises before any row is written."""

    if isinstance(records, (str, bytes, bytearray, Mapping)) or not isinstance(records, Sequence):
        raise TypeError(
            "append_ledger expects a sequence of mappings, found "
            f"{type(records).__name__}; fix: pass a list of ledger records, e.g. [record]"
        )
    rows: list[LedgerRow] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        row = _normalize_ledger_record(record, index=index)
        if row.record_id in seen:
            raise ValueError(
                f"ledger record[{index}] repeats record_id {row.record_id!r} inside one batch; "
                "fix: give every record a unique record_id before appending"
            )
        seen.add(row.record_id)
        rows.append(row)
    return tuple(rows)


def _normalize_ledger_record(record: Any, *, index: int) -> LedgerRow:
    field = f"ledger record[{index}]"
    if not isinstance(record, Mapping):
        raise TypeError(
            f"{field} must be a mapping, found {type(record).__name__}; "
            "fix: pass a dict of ledger fields"
        )
    _unknown_keys(record, LEDGER_FIELDS, field=field)
    missing = sorted(LEDGER_FIELDS - {"attrs"} - set(record))
    if missing:
        raise ValueError(
            f"{field} is missing required field(s): {', '.join(missing)}; "
            f"fix: supply every required ledger field: {', '.join(sorted(LEDGER_FIELDS))}"
        )
    mode = record["mode"]
    if mode not in LEDGER_MODES:
        expected = ", ".join(sorted(LEDGER_MODES))
        raise ValueError(
            f"{field}.mode must be one of: {expected}; "
            "fix: record shadow traffic as 'shadow' and served traffic as 'serve'"
        )
    request_hash = record["request_hash"]
    if not isinstance(request_hash, str) or not _HEX64.match(request_hash):
        raise ValueError(
            f"{field}.request_hash must be a lowercase 64-character sha256 hex digest; "
            "fix: store hashlib.sha256(canonical_request).hexdigest(), never request text"
        )
    return LedgerRow(
        record_id=identifier(record["record_id"], field=f"{field}.record_id"),
        ts=finite_float(record["ts"], field=f"{field}.ts", minimum=0.0),
        principal=identifier(record["principal"], field=f"{field}.principal"),
        lever=identifier(record["lever"], field=f"{field}.lever"),
        model=identifier(record["model"], field=f"{field}.model", max_chars=256),
        mode=mode,
        prompt_tokens=counter(record["prompt_tokens"], field=f"{field}.prompt_tokens"),
        completion_tokens=counter(record["completion_tokens"], field=f"{field}.completion_tokens"),
        cost_text=money_text(record["cost"], field=f"{field}.cost"),
        baseline_cost_text=money_text(record["baseline_cost"], field=f"{field}.baseline_cost"),
        request_hash=request_hash,
        attrs=encode_json(record.get("attrs", {}), field=f"{field}.attrs"),
    )


def normalize_cache_entry(key: Any, entry: Any, *, ttl_s: Any, now: float) -> CacheRow:
    """Validate a cache-metadata row. A finite, positive TTL is mandatory."""

    if not isinstance(entry, Mapping):
        raise TypeError(
            f"cache entry must be a mapping, found {type(entry).__name__}; "
            "fix: pass a dict with 'principal' and 'payload'"
        )
    _unknown_keys(entry, CACHE_FIELDS, field="cache entry")
    ttl = finite_float(ttl_s, field="ttl_s")
    if ttl <= 0:
        raise ValueError(
            f"ttl_s must be greater than zero, got {ttl}; "
            "fix: set an explicit per-route TTL such as ttl_s=300; there is no infinite default"
        )
    if "principal" not in entry:
        raise ValueError(
            "cache entry is missing 'principal'; fix: scope every cache entry to a principal so "
            "one tenant can never read another tenant's cached response"
        )
    created_s = (
        now if "created_s" not in entry else finite_float(entry["created_s"], field="created_s")
    )
    return CacheRow(
        key=identifier(key, field="cache key", max_chars=512),
        principal=identifier(entry["principal"], field="cache entry principal"),
        created_s=created_s,
        expires_s=created_s + ttl,
        payload=encode_json(entry.get("payload", {}), field="cache entry payload"),
    )


def normalize_batch(batch: Any, *, batch_id: str, now: float) -> BatchRow:
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"batch must be a mapping, found {type(batch).__name__}; "
            "fix: pass a dict with 'principal' and 'provider'"
        )
    _unknown_keys(batch, BATCH_FIELDS, field="batch")
    missing = sorted({"principal", "provider"} - set(batch))
    if missing:
        raise ValueError(
            f"batch is missing required field(s): {', '.join(missing)}; "
            "fix: supply principal and provider when creating a batch"
        )
    status = batch.get("status", "queued")
    created_s = (
        now if "created_s" not in batch else finite_float(batch["created_s"], field="created_s")
    )
    updated_s = (
        created_s
        if "updated_s" not in batch
        else finite_float(batch["updated_s"], field="updated_s")
    )
    return BatchRow(
        batch_id=batch_id,
        principal=identifier(batch["principal"], field="batch principal"),
        provider=identifier(batch["provider"], field="batch provider"),
        upstream_batch_id=(
            None
            if batch.get("upstream_batch_id") is None
            else identifier(batch["upstream_batch_id"], field="batch upstream_batch_id")
        ),
        status=batch_status(status),
        pending=status not in TERMINAL_BATCH_STATUSES,
        attempts=counter(batch.get("attempts", 0), field="batch attempts"),
        created_s=created_s,
        updated_s=updated_s,
        state=encode_json(batch.get("state", {}), field="batch state"),
    )


def batch_status(status: Any) -> str:
    if status not in BATCH_STATUSES:
        expected = ", ".join(sorted(BATCH_STATUSES))
        raise ValueError(
            f"batch status must be one of: {expected}; "
            "fix: use 'queued' for new work and a terminal status only once the batch settles"
        )
    return status


def apply_batch_patch(current: BatchRow, patch: Any, *, now: float) -> BatchRow:
    """Fold a partial update into a full row so the write stays one fixed statement."""

    if not isinstance(patch, Mapping):
        raise TypeError(
            f"batch patch must be a mapping, found {type(patch).__name__}; "
            "fix: pass a dict such as {'status': 'completed'}"
        )
    if not patch:
        raise ValueError(
            "batch patch is empty; fix: pass at least one of: "
            f"{', '.join(sorted(BATCH_PATCH_FIELDS))}"
        )
    _unknown_keys(patch, BATCH_PATCH_FIELDS, field="batch patch")
    status = batch_status(patch["status"]) if "status" in patch else current.status
    upstream = current.upstream_batch_id
    if "upstream_batch_id" in patch:
        upstream = (
            None
            if patch["upstream_batch_id"] is None
            else identifier(patch["upstream_batch_id"], field="batch upstream_batch_id")
        )
    return BatchRow(
        batch_id=current.batch_id,
        principal=current.principal,
        provider=current.provider,
        upstream_batch_id=upstream,
        status=status,
        pending=status not in TERMINAL_BATCH_STATUSES,
        attempts=(
            current.attempts
            if "attempts" not in patch
            else counter(patch["attempts"], field="batch attempts")
        ),
        created_s=current.created_s,
        updated_s=(
            now if "updated_s" not in patch else finite_float(patch["updated_s"], field="updated_s")
        ),
        state=(
            current.state
            if "state" not in patch
            else encode_json(patch["state"], field="batch state")
        ),
    )


@dataclass(frozen=True, slots=True)
class BucketDecision:
    """The outcome of one token-bucket admission check."""

    allowed: bool
    tokens: float
    updated_s: float


def bucket_decision(
    stored: tuple[float, float] | None,
    *,
    tokens: float,
    rate: float,
    burst: float,
    now: float,
) -> BucketDecision:
    """Pure token-bucket step, shared by every backend so the semantics cannot diverge."""

    requested = finite_float(tokens, field="tokens")
    refill_rate = finite_float(rate, field="rate", minimum=0.0)
    capacity = finite_float(burst, field="burst")
    moment = finite_float(now, field="now")
    if requested <= 0:
        raise ValueError(
            f"tokens must be greater than zero, got {requested}; "
            "fix: charge at least one token per admission check"
        )
    if capacity <= 0:
        raise ValueError(
            f"burst must be greater than zero, got {capacity}; "
            "fix: set burst to the largest single request you intend to admit"
        )
    if requested > capacity:
        raise ValueError(
            f"tokens ({requested}) exceeds burst ({capacity}), so this request could never be "
            "admitted; fix: raise burst to at least the largest request cost, or charge fewer "
            "tokens per request"
        )
    if stored is None:
        available = capacity
    else:
        # A clock that moved backwards (process restart, monotonic epoch change) must never mint
        # tokens, so elapsed time is clamped at zero and the refill is capped at the burst.
        elapsed = max(0.0, moment - stored[1])
        available = min(capacity, stored[0] + elapsed * refill_rate)
    if available + 1e-9 < requested:
        return BucketDecision(allowed=False, tokens=available, updated_s=moment)
    return BucketDecision(allowed=True, tokens=available - requested, updated_s=moment)


def rollup_window(start: Any, end: Any) -> tuple[float, float]:
    window_start = finite_float(start, field="start")
    window_end = finite_float(end, field="end")
    if window_end < window_start:
        raise ValueError(
            f"end ({window_end}) must not precede start ({window_start}); "
            "fix: pass start <= end, both as epoch seconds"
        )
    return window_start, window_end


class RollupAccumulator:
    """Sums ledger money in :class:`~decimal.Decimal`, shared by every backend.

    SQL ``SUM`` is never used on a money column: SQLite would silently coerce the stored text to
    a binary float, and Postgres would refuse the aggregate outright. Rows are folded in bounded
    chunks so a wide window cannot exhaust memory.
    """

    __slots__ = (
        "_baseline",
        "_by_lever",
        "_by_mode",
        "_completion_tokens",
        "_cost",
        "_prompt_tokens",
        "_records",
    )

    def __init__(self) -> None:
        self._records = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = Decimal(0)
        self._baseline = Decimal(0)
        self._by_lever: dict[str, dict[str, Any]] = {}
        self._by_mode: dict[str, dict[str, Any]] = {}

    def add(self, row: Any) -> None:
        cost = money_value(row["cost_text"], field="ledger cost_text")
        baseline = money_value(row["baseline_cost_text"], field="ledger baseline_cost_text")
        self._records += 1
        self._prompt_tokens += int(row["prompt_tokens"])
        self._completion_tokens += int(row["completion_tokens"])
        self._cost += cost
        self._baseline += baseline
        for group, key in ((self._by_lever, row["lever"]), (self._by_mode, row["mode"])):
            bucket = group.setdefault(
                key, {"records": 0, "cost": Decimal(0), "baseline_cost": Decimal(0)}
            )
            bucket["records"] += 1
            bucket["cost"] += cost
            bucket["baseline_cost"] += baseline

    def freeze(self, start: float, end: float) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "start": start,
                "end": end,
                "records": self._records,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cost": self._cost,
                "baseline_cost": self._baseline,
                "savings": self._baseline - self._cost,
                "by_lever": _freeze_groups(self._by_lever),
                "by_mode": _freeze_groups(self._by_mode),
            }
        )


def _freeze_groups(groups: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType(
        {
            key: MappingProxyType(
                {
                    "records": bucket["records"],
                    "cost": bucket["cost"],
                    "baseline_cost": bucket["baseline_cost"],
                    "savings": bucket["baseline_cost"] - bucket["cost"],
                }
            )
            for key, bucket in groups.items()
        }
    )


def cache_mapping(row: Any) -> Mapping[str, Any]:
    """Render a stored cache row for callers, decoding the payload back to Python."""

    return MappingProxyType(
        {
            "key": row["key"],
            "principal": row["principal"],
            "created_s": float(row["created_s"]),
            "expires_s": float(row["expires_s"]),
            "payload": decode_json(row["payload"], field="cache entry payload"),
        }
    )


def batch_row_of(row: Any) -> BatchRow:
    """Rebuild a :class:`BatchRow` from a backend result row."""

    return BatchRow(
        batch_id=row["batch_id"],
        principal=row["principal"],
        provider=row["provider"],
        upstream_batch_id=row["upstream_batch_id"],
        status=row["status"],
        pending=bool(row["pending"]),
        attempts=int(row["attempts"]),
        created_s=float(row["created_s"]),
        updated_s=float(row["updated_s"]),
        state=row["state"],
    )


def batch_mapping(row: Any) -> Mapping[str, Any]:
    """Render a stored batch row for callers, decoding its state back to Python."""

    record = batch_row_of(row)
    return MappingProxyType(
        {
            "batch_id": record.batch_id,
            "principal": record.principal,
            "provider": record.provider,
            "upstream_batch_id": record.upstream_batch_id,
            "status": record.status,
            "attempts": record.attempts,
            "created_s": record.created_s,
            "updated_s": record.updated_s,
            "state": decode_json(record.state, field="batch state"),
        }
    )


@runtime_checkable
class Store(Protocol):
    """The persistence surface every later card codes against.

    Every method is async, every money value crossing it is a :class:`decimal.Decimal`, and no
    method accepts or returns prompt or completion text.
    """

    async def migrate(self) -> int:
        """Apply pending forward-only migrations and return the resulting schema version."""

    async def close(self) -> None:
        """Release backend resources. Idempotent."""

    async def append_ledger(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Append ledger rows. Idempotent per ``record_id`` across calls."""

    async def rollup_ledger(self, *, start: float, end: float) -> Mapping[str, Any]:
        """Aggregate ledger rows in ``[start, end)``. Money aggregates are ``Decimal``."""

    async def get_cache_entry(self, key: str) -> Mapping[str, Any] | None:
        """Return live cache metadata, or ``None`` when absent or expired."""

    async def put_cache_entry(self, key: str, entry: Mapping[str, Any], *, ttl_s: float) -> None:
        """Upsert cache metadata. ``ttl_s`` is mandatory and must be finite and positive."""

    async def evict_expired(self, *, now: float) -> int:
        """Delete entries whose TTL elapsed at or before ``now``; return the number deleted."""

    async def create_batch(self, batch: Mapping[str, Any]) -> str:
        """Durably record a new batch and return its id."""

    async def update_batch(self, batch_id: str, patch: Mapping[str, Any]) -> None:
        """Apply a partial update to an existing batch."""

    async def pending_batches(self) -> Sequence[Mapping[str, Any]]:
        """Return unsettled batches, oldest first, so a restart can resume them."""

    async def consume_tokens(
        self, principal: str, *, tokens: float, rate: float, burst: float, now: float
    ) -> bool:
        """Atomically charge a token bucket. ``now`` comes from the caller's monotonic clock."""
