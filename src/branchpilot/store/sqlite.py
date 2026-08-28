"""The default store: stdlib :mod:`sqlite3` driven from asyncio worker threads.

Design notes worth keeping:

* One connection, one :class:`asyncio.Lock`, every statement executed inside
  :func:`asyncio.to_thread`. Serializing all access is what makes ``database is locked``
  unreachable from inside the process instead of merely unlikely; WAL and the busy timeout cover
  the other processes (CLI, a second gateway) that may share the file.
* No async sqlite dependency. ``sqlite3`` plus ``to_thread`` is the whole runtime.
* Money lives in TEXT columns and is converted with :func:`branchpilot.store.base.money_value`.
  There is no REAL money column anywhere in the schema.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from branchpilot.store.base import (
    RollupAccumulator,
    StoreError,
    apply_batch_patch,
    batch_mapping,
    batch_row_of,
    bucket_decision,
    cache_mapping,
    finite_float,
    identifier,
    normalize_batch,
    normalize_cache_entry,
    normalize_ledger_records,
    rollup_window,
)
from branchpilot.store.migrations import migrations_for

MEMORY_PATH = ":memory:"
_MIN_SQLITE_VERSION = (3, 24, 0)
_ROLLUP_CHUNK = 1000

_CREATE_SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at_s REAL NOT NULL
)
"""

_SELECT_SCHEMA_MIGRATIONS = """
SELECT version, name, checksum FROM schema_migrations ORDER BY version
"""

_INSERT_SCHEMA_MIGRATION = """
INSERT INTO schema_migrations (version, name, checksum, applied_at_s) VALUES (?, ?, ?, ?)
"""

_INSERT_LEDGER = """
INSERT INTO ledger (
    record_id, ts, principal, lever, model, mode,
    prompt_tokens, completion_tokens, cost_text, baseline_cost_text, request_hash, attrs
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (record_id) DO NOTHING
"""

_SELECT_LEDGER_WINDOW = """
SELECT lever, mode, prompt_tokens, completion_tokens, cost_text, baseline_cost_text
FROM ledger
WHERE ts >= ? AND ts < ?
ORDER BY seq
"""

_SELECT_CACHE_ENTRY = """
SELECT key, principal, created_s, expires_s, payload FROM cache_entries WHERE key = ?
"""

_UPSERT_CACHE_ENTRY = """
INSERT INTO cache_entries (key, principal, created_s, expires_s, payload)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (key) DO UPDATE SET
    principal = excluded.principal,
    created_s = excluded.created_s,
    expires_s = excluded.expires_s,
    payload = excluded.payload
"""

_DELETE_EXPIRED_CACHE = "DELETE FROM cache_entries WHERE expires_s <= ?"

_SELECT_BATCH = """
SELECT batch_id, principal, provider, upstream_batch_id, status, pending, attempts,
       created_s, updated_s, state
FROM batches
WHERE batch_id = ?
"""

_SELECT_PENDING_BATCHES = """
SELECT batch_id, principal, provider, upstream_batch_id, status, pending, attempts,
       created_s, updated_s, state
FROM batches
WHERE pending = 1
ORDER BY created_s, batch_id
"""

_INSERT_BATCH = """
INSERT INTO batches (
    batch_id, principal, provider, upstream_batch_id, status, pending, attempts,
    created_s, updated_s, state
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_BATCH = """
UPDATE batches
SET upstream_batch_id = ?, status = ?, pending = ?, attempts = ?, updated_s = ?, state = ?
WHERE batch_id = ?
"""

_SELECT_BUCKET = "SELECT tokens, updated_s FROM rate_buckets WHERE principal = ?"

_UPSERT_BUCKET = """
INSERT INTO rate_buckets (principal, tokens, updated_s) VALUES (?, ?, ?)
ON CONFLICT (principal) DO UPDATE SET tokens = excluded.tokens, updated_s = excluded.updated_s
"""


def _missing_schema(exc: sqlite3.OperationalError) -> bool:
    return "no such table" in str(exc)


class SQLiteStore:
    """Single-node persistence with zero external services."""

    __slots__ = (
        "_busy_timeout_s",
        "_clock",
        "_closed",
        "_connection",
        "_lock",
        "_loop",
        "_path",
    )

    dialect = "sqlite"

    def __init__(
        self,
        path: str | Path = MEMORY_PATH,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_s: float = 5.0,
    ) -> None:
        if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
            required = ".".join(str(part) for part in _MIN_SQLITE_VERSION)
            raise StoreError(
                f"SQLite {sqlite3.sqlite_version} cannot run upserts; branchpilot needs "
                f"{required} or newer; fix: install a Python built against SQLite {required}+, "
                "or point the store at Postgres with the 'postgres' extra"
            )
        if not callable(clock):
            raise TypeError("clock must be a zero-argument callable; fix: pass time.time")
        self._path = MEMORY_PATH if str(path) == MEMORY_PATH else str(Path(path))
        self._clock = clock
        self._busy_timeout_s = finite_float(busy_timeout_s, field="busy_timeout_s", minimum=0.001)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    @property
    def path(self) -> str:
        return self._path

    # -- plumbing ---------------------------------------------------------------------------

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise StoreError(
                "this store is bound to the event loop that first used it; "
                "fix: create one store per event loop, or reuse a single loop"
            )

    def _open_connection(self) -> sqlite3.Connection:
        if self._path != MEMORY_PATH:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_s,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        try:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if self._path != MEMORY_PATH and str(mode).lower() != "wal":
                raise StoreError(
                    f"SQLite refused write-ahead logging on {self._path!r} and reported "
                    f"journal_mode={mode!r}; fix: place the database on a local filesystem "
                    "(network filesystems cannot host WAL) or use the 'postgres' extra"
                )
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
        except BaseException:
            connection.close()
            raise
        return connection

    def _connection_for_thread(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._open_connection()
        return self._connection

    def _invoke(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        connection = self._connection_for_thread()
        try:
            return operation(connection)
        except sqlite3.OperationalError as exc:
            if _missing_schema(exc):
                raise StoreError(
                    "the store schema is missing; "
                    "fix: call `await store.migrate()` once at startup, it is idempotent"
                ) from exc
            raise

    async def _run(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        if self._closed:
            raise StoreError(
                "this store is closed; fix: open a new store with open_store(...) instead of "
                "reusing a closed one"
            )
        self._bind_loop()
        async with self._lock:
            if self._closed:
                raise StoreError(
                    "this store is closed; fix: open a new store with open_store(...) instead "
                    "of reusing a closed one"
                )
            return await asyncio.to_thread(self._invoke, operation)

    # -- lifecycle --------------------------------------------------------------------------

    async def migrate(self) -> int:
        return await self._run(self._migrate)

    def _migrate(self, connection: sqlite3.Connection) -> int:
        migrations = migrations_for(self.dialect)
        latest = migrations[-1].version
        connection.execute(_CREATE_SCHEMA_MIGRATIONS)
        rows = connection.execute(_SELECT_SCHEMA_MIGRATIONS).fetchall()
        applied = {int(row["version"]): (row["name"], row["checksum"]) for row in rows}
        if applied and sorted(applied) != list(range(1, max(applied) + 1)):
            raise StoreError(
                f"schema_migrations records a gap: applied versions {sorted(applied)}; "
                "fix: restore the database from backup, or recreate it from empty"
            )
        newer = sorted(version for version in applied if version > latest)
        if newer:
            raise StoreError(
                f"the database is at schema version {max(newer)} but this branchpilot only "
                f"knows {latest}; fix: upgrade branchpilot, migrations are forward-only"
            )
        for migration in migrations:
            record = applied.get(migration.version)
            if record is not None:
                if record[1] != migration.checksum:
                    raise StoreError(
                        f"migration {migration.version} ({record[0]}) was applied with different "
                        "SQL than this build ships; fix: reinstall the branchpilot version that "
                        "created this database, or recreate the database from empty"
                    )
                continue
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    _INSERT_SCHEMA_MIGRATION,
                    (migration.version, migration.name, migration.checksum, self._clock()),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return latest

    async def close(self) -> None:
        if self._closed:
            return
        self._bind_loop()
        async with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
        if connection is not None:
            await asyncio.to_thread(connection.close)

    # -- ledger -----------------------------------------------------------------------------

    async def append_ledger(self, records: Sequence[Mapping[str, Any]]) -> None:
        rows = normalize_ledger_records(records)
        if not rows:
            return
        params = [row.params for row in rows]

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(_INSERT_LEDGER, params)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        await self._run(operation)

    async def rollup_ledger(self, *, start: float, end: float) -> Mapping[str, Any]:
        window_start, window_end = rollup_window(start, end)

        def operation(connection: sqlite3.Connection) -> Mapping[str, Any]:
            cursor = connection.execute(_SELECT_LEDGER_WINDOW, (window_start, window_end))
            totals = RollupAccumulator()
            while True:
                rows = cursor.fetchmany(_ROLLUP_CHUNK)
                if not rows:
                    break
                for row in rows:
                    totals.add(row)
            return totals.freeze(window_start, window_end)

        return await self._run(operation)

    # -- cache metadata ---------------------------------------------------------------------

    async def get_cache_entry(self, key: str) -> Mapping[str, Any] | None:
        lookup = identifier(key, field="cache key", max_chars=512)
        now = finite_float(self._clock(), field="clock reading")

        def operation(connection: sqlite3.Connection) -> Mapping[str, Any] | None:
            row = connection.execute(_SELECT_CACHE_ENTRY, (lookup,)).fetchone()
            if row is None or float(row["expires_s"]) <= now:
                return None
            return cache_mapping(row)

        return await self._run(operation)

    async def put_cache_entry(self, key: str, entry: Mapping[str, Any], *, ttl_s: float) -> None:
        now = finite_float(self._clock(), field="clock reading")
        row = normalize_cache_entry(key, entry, ttl_s=ttl_s, now=now)

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(_SELECT_CACHE_ENTRY, (row.key,)).fetchone()
                if existing is not None and existing["principal"] != row.principal:
                    raise StoreError(
                        f"cache key {row.key!r} already belongs to another principal; refusing to "
                        "overwrite it so one tenant cannot poison another's cache; "
                        "fix: include the principal in the cache key"
                    )
                connection.execute(_UPSERT_CACHE_ENTRY, row.params)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        await self._run(operation)

    async def evict_expired(self, *, now: float) -> int:
        moment = finite_float(now, field="now")

        def operation(connection: sqlite3.Connection) -> int:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(_DELETE_EXPIRED_CACHE, (moment,))
                deleted = cursor.rowcount
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return max(0, deleted)

        return await self._run(operation)

    # -- batch state ------------------------------------------------------------------------

    async def create_batch(self, batch: Mapping[str, Any]) -> str:
        now = finite_float(self._clock(), field="clock reading")
        supplied = batch.get("batch_id") if isinstance(batch, Mapping) else None
        batch_id = uuid.uuid4().hex if supplied is None else identifier(supplied, field="batch_id")
        row = normalize_batch(batch, batch_id=batch_id, now=now)
        params = (*row.params[:5], int(row.pending), *row.params[6:])

        def operation(connection: sqlite3.Connection) -> str:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if connection.execute(_SELECT_BATCH, (row.batch_id,)).fetchone() is not None:
                    raise StoreError(
                        f"batch {row.batch_id!r} already exists; fix: call update_batch to change "
                        "an existing batch, or omit batch_id to have one generated"
                    )
                connection.execute(_INSERT_BATCH, params)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return row.batch_id

        return await self._run(operation)

    async def update_batch(self, batch_id: str, patch: Mapping[str, Any]) -> None:
        target = identifier(batch_id, field="batch_id")
        now = finite_float(self._clock(), field="clock reading")

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(_SELECT_BATCH, (target,)).fetchone()
                if existing is None:
                    raise StoreError(
                        f"batch {target!r} does not exist; fix: create it with create_batch "
                        "before updating it"
                    )
                updated = apply_batch_patch(batch_row_of(existing), patch, now=now)
                connection.execute(
                    _UPDATE_BATCH,
                    (
                        updated.upstream_batch_id,
                        updated.status,
                        int(updated.pending),
                        updated.attempts,
                        updated.updated_s,
                        updated.state,
                        updated.batch_id,
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        await self._run(operation)

    async def pending_batches(self) -> Sequence[Mapping[str, Any]]:
        def operation(connection: sqlite3.Connection) -> tuple[Mapping[str, Any], ...]:
            rows = connection.execute(_SELECT_PENDING_BATCHES).fetchall()
            return tuple(batch_mapping(row) for row in rows)

        return await self._run(operation)

    # -- rate limits ------------------------------------------------------------------------

    async def consume_tokens(
        self, principal: str, *, tokens: float, rate: float, burst: float, now: float
    ) -> bool:
        target = identifier(principal, field="principal")

        def operation(connection: sqlite3.Connection) -> bool:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(_SELECT_BUCKET, (target,)).fetchone()
                stored = None if row is None else (float(row["tokens"]), float(row["updated_s"]))
                decision = bucket_decision(stored, tokens=tokens, rate=rate, burst=burst, now=now)
                connection.execute(_UPSERT_BUCKET, (target, decision.tokens, decision.updated_s))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return decision.allowed

        return await self._run(operation)
