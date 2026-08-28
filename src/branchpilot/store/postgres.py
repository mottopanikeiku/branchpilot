"""Optional Postgres backend. Requires the ``postgres`` extra; the core path never imports it.

SQLite remains the default and a supported production backend for single-node deployments. This
module exists for operators who already run Postgres, not because the default needs it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
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

try:
    import asyncpg
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "branchpilot.store.postgres needs the optional 'postgres' extra; "
        "fix: install branchpilot[postgres] (asyncpg==0.31.0), or keep the default SQLite "
        'backend with open_store("sqlite:///branchpilot.db")'
    ) from exc

_ROLLUP_CHUNK = 1000

_CREATE_SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at_s DOUBLE PRECISION NOT NULL
)
"""

_SELECT_SCHEMA_MIGRATIONS = """
SELECT version, name, checksum FROM schema_migrations ORDER BY version
"""

_INSERT_SCHEMA_MIGRATION = """
INSERT INTO schema_migrations (version, name, checksum, applied_at_s) VALUES ($1, $2, $3, $4)
"""

_INSERT_LEDGER = """
INSERT INTO ledger (
    record_id, ts, principal, lever, model, mode,
    prompt_tokens, completion_tokens, cost_text, baseline_cost_text, request_hash, attrs
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
ON CONFLICT (record_id) DO NOTHING
"""

_SELECT_LEDGER_WINDOW = """
SELECT lever, mode, prompt_tokens, completion_tokens, cost_text, baseline_cost_text
FROM ledger
WHERE ts >= $1 AND ts < $2
ORDER BY seq
"""

_SELECT_CACHE_ENTRY = """
SELECT key, principal, created_s, expires_s, payload FROM cache_entries WHERE key = $1
"""

_UPSERT_CACHE_ENTRY = """
INSERT INTO cache_entries (key, principal, created_s, expires_s, payload)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (key) DO UPDATE SET
    principal = excluded.principal,
    created_s = excluded.created_s,
    expires_s = excluded.expires_s,
    payload = excluded.payload
"""

_DELETE_EXPIRED_CACHE = "DELETE FROM cache_entries WHERE expires_s <= $1"

_SELECT_BATCH = """
SELECT batch_id, principal, provider, upstream_batch_id, status, pending, attempts,
       created_s, updated_s, state
FROM batches
WHERE batch_id = $1
FOR UPDATE
"""

_SELECT_PENDING_BATCHES = """
SELECT batch_id, principal, provider, upstream_batch_id, status, pending, attempts,
       created_s, updated_s, state
FROM batches
WHERE pending
ORDER BY created_s, batch_id
"""

_INSERT_BATCH = """
INSERT INTO batches (
    batch_id, principal, provider, upstream_batch_id, status, pending, attempts,
    created_s, updated_s, state
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (batch_id) DO NOTHING
"""

_UPDATE_BATCH = """
UPDATE batches
SET upstream_batch_id = $1, status = $2, pending = $3, attempts = $4, updated_s = $5, state = $6
WHERE batch_id = $7
"""

_SEED_BUCKET = """
INSERT INTO rate_buckets (principal, tokens, updated_s) VALUES ($1, $2, $3)
ON CONFLICT (principal) DO NOTHING
"""

_SELECT_BUCKET = "SELECT tokens, updated_s FROM rate_buckets WHERE principal = $1 FOR UPDATE"

_UPDATE_BUCKET = "UPDATE rate_buckets SET tokens = $1, updated_s = $2 WHERE principal = $3"


class PostgresStore:
    """Postgres-backed persistence with the same semantics as :class:`SQLiteStore`."""

    __slots__ = ("_clock", "_closed", "_dsn", "_max_size", "_min_size", "_pool")

    dialect = "postgres"

    def __init__(
        self,
        dsn: str,
        *,
        clock: Callable[[], float] = time.time,
        min_size: int = 1,
        max_size: int = 8,
    ) -> None:
        if not isinstance(dsn, str) or not dsn:
            raise ValueError(
                "dsn must be a non-empty connection string; "
                'fix: pass "postgresql://user:password@host:5432/branchpilot"'
            )
        if not callable(clock):
            raise TypeError("clock must be a zero-argument callable; fix: pass time.time")
        if min_size < 1 or max_size < min_size:
            raise ValueError(
                f"pool sizing must satisfy 1 <= min_size <= max_size, got {min_size} and "
                f"{max_size}; fix: pass min_size=1, max_size=8"
            )
        self._dsn = dsn
        self._clock = clock
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None
        self._closed = False

    async def _acquire_pool(self) -> Any:
        if self._closed:
            raise StoreError(
                "this store is closed; fix: open a new store with open_store(...) instead of "
                "reusing a closed one"
            )
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn, min_size=self._min_size, max_size=self._max_size
                )
            except (OSError, asyncpg.PostgresError) as exc:
                raise StoreError(
                    "cannot reach the configured Postgres server; fix: check the DSN host, port, "
                    "and credentials, or switch to the default SQLite backend with "
                    'open_store("sqlite:///branchpilot.db")'
                ) from exc
        return self._pool

    @staticmethod
    def _translate(exc: BaseException) -> BaseException:
        if isinstance(exc, asyncpg.UndefinedTableError):
            return StoreError(
                "the store schema is missing; "
                "fix: call `await store.migrate()` once at startup, it is idempotent"
            )
        return exc

    # -- lifecycle --------------------------------------------------------------------------

    async def migrate(self) -> int:
        migrations = migrations_for(self.dialect)
        latest = migrations[-1].version
        pool = await self._acquire_pool()
        async with pool.acquire() as connection:
            await connection.execute(_CREATE_SCHEMA_MIGRATIONS)
            rows = await connection.fetch(_SELECT_SCHEMA_MIGRATIONS)
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
                            f"migration {migration.version} ({record[0]}) was applied with "
                            "different SQL than this build ships; fix: reinstall the branchpilot "
                            "version that created this database, or recreate it from empty"
                        )
                    continue
                async with connection.transaction():
                    for statement in migration.statements:
                        await connection.execute(statement)
                    await connection.execute(
                        _INSERT_SCHEMA_MIGRATION,
                        migration.version,
                        migration.name,
                        migration.checksum,
                        self._clock(),
                    )
        return latest

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pool = self._pool
        self._pool = None
        if pool is not None:
            await pool.close()

    # -- ledger -----------------------------------------------------------------------------

    async def append_ledger(self, records: Sequence[Mapping[str, Any]]) -> None:
        rows = normalize_ledger_records(records)
        if not rows:
            return
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                await connection.executemany(_INSERT_LEDGER, [row.params for row in rows])
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc

    async def rollup_ledger(self, *, start: float, end: float) -> Mapping[str, Any]:
        window_start, window_end = rollup_window(start, end)
        totals = RollupAccumulator()
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                cursor = connection.cursor(
                    _SELECT_LEDGER_WINDOW, window_start, window_end, prefetch=_ROLLUP_CHUNK
                )
                async for row in cursor:
                    totals.add(row)
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc
        return totals.freeze(window_start, window_end)

    # -- cache metadata ---------------------------------------------------------------------

    async def get_cache_entry(self, key: str) -> Mapping[str, Any] | None:
        lookup = identifier(key, field="cache key", max_chars=512)
        now = finite_float(self._clock(), field="clock reading")
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection:
                row = await connection.fetchrow(_SELECT_CACHE_ENTRY, lookup)
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc
        if row is None or float(row["expires_s"]) <= now:
            return None
        return cache_mapping(row)

    async def put_cache_entry(self, key: str, entry: Mapping[str, Any], *, ttl_s: float) -> None:
        now = finite_float(self._clock(), field="clock reading")
        row = normalize_cache_entry(key, entry, ttl_s=ttl_s, now=now)
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                existing = await connection.fetchrow(_SELECT_CACHE_ENTRY, row.key)
                if existing is not None and existing["principal"] != row.principal:
                    raise StoreError(
                        f"cache key {row.key!r} already belongs to another principal; refusing to "
                        "overwrite it so one tenant cannot poison another's cache; "
                        "fix: include the principal in the cache key"
                    )
                await connection.execute(_UPSERT_CACHE_ENTRY, *row.params)
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc

    async def evict_expired(self, *, now: float) -> int:
        moment = finite_float(now, field="now")
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection:
                status = await connection.execute(_DELETE_EXPIRED_CACHE, moment)
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc
        return int(status.rsplit(" ", 1)[-1])

    # -- batch state ------------------------------------------------------------------------

    async def create_batch(self, batch: Mapping[str, Any]) -> str:
        now = finite_float(self._clock(), field="clock reading")
        supplied = batch.get("batch_id") if isinstance(batch, Mapping) else None
        batch_id = uuid.uuid4().hex if supplied is None else identifier(supplied, field="batch_id")
        row = normalize_batch(batch, batch_id=batch_id, now=now)
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                status = await connection.execute(_INSERT_BATCH, *row.params)
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc
        if status.rsplit(" ", 1)[-1] == "0":
            raise StoreError(
                f"batch {row.batch_id!r} already exists; fix: call update_batch to change an "
                "existing batch, or omit batch_id to have one generated"
            )
        return row.batch_id

    async def update_batch(self, batch_id: str, patch: Mapping[str, Any]) -> None:
        target = identifier(batch_id, field="batch_id")
        now = finite_float(self._clock(), field="clock reading")
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                existing = await connection.fetchrow(_SELECT_BATCH, target)
                if existing is None:
                    raise StoreError(
                        f"batch {target!r} does not exist; fix: create it with create_batch "
                        "before updating it"
                    )
                updated = apply_batch_patch(batch_row_of(existing), patch, now=now)
                await connection.execute(
                    _UPDATE_BATCH,
                    updated.upstream_batch_id,
                    updated.status,
                    updated.pending,
                    updated.attempts,
                    updated.updated_s,
                    updated.state,
                    updated.batch_id,
                )
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc

    async def pending_batches(self) -> Sequence[Mapping[str, Any]]:
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection:
                rows = await connection.fetch(_SELECT_PENDING_BATCHES)
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc
        return tuple(batch_mapping(row) for row in rows)

    # -- rate limits ------------------------------------------------------------------------

    async def consume_tokens(
        self, principal: str, *, tokens: float, rate: float, burst: float, now: float
    ) -> bool:
        target = identifier(principal, field="principal")
        capacity = finite_float(burst, field="burst")
        moment = finite_float(now, field="now")
        pool = await self._acquire_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                await connection.execute(_SEED_BUCKET, target, capacity, moment)
                row = await connection.fetchrow(_SELECT_BUCKET, target)
                stored = (float(row["tokens"]), float(row["updated_s"]))
                decision = bucket_decision(
                    stored, tokens=tokens, rate=rate, burst=burst, now=moment
                )
                await connection.execute(
                    _UPDATE_BUCKET, decision.tokens, decision.updated_s, target
                )
        except asyncpg.PostgresError as exc:
            raise self._translate(exc) from exc
        return decision.allowed
