"""Postgres migration statements. Money columns are TEXT; never DOUBLE PRECISION or NUMERIC.

This module holds SQL text only. It imports nothing optional, so the core import path can check
dialect parity without ``asyncpg`` installed.
"""

from __future__ import annotations

_V1_LEDGER = """
CREATE TABLE IF NOT EXISTS ledger (
    seq BIGSERIAL PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE,
    ts DOUBLE PRECISION NOT NULL,
    principal TEXT NOT NULL,
    lever TEXT NOT NULL,
    model TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('serve', 'shadow')),
    prompt_tokens BIGINT NOT NULL CHECK (prompt_tokens >= 0),
    completion_tokens BIGINT NOT NULL CHECK (completion_tokens >= 0),
    cost_text TEXT NOT NULL,
    baseline_cost_text TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    attrs TEXT NOT NULL
)
"""

_V1_LEDGER_TS = "CREATE INDEX IF NOT EXISTS ledger_ts ON ledger (ts)"

_V1_CACHE = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    created_s DOUBLE PRECISION NOT NULL,
    expires_s DOUBLE PRECISION NOT NULL,
    payload TEXT NOT NULL
)
"""

_V1_CACHE_EXPIRES = """
CREATE INDEX IF NOT EXISTS cache_entries_expires_s ON cache_entries (expires_s)
"""

_V1_BATCHES = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    provider TEXT NOT NULL,
    upstream_batch_id TEXT,
    status TEXT NOT NULL,
    pending BOOLEAN NOT NULL,
    attempts BIGINT NOT NULL CHECK (attempts >= 0),
    created_s DOUBLE PRECISION NOT NULL,
    updated_s DOUBLE PRECISION NOT NULL,
    state TEXT NOT NULL
)
"""

_V1_BATCHES_PENDING = """
CREATE INDEX IF NOT EXISTS batches_pending ON batches (pending, created_s)
"""

_V1_BUCKETS = """
CREATE TABLE IF NOT EXISTS rate_buckets (
    principal TEXT PRIMARY KEY,
    tokens DOUBLE PRECISION NOT NULL,
    updated_s DOUBLE PRECISION NOT NULL
)
"""

POSTGRES_MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (
        1,
        "initial",
        (
            _V1_LEDGER,
            _V1_LEDGER_TS,
            _V1_CACHE,
            _V1_CACHE_EXPIRES,
            _V1_BATCHES,
            _V1_BATCHES_PENDING,
            _V1_BUCKETS,
        ),
    ),
)
