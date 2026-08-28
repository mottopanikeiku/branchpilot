"""Persistence for the ledger, cache metadata, batch state, and rate-limit buckets.

The default backend is SQLite through the standard library, so a caller who configures nothing
gets durable single-node persistence with zero external services. Postgres is available behind
the ``postgres`` extra and is never imported by the default path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from branchpilot.store.base import Store, StoreError, money_text, money_value
from branchpilot.store.migrations import LATEST_VERSION
from branchpilot.store.sqlite import MEMORY_PATH, SQLiteStore

DEFAULT_STORE_URL = "sqlite:///branchpilot.db"
SQLITE_SCHEMES = ("sqlite",)
POSTGRES_SCHEMES = ("postgres", "postgresql")


def sqlite_path(url: str) -> str:
    """Extract the database path from a ``sqlite://`` URL."""

    parts = urlsplit(url)
    if parts.netloc:
        raise StoreError(
            f"{url!r} points at a host; SQLite is a local file; "
            'fix: use three slashes, sqlite:///branchpilot.db, or switch to a "postgresql://" '
            "URL with the 'postgres' extra"
        )
    if parts.query or parts.fragment:
        raise StoreError(
            f"{url!r} carries a query or fragment, which SQLite URLs do not support; "
            "fix: pass sqlite:///branchpilot.db"
        )
    # SQLAlchemy's convention: sqlite:///relative.db, sqlite:////absolute.db, sqlite:///:memory:.
    path = parts.path[1:] if parts.path.startswith("/") else parts.path
    if not path:
        raise StoreError(
            f"{url!r} names no database file; "
            "fix: pass sqlite:///branchpilot.db, or sqlite:///:memory: for an ephemeral store"
        )
    return MEMORY_PATH if path == MEMORY_PATH else path


def open_store(
    url: str | Path = DEFAULT_STORE_URL,
    *,
    clock: Callable[[], float] = time.time,
) -> Store:
    """Open the store named by ``url``.

    ``sqlite:///path`` (the default), a bare filesystem path, and ``postgresql://…`` are the
    supported forms. The returned store is not migrated yet: call ``await store.migrate()`` once
    at startup, which is idempotent and safe on every boot.
    """

    if isinstance(url, Path):
        return SQLiteStore(url, clock=clock)
    if not isinstance(url, str) or not url.strip():
        raise StoreError(
            "the store URL must be a non-empty string or Path; "
            f"fix: pass {DEFAULT_STORE_URL!r} for the default local database"
        )
    target = url.strip()
    scheme = urlsplit(target).scheme.lower()
    if scheme in SQLITE_SCHEMES:
        return SQLiteStore(sqlite_path(target), clock=clock)
    if scheme in POSTGRES_SCHEMES:
        from branchpilot.store.postgres import PostgresStore

        return PostgresStore(target, clock=clock)
    if not scheme or (len(scheme) == 1 and target[1:2] == ":"):
        # A bare path, including a Windows drive letter, means the default backend.
        return SQLiteStore(target, clock=clock)
    supported = ", ".join(f"{name}://" for name in (*SQLITE_SCHEMES, *POSTGRES_SCHEMES))
    raise StoreError(
        f"unsupported store URL scheme {scheme!r}; fix: use one of: {supported}, or a plain "
        "filesystem path for the default SQLite backend"
    )


__all__ = [
    "DEFAULT_STORE_URL",
    "LATEST_VERSION",
    "MEMORY_PATH",
    "SQLiteStore",
    "Store",
    "StoreError",
    "money_text",
    "money_value",
    "open_store",
    "sqlite_path",
]
