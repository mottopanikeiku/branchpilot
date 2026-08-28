from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import importlib.util
import os
import re
import sqlite3
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

import pytest

from branchpilot.store import (
    DEFAULT_STORE_URL,
    LATEST_VERSION,
    MEMORY_PATH,
    SQLiteStore,
    Store,
    StoreError,
    open_store,
)
from branchpilot.store.base import LEDGER_MODES, TERMINAL_BATCH_STATUSES
from branchpilot.store.migrations import Migration, migrations_for

_T = TypeVar("_T")

STORE_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "branchpilot" / "store"
SQL_KEYWORDS = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\b")
FSTRING = re.compile(r"""f["']""")
EXECUTE_PERCENT = re.compile(r"\.execute(?:many)?\([^)]*%")


def run(main: Awaitable[_T]) -> _T:
    return asyncio.run(main)  # type: ignore[arg-type]


class Clock:
    """A deterministic clock; no test depends on wall time."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def record(index: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_id": f"rec-{index}",
        "ts": 1_000.0 + index,
        "principal": "tenant-a",
        "lever": "passthrough",
        "model": "gpt-4o-mini",
        "mode": "serve",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost": Decimal("0.0000123456"),
        "baseline_cost": Decimal("0.0000246912"),
        "request_hash": hashlib.sha256(str(index).encode()).hexdigest(),
    }
    payload.update(overrides)
    return payload


async def migrated(path: str | Path, clock: Clock | None = None) -> SQLiteStore:
    store = SQLiteStore(path, clock=clock or Clock())
    await store.migrate()
    return store


def with_store(
    tmp_path: Path,
    scenario: Callable[[SQLiteStore], Awaitable[_T]],
    *,
    clock: Clock | None = None,
) -> _T:
    async def main() -> _T:
        store = await migrated(tmp_path / "store.db", clock)
        try:
            return await scenario(store)
        finally:
            await store.close()

    return run(main())


# -- lifecycle and migrations ---------------------------------------------------------------


def test_migrate_from_empty_is_idempotent_and_reports_the_version(tmp_path: Path) -> None:
    async def scenario() -> tuple[int, int, list[tuple[Any, ...]]]:
        store = SQLiteStore(tmp_path / "nested" / "store.db")
        try:
            first = await store.migrate()
            second = await store.migrate()
        finally:
            await store.close()
        with sqlite3.connect(tmp_path / "nested" / "store.db") as raw:
            rows = raw.execute(
                "SELECT version, name FROM schema_migrations ORDER BY version"
            ).fetchall()
        return first, second, rows

    first, second, rows = run(scenario())

    assert first == second == LATEST_VERSION
    assert rows == [(item.version, item.name) for item in migrations_for("sqlite")]


def test_default_path_needs_no_external_service_and_hardens_the_connection(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, Any]:
        store = await migrated(tmp_path / "store.db")
        try:
            # The connection is private on purpose; the pragmas it carries are a contract.
            connection = store._connection
            assert connection is not None
            return {
                "journal": connection.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
                "foreign_keys": connection.execute("PRAGMA foreign_keys").fetchone()[0],
                "busy_timeout": connection.execute("PRAGMA busy_timeout").fetchone()[0],
            }
        finally:
            await store.close()

    pragmas = run(scenario())

    assert str(pragmas["journal"]).lower() == "wal"
    assert pragmas["synchronous"] == 1
    assert pragmas["foreign_keys"] == 1
    assert pragmas["busy_timeout"] >= 1_000


def test_operations_before_migrate_name_the_fix(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SQLiteStore(tmp_path / "store.db")
        try:
            with pytest.raises(StoreError, match=r"fix: call `await store\.migrate\(\)`"):
                await store.append_ledger([record(0)])
        finally:
            await store.close()

    run(scenario())


def test_closed_store_refuses_further_work(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = await migrated(tmp_path / "store.db")
        await store.close()
        await store.close()
        with pytest.raises(StoreError, match="fix: open a new store"):
            await store.pending_batches()

    run(scenario())


def test_tampered_migration_checksum_is_detected(tmp_path: Path) -> None:
    database = tmp_path / "store.db"

    async def scenario() -> None:
        store = await migrated(database)
        await store.close()
        with sqlite3.connect(database) as raw:
            raw.execute("UPDATE schema_migrations SET checksum = ? WHERE version = 1", ("00",))
        reopened = SQLiteStore(database)
        try:
            with pytest.raises(StoreError, match="different SQL than this build ships"):
                await reopened.migrate()
        finally:
            await reopened.close()

    run(scenario())


def test_future_schema_version_refuses_to_downgrade(tmp_path: Path) -> None:
    database = tmp_path / "store.db"

    async def scenario() -> None:
        store = await migrated(database)
        await store.close()
        with sqlite3.connect(database) as raw:
            raw.execute(
                "INSERT INTO schema_migrations (version, name, checksum, applied_at_s) "
                "VALUES (?, ?, ?, ?)",
                (LATEST_VERSION + 1, "future", "00", 0.0),
            )
        reopened = SQLiteStore(database)
        try:
            with pytest.raises(StoreError, match="fix: upgrade branchpilot"):
                await reopened.migrate()
        finally:
            await reopened.close()

    run(scenario())


def test_migration_registry_is_contiguous_and_dialect_parity_holds() -> None:
    sqlite_migrations = migrations_for("sqlite")
    postgres_migrations = migrations_for("postgres")

    assert [item.version for item in sqlite_migrations] == list(
        range(1, len(sqlite_migrations) + 1)
    )
    assert [(item.version, item.name) for item in sqlite_migrations] == [
        (item.version, item.name) for item in postgres_migrations
    ]
    for dialect in (sqlite_migrations, postgres_migrations):
        statements = "\n".join(statement for item in dialect for statement in item.statements)
        for mode in LEDGER_MODES:
            assert f"'{mode}'" in statements
        assert "cost_text TEXT NOT NULL" in statements
        assert "baseline_cost_text TEXT NOT NULL" in statements
        money_columns = [
            line
            for line in statements.splitlines()
            if "cost" in line and ("REAL" in line or "DOUBLE" in line or "NUMERIC" in line)
        ]
        assert money_columns == []


def test_unknown_dialect_and_malformed_migrations_are_refused() -> None:
    with pytest.raises(StoreError, match="fix: use one of: postgres, sqlite"):
        migrations_for("mysql")
    with pytest.raises(ValueError, match="fix: split it"):
        Migration(version=1, name="bad", statements=("SELECT 1; SELECT 2",))
    with pytest.raises(ValueError, match="no statements"):
        Migration(version=1, name="bad", statements=())


def test_store_implements_the_protocol(tmp_path: Path) -> None:
    assert isinstance(SQLiteStore(tmp_path / "store.db"), Store)


def test_open_store_resolves_urls(tmp_path: Path) -> None:
    assert DEFAULT_STORE_URL == "sqlite:///branchpilot.db"
    assert open_store(f"sqlite:///{tmp_path / 'a.db'}").path == str(tmp_path / "a.db")
    assert open_store("sqlite:///:memory:").path == MEMORY_PATH
    assert open_store(tmp_path / "b.db").path == str(tmp_path / "b.db")
    assert open_store("relative/c.db").path == "relative/c.db"
    with pytest.raises(StoreError, match="fix: use one of: sqlite://"):
        open_store("redis://localhost:6379/0")
    with pytest.raises(StoreError, match="fix: use three slashes"):
        open_store("sqlite://remote-host/branchpilot.db")
    with pytest.raises(StoreError, match="fix: pass"):
        open_store("   ")


# -- ledger ---------------------------------------------------------------------------------


def test_money_round_trips_through_text_columns_exactly(tmp_path: Path) -> None:
    exact = Decimal("0.0000123456")

    async def scenario(store: SQLiteStore) -> Any:
        await store.append_ledger(
            [record(0, cost=exact, baseline_cost=exact, attrs={"unit_price": exact})]
        )
        return await store.rollup_ledger(start=0.0, end=2_000.0)

    rollup = with_store(tmp_path, scenario)

    assert isinstance(rollup["cost"], Decimal)
    assert rollup["cost"] == exact
    assert rollup["baseline_cost"] == exact
    assert rollup["savings"] == Decimal(0)
    with sqlite3.connect(tmp_path / "store.db") as raw:
        stored = raw.execute("SELECT cost_text, attrs FROM ledger").fetchone()
    assert stored[0] == "0.0000123456"
    assert '{"$decimal":"0.0000123456"}' in stored[1]


def test_rollup_is_half_open_and_grouped(tmp_path: Path) -> None:
    async def scenario(store: SQLiteStore) -> Any:
        await store.append_ledger(
            [
                record(0, ts=1_000.0, lever="prompt_cache", cost=Decimal("0.10")),
                record(1, ts=1_500.0, lever="prompt_cache", cost=Decimal("0.20")),
                record(2, ts=2_000.0, lever="batch", mode="shadow", cost=Decimal("0.40")),
            ]
        )
        return (
            await store.rollup_ledger(start=1_000.0, end=2_000.0),
            await store.rollup_ledger(start=1_000.0, end=2_000.1),
        )

    inside, including_edge = with_store(tmp_path, scenario)

    assert inside["records"] == 2
    assert inside["cost"] == Decimal("0.30")
    assert inside["prompt_tokens"] == 20
    assert set(inside["by_lever"]) == {"prompt_cache"}
    assert inside["by_lever"]["prompt_cache"]["cost"] == Decimal("0.30")
    assert including_edge["records"] == 3
    assert set(including_edge["by_mode"]) == {"serve", "shadow"}
    assert including_edge["by_mode"]["shadow"]["cost"] == Decimal("0.40")
    with pytest.raises(ValueError, match="fix: pass start <= end"):
        with_store(tmp_path, lambda store: store.rollup_ledger(start=5.0, end=1.0))


def test_append_ledger_is_idempotent_per_record_id(tmp_path: Path) -> None:
    async def scenario(store: SQLiteStore) -> Any:
        await store.append_ledger([record(0)])
        await store.append_ledger([record(0), record(1)])
        await store.append_ledger([])
        return await store.rollup_ledger(start=0.0, end=9_999.0)

    rollup = with_store(tmp_path, scenario)

    assert rollup["records"] == 2


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"cost": 0.01}, TypeError, "float is rejected in cost paths"),
        ({"cost": "0.01"}, TypeError, "fix: wrap the amount"),
        ({"cost": Decimal("-0.01")}, ValueError, "must not be negative"),
        ({"cost": Decimal("NaN")}, ValueError, "must be a finite Decimal"),
        ({"mode": "serving"}, ValueError, "fix: record shadow traffic"),
        ({"request_hash": "abc"}, ValueError, "sha256 hex digest"),
        ({"request_hash": "A" * 64}, ValueError, "sha256 hex digest"),
        ({"prompt_tokens": -1}, ValueError, "must not be negative"),
        ({"prompt_tokens": 1.5}, TypeError, "must be an integer"),
        ({"ts": float("inf")}, ValueError, "must be finite"),
        ({"principal": ""}, ValueError, "non-empty"),
        ({"attrs": {"prompt": "hello"}}, ValueError, "fix: hash the text"),
        ({"attrs": {"nested": [{"messages": []}]}}, ValueError, "fix: hash the text"),
        ({"attrs": {"latency_ms": float("nan")}}, ValueError, "must be a finite number"),
        ({"attrs": {"path": Path(".")}}, TypeError, "must be JSON-encodable"),
    ],
)
def test_invalid_ledger_records_are_refused_with_a_fix(
    tmp_path: Path, overrides: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message) as raised:
        with_store(tmp_path, lambda store: store.append_ledger([record(0, **overrides)]))
    assert "fix:" in str(raised.value)


def test_unknown_and_missing_ledger_fields_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        with_store(tmp_path, lambda store: store.append_ledger([record(0, savings=Decimal(1))]))
    incomplete = record(0)
    del incomplete["model"]
    with pytest.raises(ValueError, match="missing required field"):
        with_store(tmp_path, lambda store: store.append_ledger([incomplete]))
    with pytest.raises(ValueError, match="repeats record_id"):
        with_store(tmp_path, lambda store: store.append_ledger([record(0), record(0)]))
    with pytest.raises(TypeError, match="fix: pass a list of ledger records"):
        with_store(tmp_path, lambda store: store.append_ledger(record(0)))


def test_thirty_two_concurrent_writers_lose_no_ledger_record(tmp_path: Path) -> None:
    writers = 32
    calls_per_writer = 10
    per_call = 10

    async def scenario() -> Any:
        store = await migrated(tmp_path / "store.db")

        async def writer(worker: int) -> None:
            for call in range(calls_per_writer):
                await store.append_ledger(
                    [
                        record(
                            0,
                            record_id=f"rec-{worker}-{call}-{item}",
                            request_hash=hashlib.sha256(
                                f"{worker}-{call}-{item}".encode()
                            ).hexdigest(),
                        )
                        for item in range(per_call)
                    ]
                )

        try:
            await asyncio.wait_for(
                asyncio.gather(*(writer(worker) for worker in range(writers))),
                timeout=120,
            )
            rollup = await store.rollup_ledger(start=0.0, end=9_999.0)
        finally:
            await store.close()
        with sqlite3.connect(tmp_path / "store.db") as raw:
            distinct = raw.execute("SELECT COUNT(DISTINCT record_id) FROM ledger").fetchone()[0]
        return rollup, distinct

    rollup, distinct = run(scenario())
    expected = writers * calls_per_writer * per_call

    assert rollup["records"] == expected
    assert distinct == expected
    assert rollup["cost"] == Decimal("0.0000123456") * expected


# -- cache metadata -------------------------------------------------------------------------


def test_cache_entry_round_trip_honours_the_required_ttl(tmp_path: Path) -> None:
    clock = Clock()

    async def scenario(store: SQLiteStore) -> Any:
        await store.put_cache_entry(
            "tenant-a:sha256:abc",
            {"principal": "tenant-a", "payload": {"tokens": 12, "unit_price": Decimal("0.002")}},
            ttl_s=60.0,
        )
        live = await store.get_cache_entry("tenant-a:sha256:abc")
        clock.advance(59.0)
        still_live = await store.get_cache_entry("tenant-a:sha256:abc")
        clock.advance(2.0)
        expired = await store.get_cache_entry("tenant-a:sha256:abc")
        evicted = await store.evict_expired(now=clock.now)
        return live, still_live, expired, evicted, await store.evict_expired(now=clock.now)

    live, still_live, expired, evicted, again = with_store(tmp_path, scenario, clock=clock)

    assert live is not None and still_live is not None
    assert live["principal"] == "tenant-a"
    assert live["payload"]["unit_price"] == Decimal("0.002")
    assert isinstance(live["payload"]["unit_price"], Decimal)
    assert live["expires_s"] == pytest.approx(live["created_s"] + 60.0)
    assert expired is None
    assert evicted == 1
    assert again == 0


def test_cache_put_overwrites_only_within_the_same_principal(tmp_path: Path) -> None:
    async def scenario(store: SQLiteStore) -> Any:
        await store.put_cache_entry(
            "shared-key", {"principal": "tenant-a", "payload": {"v": 1}}, ttl_s=30.0
        )
        await store.put_cache_entry(
            "shared-key", {"principal": "tenant-a", "payload": {"v": 2}}, ttl_s=30.0
        )
        updated = await store.get_cache_entry("shared-key")
        with pytest.raises(StoreError, match="fix: include the principal in the cache key"):
            await store.put_cache_entry(
                "shared-key", {"principal": "tenant-b", "payload": {"v": 3}}, ttl_s=30.0
            )
        return updated, await store.get_cache_entry("shared-key")

    updated, after_refusal = with_store(tmp_path, scenario)

    assert updated is not None and updated["payload"] == {"v": 2}
    assert after_refusal is not None and after_refusal["principal"] == "tenant-a"


@pytest.mark.parametrize(
    ("kwargs", "entry", "error", "message"),
    [
        ({"ttl_s": 0.0}, {"principal": "t"}, ValueError, "fix: set an explicit per-route TTL"),
        ({"ttl_s": -1.0}, {"principal": "t"}, ValueError, "fix: set an explicit per-route TTL"),
        (
            {"ttl_s": float("inf")},
            {"principal": "t"},
            ValueError,
            "ttl_s must be finite",
        ),
        ({"ttl_s": None}, {"principal": "t"}, TypeError, "must be a real number"),
        ({"ttl_s": 5.0}, {"payload": {}}, ValueError, "fix: scope every cache entry"),
        (
            {"ttl_s": 5.0},
            {"principal": "t", "payload": {"completion": "hi"}},
            ValueError,
            "fix: hash the text",
        ),
        (
            {"ttl_s": 5.0},
            {"principal": "t", "response": {}},
            ValueError,
            "unsupported key",
        ),
    ],
)
def test_cache_entries_without_a_safe_ttl_or_scope_are_refused(
    tmp_path: Path,
    kwargs: dict[str, Any],
    entry: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message) as raised:
        with_store(tmp_path, lambda store: store.put_cache_entry("k", entry, **kwargs))
    assert "fix:" in str(raised.value)


# -- batch state ----------------------------------------------------------------------------


def test_batch_lifecycle_survives_and_lists_only_unsettled_work(tmp_path: Path) -> None:
    clock = Clock()

    async def scenario(store: SQLiteStore) -> Any:
        generated = await store.create_batch({"principal": "tenant-a", "provider": "openai"})
        clock.advance(1.0)
        named = await store.create_batch(
            {
                "batch_id": "batch-2",
                "principal": "tenant-a",
                "provider": "anthropic",
                "state": {"request_hashes": 3, "unit_price": Decimal("0.5")},
            }
        )
        clock.advance(1.0)
        await store.update_batch(
            generated, {"status": "submitted", "upstream_batch_id": "up-1", "attempts": 1}
        )
        pending = await store.pending_batches()
        await store.update_batch("batch-2", {"status": "completed"})
        return generated, named, pending, await store.pending_batches()

    generated, named, pending, settled = with_store(tmp_path, scenario, clock=clock)

    assert named == "batch-2"
    assert len(generated) == 32
    assert [item["batch_id"] for item in pending] == [generated, "batch-2"]
    assert pending[0]["status"] == "submitted"
    assert pending[0]["upstream_batch_id"] == "up-1"
    assert pending[0]["attempts"] == 1
    assert pending[0]["updated_s"] > pending[0]["created_s"]
    assert pending[1]["state"]["unit_price"] == Decimal("0.5")
    assert [item["batch_id"] for item in settled] == [generated]
    assert {"completed", "failed", "cancelled"} == TERMINAL_BATCH_STATUSES


def test_batch_errors_name_the_fix(tmp_path: Path) -> None:
    async def scenario(store: SQLiteStore) -> None:
        await store.create_batch({"batch_id": "b1", "principal": "t", "provider": "openai"})
        with pytest.raises(StoreError, match="fix: call update_batch"):
            await store.create_batch({"batch_id": "b1", "principal": "t", "provider": "openai"})
        with pytest.raises(StoreError, match="fix: create it with create_batch"):
            await store.update_batch("missing", {"status": "failed"})
        with pytest.raises(ValueError, match="unsupported key"):
            await store.update_batch("b1", {"provider": "anthropic"})
        with pytest.raises(ValueError, match="fix: pass at least one of"):
            await store.update_batch("b1", {})
        with pytest.raises(ValueError, match="fix: use 'queued' for new work"):
            await store.update_batch("b1", {"status": "done"})
        with pytest.raises(ValueError, match="fix: supply principal and provider"):
            await store.create_batch({"provider": "openai"})
        with pytest.raises(ValueError, match="fix: hash the text"):
            await store.create_batch(
                {"principal": "t", "provider": "openai", "state": {"messages": []}}
            )

    with_store(tmp_path, scenario)


# -- rate limits ----------------------------------------------------------------------------


def test_consume_tokens_is_a_monotonic_token_bucket(tmp_path: Path) -> None:
    async def scenario(store: SQLiteStore) -> list[bool]:
        results = [
            await store.consume_tokens("tenant-a", tokens=4.0, rate=2.0, burst=10.0, now=100.0),
            await store.consume_tokens("tenant-a", tokens=6.0, rate=2.0, burst=10.0, now=100.0),
            await store.consume_tokens("tenant-a", tokens=1.0, rate=2.0, burst=10.0, now=100.0),
            # A clock that goes backwards must never mint tokens.
            await store.consume_tokens("tenant-a", tokens=1.0, rate=2.0, burst=10.0, now=1.0),
            await store.consume_tokens("tenant-a", tokens=1.0, rate=2.0, burst=10.0, now=100.5),
            await store.consume_tokens("tenant-a", tokens=2.0, rate=2.0, burst=10.0, now=101.0),
            # A separate principal has its own bucket.
            await store.consume_tokens("tenant-b", tokens=10.0, rate=2.0, burst=10.0, now=101.0),
            # rate=0 never refills.
            await store.consume_tokens("tenant-c", tokens=3.0, rate=0.0, burst=3.0, now=0.0),
            await store.consume_tokens("tenant-c", tokens=3.0, rate=0.0, burst=3.0, now=10_000.0),
        ]
        return results

    assert with_store(tmp_path, scenario) == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tokens": 11.0, "rate": 1.0, "burst": 10.0}, "fix: raise burst"),
        ({"tokens": 0.0, "rate": 1.0, "burst": 10.0}, "fix: charge at least one token"),
        ({"tokens": 1.0, "rate": -1.0, "burst": 10.0}, "rate must be at least"),
        ({"tokens": 1.0, "rate": 1.0, "burst": 0.0}, "fix: set burst"),
        ({"tokens": 1.0, "rate": 1.0, "burst": float("nan")}, "burst must be finite"),
    ],
)
def test_unsatisfiable_rate_limits_are_refused(
    tmp_path: Path, kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as raised:
        with_store(
            tmp_path,
            lambda store: store.consume_tokens("tenant-a", now=0.0, **kwargs),
        )
    assert "fix:" in str(raised.value)


# -- every raised message names a fix --------------------------------------------------------


def _literal_message(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = [_literal_message(value) or "" for value in node.values]
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_message(node.left)
        right = _literal_message(node.right)
        return None if left is None or right is None else left + right
    return None


def test_every_raised_message_in_the_store_names_a_fix() -> None:
    offenders: list[str] = []
    checked = 0
    for path in sorted(STORE_SOURCE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            if not node.exc.args:
                continue
            message = _literal_message(node.exc.args[0])
            if message is None:
                continue
            checked += 1
            if "fix:" not in message:
                offenders.append(f"{path.name}:{node.lineno}: {message[:60]}")

    assert checked >= 30
    assert offenders == []


# -- no string-formatted SQL ----------------------------------------------------------------


def _formatted_sql_offences(text: str) -> list[str]:
    offences = []
    for number, line in enumerate(text.splitlines(), start=1):
        formatted = SQL_KEYWORDS.search(line) and (FSTRING.search(line) or ".format(" in line)
        if formatted or EXECUTE_PERCENT.search(line):
            offences.append(f"{number}: {line.strip()}")
    return offences


def test_store_sources_contain_no_string_formatted_sql() -> None:
    sources = sorted(STORE_SOURCE_DIR.rglob("*.py")) + sorted(STORE_SOURCE_DIR.rglob("*.sql"))

    assert sources, f"no store sources found under {STORE_SOURCE_DIR}"
    offences = {
        str(path.relative_to(STORE_SOURCE_DIR)): _formatted_sql_offences(
            path.read_text(encoding="utf-8")
        )
        for path in sources
    }
    assert {path: hits for path, hits in offences.items() if hits} == {}


def test_the_formatted_sql_scanner_actually_catches_offences() -> None:
    assert _formatted_sql_offences('cur.execute(f"SELECT * FROM t WHERE k = {key}")')
    assert _formatted_sql_offences("cur.execute('DELETE FROM t WHERE k = %s' % key)")
    assert _formatted_sql_offences('sql = "INSERT INTO t VALUES ({})".format(value)')
    assert not _formatted_sql_offences('cur.execute("SELECT * FROM t WHERE k = ?", (key,))')


# -- optional Postgres backend --------------------------------------------------------------

_HAS_ASYNCPG = importlib.util.find_spec("asyncpg") is not None
_POSTGRES_DSN = os.environ.get("BRANCHPILOT_TEST_POSTGRES_DSN", "")


def test_postgres_module_without_the_extra_names_the_extra() -> None:
    if _HAS_ASYNCPG:
        pytest.skip("asyncpg is installed; the missing-extra path cannot be exercised here")
    with pytest.raises(ImportError, match=r"fix: install branchpilot\[postgres\]"):
        importlib.import_module("branchpilot.store.postgres")
    with pytest.raises(ImportError, match=r"fix: install branchpilot\[postgres\]"):
        open_store("postgresql://user@localhost:5432/branchpilot")


def test_core_import_path_never_imports_asyncpg() -> None:
    sources = sorted(STORE_SOURCE_DIR.rglob("*.py"))
    importers = [
        str(path.relative_to(STORE_SOURCE_DIR))
        for path in sources
        if path.name != "postgres.py" and "import asyncpg" in path.read_text(encoding="utf-8")
    ]

    assert importers == []
    assert "asyncpg" not in (STORE_SOURCE_DIR / "__init__.py").read_text(encoding="utf-8")


@pytest.mark.skipif(
    not (_HAS_ASYNCPG and _POSTGRES_DSN),
    reason="needs the 'postgres' extra and BRANCHPILOT_TEST_POSTGRES_DSN pointing at a server",
)
def test_postgres_backend_matches_the_sqlite_semantics() -> None:
    from branchpilot.store.postgres import PostgresStore

    async def scenario() -> Any:
        store = PostgresStore(_POSTGRES_DSN, clock=Clock())
        try:
            version = await store.migrate()
            assert version == await store.migrate() == LATEST_VERSION
            await store.append_ledger([record(0, record_id="pg-rec-0")])
            rollup = await store.rollup_ledger(start=0.0, end=9_999.0)
            allowed = await store.consume_tokens(
                "pg-tenant", tokens=1.0, rate=1.0, burst=1.0, now=0.0
            )
            denied = await store.consume_tokens(
                "pg-tenant", tokens=1.0, rate=1.0, burst=1.0, now=0.0
            )
        finally:
            await store.close()
        return rollup, allowed, denied

    rollup, allowed, denied = run(scenario())

    assert rollup["cost"] == Decimal("0.0000123456")
    assert isinstance(rollup["cost"], Decimal)
    assert allowed is True
    assert denied is False
