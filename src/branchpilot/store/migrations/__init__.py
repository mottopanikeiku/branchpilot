"""Forward-only, integer-versioned schema migrations.

Migrations are Python tuples of SQL statements rather than ``.sql`` files on purpose:

* No statement splitter. A ``.sql`` file must be cut into statements before it can run inside one
  explicit transaction (``executescript`` commits first), and a hand-rolled splitter is a silent
  corruption risk the moment a literal contains a semicolon.
* No packaging risk. Statements ship as importable code, so a wheel that imports at all has its
  migrations, instead of failing at first startup on missing package data.
* Dialect parity is checked at import time: adding a table to SQLite and forgetting Postgres is a
  refusal here, not a production surprise later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from branchpilot.store.base import DIALECTS, StoreError
from branchpilot.store.migrations.postgres import POSTGRES_MIGRATIONS
from branchpilot.store.migrations.sqlite import SQLITE_MIGRATIONS


@dataclass(frozen=True, slots=True)
class Migration:
    """One forward migration, applied as a single transaction."""

    version: int
    name: str
    statements: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError(
                "migration version must be an integer; "
                "fix: number the migration with the next integer after the current latest"
            )
        if self.version < 1:
            raise ValueError(
                f"migration version must start at 1, got {self.version}; "
                "fix: renumber the migration tuple in branchpilot.store.migrations"
            )
        if not self.name or self.name.strip() != self.name:
            raise ValueError(
                f"migration {self.version} needs a non-empty trimmed name; "
                "fix: name it after what it changes, such as 'initial'"
            )
        if not self.statements:
            raise ValueError(
                f"migration {self.version} has no statements; "
                "fix: add the SQL statements, or drop the migration entry entirely"
            )
        for statement in self.statements:
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError(
                    f"migration {self.version} has an empty statement; "
                    "fix: remove the empty entry from the statement tuple"
                )
            if ";" in statement:
                raise ValueError(
                    f"migration {self.version} statement contains ';'; "
                    "fix: split it into one statement per tuple entry"
                )

    @property
    def checksum(self) -> str:
        """Digest of the exact statements, recorded so drift is detected, not assumed away."""

        digest = hashlib.sha256()
        for statement in self.statements:
            digest.update(statement.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()


def _validate(
    dialect: str, raw: Sequence[tuple[int, str, tuple[str, ...]]]
) -> tuple[Migration, ...]:
    migrations = tuple(
        Migration(version=version, name=name, statements=statements)
        for version, name, statements in raw
    )
    for offset, migration in enumerate(migrations, start=1):
        if migration.version != offset:
            raise StoreError(
                f"{dialect} migrations must be numbered 1..n without gaps; found version "
                f"{migration.version} at position {offset}; "
                "fix: renumber the migration tuple in branchpilot.store.migrations"
            )
    return migrations


_REGISTRY: Mapping[str, tuple[Migration, ...]] = MappingProxyType(
    {
        "sqlite": _validate("sqlite", SQLITE_MIGRATIONS),
        "postgres": _validate("postgres", POSTGRES_MIGRATIONS),
    }
)

if set(_REGISTRY) != set(DIALECTS):  # pragma: no cover - guarded by test_store.py
    raise StoreError(
        "every supported dialect needs a migration list; "
        f"fix: add migrations for: {', '.join(sorted(set(DIALECTS) - set(_REGISTRY)))}"
    )

_SHAPES = {
    dialect: tuple((item.version, item.name) for item in migrations)
    for dialect, migrations in _REGISTRY.items()
}
if len(set(_SHAPES.values())) != 1:
    raise StoreError(
        "sqlite and postgres migrations disagree on versions or names; fix: add the matching "
        "migration to both dialects in branchpilot.store.migrations"
    )

LATEST_VERSION: int = _REGISTRY["sqlite"][-1].version


def migrations_for(dialect: str) -> tuple[Migration, ...]:
    """Return the ordered migrations for a dialect."""

    try:
        return _REGISTRY[dialect]
    except KeyError:
        expected = ", ".join(sorted(_REGISTRY))
        raise StoreError(
            f"unknown store dialect {dialect!r}; fix: use one of: {expected}"
        ) from None


__all__ = ["LATEST_VERSION", "Migration", "migrations_for"]
