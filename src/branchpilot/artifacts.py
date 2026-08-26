from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Any, BinaryIO, TextIO, cast

__all__ = [
    "atomic_text_writer",
    "atomic_write_bytes",
    "atomic_write_text",
    "paths_alias",
    "sha256_file",
]

_PathLike = str | os.PathLike[str]


@contextmanager
def _atomic_file(
    path: _PathLike,
    mode: str,
    *,
    encoding: str | None = None,
) -> Iterator[IO[Any]]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    handle: IO[Any] | None = None
    committed = False
    try:
        if encoding is None:
            handle = os.fdopen(descriptor, mode)
        else:
            handle = os.fdopen(descriptor, mode, encoding=encoding, newline="")
        descriptor = -1
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        committed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if handle is not None and not handle.closed:
            handle.close()
        if not committed:
            with suppress(FileNotFoundError):
                temporary_path.unlink()


@contextmanager
def atomic_text_writer(path: _PathLike) -> Iterator[TextIO]:
    """Yield a UTF-8 writer whose contents replace *path* only on success."""
    with _atomic_file(path, "w", encoding="utf-8") as handle:
        yield cast(TextIO, handle)


def atomic_write_text(path: _PathLike, text: str) -> None:
    with atomic_text_writer(path) as handle:
        handle.write(text)


def atomic_write_bytes(path: _PathLike, data: bytes) -> None:
    with _atomic_file(path, "wb") as handle:
        cast(BinaryIO, handle).write(data)


def sha256_file(path: _PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_alias(left: _PathLike, right: _PathLike) -> bool:
    left_path = os.fspath(left)
    right_path = os.fspath(right)
    if os.path.normcase(os.path.abspath(left_path)) == os.path.normcase(
        os.path.abspath(right_path)
    ):
        return True
    try:
        return os.path.samefile(left_path, right_path)
    except OSError:
        return False
