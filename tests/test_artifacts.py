import os
from pathlib import Path

import pytest

from branchpilot.artifacts import (
    atomic_text_writer,
    atomic_write_bytes,
    atomic_write_text,
    paths_alias,
    sha256_file,
)


def test_atomic_text_writer_creates_parents_and_writes_utf8(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "artifact.txt"

    with atomic_text_writer(path) as handle:
        handle.write("BranchPilot — ready\n")

    assert path.read_bytes() == "BranchPilot — ready\n".encode()


def test_atomic_write_text_replaces_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("old", encoding="utf-8")

    atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "new"


def test_atomic_write_bytes_preserves_bytes(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.bin"
    data = b"\x00\xffBranchPilot\n"

    atomic_write_bytes(path, data)

    assert path.read_bytes() == data


def test_atomic_text_writer_preserves_old_content_and_cleans_temp_on_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("old content", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="injected failure"),
        atomic_text_writer(path) as handle,
    ):
        handle.write("partial new content")
        raise RuntimeError("injected failure")

    assert path.read_text(encoding="utf-8") == "old content"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_replaces_final_symlink_without_mutating_target(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("sentinel", encoding="utf-8")
    destination = tmp_path / "artifact.txt"
    destination.symlink_to(sentinel)

    atomic_write_text(destination, "artifact")

    assert not destination.is_symlink()
    assert destination.read_text(encoding="utf-8") == "artifact"
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def test_sha256_file_returns_stable_hex_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_paths_alias_matches_normalized_relative_and_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    absolute = tmp_path / "nested" / "artifact.txt"
    relative = Path("nested") / ".." / "nested" / "artifact.txt"

    assert paths_alias(relative, absolute)
    assert not absolute.exists()
    assert not paths_alias(relative, tmp_path / "other.txt")


def test_paths_alias_matches_hardlinks(tmp_path: Path) -> None:
    original = tmp_path / "artifact.txt"
    original.write_text("artifact", encoding="utf-8")
    hardlink = tmp_path / "artifact-hardlink.txt"
    os.link(original, hardlink)

    assert paths_alias(original, hardlink)
