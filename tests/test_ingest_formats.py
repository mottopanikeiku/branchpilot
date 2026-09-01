from __future__ import annotations

import hashlib
import json
import tracemalloc
from collections.abc import Iterator
from pathlib import Path

import pytest

from branchpilot.ingest import (
    EMPTY_TEXT_HASH,
    FORMAT_IDS,
    MAX_RECORD_BYTES,
    SKIP_REASONS,
    AmbiguousFormatError,
    EmptySourceError,
    IngestError,
    IngestReport,
    MalformedRecordError,
    MappingError,
    RequestRecord,
    SourceNotFoundError,
    UnknownFormatError,
    detect,
    read_helicone_export,
    read_litellm_jsonl,
    read_openai_jsonl,
    read_openrouter_export,
    read_requests,
    read_requests_report,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"

# --------------------------------------------------------------------------------------
# the canonical dataset every format fixture encodes
# --------------------------------------------------------------------------------------

S_A = "You are the pangolin escalation assistant. Reply with a single short sentence."
S_B = "You are the toboggan summarizer. Produce three bullet points."
SYSTEMS = [S_A, S_A, S_A, None, S_B, S_B]
USERS = [
    "Sort the kumquat shipment manifest for warehouse nine.",
    "Where is the zeppelin invoice from last quarter?",
    "Draft a wristwatch refund note for order 88213.",
    "Ping the quokka uptime page and confirm nine nines.",
    "Summarize the marmoset quarterly memo.",
    "Summarize the narwhal onboarding packet.",
]
COMPLETIONS = [
    "Manifest sorted; kumquat crates are queued first.",
    "The zeppelin invoice is filed under vendor ledger twelve.",
    "Refund note drafted for the wristwatch order.",
    "Quokka uptime page confirms nine nines.",
    "Marmoset memo: three bullets on spend, staffing, and queue wait.",
    "Narwhal packet: three bullets on access, tooling, and review.",
]
PROMPT_TOKENS = [1200, 1180, 1520, 240, 2200, 2150]
CACHED_TOKENS = [1024, 1024, 1024, None, 2048, 2048]
COMPLETION_TOKENS = [96, 64, 220, 32, 310, 288]
LATENCY_MS = [812.5, 640.0, 1503.25, 210.0, 2410.0, 2280.5]
GROUPS = ["triage", "triage", "triage", None, "digest", "digest"]
TIMESTAMPS = [1775030400.0 + i * 90 for i in range(6)]

# Keys whose string values are prompt or completion text in some fixture format.
TEXT_KEYS = frozenset({"completion", "content", "input", "prompt", "reply", "system", "text"})


# --------------------------------------------------------------------------------------
# an independent implementation of the documented hash framing
# --------------------------------------------------------------------------------------


def _frame(hasher, value: str) -> None:
    payload = value.encode("utf-8")
    hasher.update(str(len(payload)).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(payload)


def digest_messages(messages: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """``sha256`` over ``frame(role) frame(part_count) (frame(kind) frame(text))*``."""
    hasher = hashlib.sha256()
    for role, parts in messages:
        _frame(hasher, role)
        _frame(hasher, str(len(parts)))
        for kind, text in parts:
            _frame(hasher, kind)
            _frame(hasher, text)
    return hasher.digest()[:16].hex()


def expected_messages_hash(i: int) -> str:
    messages: list[tuple[str, list[tuple[str, str]]]] = []
    if SYSTEMS[i] is not None:
        messages.append(("system", [("text", SYSTEMS[i])]))
    messages.append(("user", [("text", USERS[i])]))
    return digest_messages(messages)


def expected_system_prefix_hash(i: int) -> str:
    if SYSTEMS[i] is None:
        return EMPTY_TEXT_HASH
    return digest_messages([("system", [("text", SYSTEMS[i])])])


def expected_system_prefix_chars(i: int) -> int:
    system = SYSTEMS[i]
    return 0 if system is None else len(system)


# --------------------------------------------------------------------------------------
# per-format expectations
# --------------------------------------------------------------------------------------

EXPECTATIONS = {
    "openai-jsonl": {
        "file": "openai.jsonl",
        "ids": [f"chatcmpl-a{i}" for i in range(6)],
        "model": "gpt-4o-mini",
        "provider": "openai",
        "statuses": ["ok"] * 6,
        "groups": GROUPS,
    },
    "anthropic-jsonl": {
        "file": "anthropic.jsonl",
        "ids": [f"msg_01a{i}" for i in range(6)],
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "statuses": ["ok"] * 6,
        "groups": GROUPS,
    },
    "litellm-jsonl": {
        "file": "litellm.jsonl",
        "ids": [f"req-b{i}" for i in range(6)],
        "model": "gpt-4o",
        "provider": "openai",
        "statuses": ["ok", "ok", "ok", "ok", "ok", "error"],
        "groups": GROUPS,
    },
    "helicone-export": {
        "file": "helicone.jsonl",
        "ids": [f"hel-c{i}" for i in range(6)],
        "model": "gpt-4o-mini",
        "provider": "openai",
        "statuses": ["ok"] * 6,
        "groups": GROUPS,
    },
    "openrouter-export": {
        "file": "openrouter.jsonl",
        "ids": [f"gen-d{i}" for i in range(6)],
        "model": "meta-llama/llama-3.3-70b-instruct",
        "provider": "deepinfra",
        "statuses": ["ok", "ok", "ok", "ok", "ok", "cancelled"],
        "groups": ["4101", "4101", "4101", None, "4102", "4102"],
    },
    "generic-jsonl": {
        "file": "generic.jsonl",
        "ids": [f"tr-e{i}" for i in range(6)],
        "model": "gpt-4o-mini",
        "provider": "openai",
        "statuses": ["ok"] * 6,
        "groups": GROUPS,
    },
}

GENERIC_MAPPING = {
    "id": "trace_id",
    "timestamp": "ts",
    "model": "llm.model",
    "provider": {"const": "openai"},
    "messages": "turns",
    "message_role_key": "speaker",
    "message_content_key": "text",
    "completion": "reply",
    "prompt_tokens": "counters.in",
    "cached_prompt_tokens": "counters.in_cached",
    "completion_tokens": "counters.out",
    "latency_ms": "took_ms",
    "status": "outcome",
    "group_key": "tenant",
}


def open_fixture(format_id: str):
    spec = EXPECTATIONS[format_id]
    mapping = GENERIC_MAPPING if format_id == "generic-jsonl" else None
    return read_requests(FIXTURES / spec["file"], format=format_id, mapping=mapping)


def read_fixture(format_id: str) -> tuple[list[RequestRecord], IngestReport]:
    stream = open_fixture(format_id)
    records = list(stream)
    return records, stream.report()


# --------------------------------------------------------------------------------------
# field-by-field mapping
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("format_id", sorted(EXPECTATIONS))
def test_fixture_maps_to_expected_records(format_id: str) -> None:
    spec = EXPECTATIONS[format_id]
    records, report = read_fixture(format_id)

    assert len(records) == 6
    assert report == IngestReport(parsed=6, skipped=0, reasons={})

    for i, record in enumerate(records):
        assert record.id == spec["ids"][i]
        assert record.timestamp == TIMESTAMPS[i]
        assert record.model == spec["model"]
        assert record.provider == spec["provider"]
        assert record.messages_hash == expected_messages_hash(i)
        assert record.system_prefix_hash == expected_system_prefix_hash(i)
        assert record.system_prefix_chars == expected_system_prefix_chars(i)
        assert record.prompt_tokens == PROMPT_TOKENS[i]
        assert record.cached_prompt_tokens == CACHED_TOKENS[i]
        assert record.completion_tokens == COMPLETION_TOKENS[i]
        assert record.latency_ms == pytest.approx(LATENCY_MS[i], rel=1e-9)
        assert record.status == spec["statuses"][i]
        assert record.group_key == spec["groups"][i]
        assert record.raw_index == i


def test_request_without_system_prefix_uses_empty_string_hash() -> None:
    records, _ = read_fixture("openai-jsonl")
    assert SYSTEMS[3] is None
    assert records[3].system_prefix_hash == EMPTY_TEXT_HASH
    assert hashlib.sha256(b"").digest()[:16].hex() == EMPTY_TEXT_HASH
    # A request with a real prefix is never confused with one that has none.
    assert records[0].system_prefix_hash != EMPTY_TEXT_HASH


def test_hashes_are_identical_across_formats() -> None:
    per_format = {fmt: read_fixture(fmt)[0] for fmt in EXPECTATIONS}
    for i in range(6):
        message_hashes = {records[i].messages_hash for records in per_format.values()}
        prefix_hashes = {records[i].system_prefix_hash for records in per_format.values()}
        prefix_chars = {records[i].system_prefix_chars for records in per_format.values()}
        assert len(message_hashes) == 1, f"record {i} hashes differ between formats"
        assert len(prefix_hashes) == 1, f"record {i} prefix hashes differ between formats"
        assert prefix_chars == {expected_system_prefix_chars(i)}, (
            f"record {i} prefix char counts differ between formats"
        )


def test_shared_prefix_clusters_by_system_prefix_hash() -> None:
    records, _ = read_fixture("openai-jsonl")
    assert records[0].system_prefix_hash == records[1].system_prefix_hash
    assert records[4].system_prefix_hash == records[5].system_prefix_hash
    assert records[0].system_prefix_hash != records[4].system_prefix_hash


def test_system_prefix_chars_counts_content_text_only() -> None:
    """The count covers prefix content characters: not roles, framing, or non-text blocks."""
    base = {
        "id": "gen-mixed",
        "created_at": "2026-04-01T09:00:00Z",
        "model": "gpt-4o-mini",
        "provider_name": "OpenAI",
        "native_tokens_prompt": 100,
        "native_tokens_completion": 5,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "abcde"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "fg"},
                ],
            },
            {"role": "developer", "content": "hij"},
            # The prefix ends here, so nothing below is counted.
            {"role": "user", "content": "a much longer user turn that must not be counted"},
            {"role": "system", "content": "a late system message is outside the prefix"},
        ],
    }
    record = read_openrouter_export(base, 0)
    assert isinstance(record, RequestRecord)
    assert record.system_prefix_chars == len("abcde") + len("fg") + len("hij")


def test_system_prefix_chars_is_zero_without_a_prefix() -> None:
    records, _ = read_fixture("openai-jsonl")
    assert SYSTEMS[3] is None
    assert records[3].system_prefix_chars == 0
    assert records[0].system_prefix_chars > 0


def test_stable_prefix_length_with_a_churning_hash_is_visible() -> None:
    """The volatile-prefix signal: equal lengths, differing hashes, from ingest alone."""
    prefix = "You are the pangolin assistant. Session f47ac10b-58cc-4372-a567-0e02b2c3d479."
    other = prefix.replace("f47ac10b", "9c858901")
    assert len(prefix) == len(other)
    base = {
        "id": "gen-volatile-0",
        "created_at": "2026-04-01T09:00:00Z",
        "model": "gpt-4o-mini",
        "provider_name": "OpenAI",
        "native_tokens_prompt": 100,
        "native_tokens_completion": 5,
        "messages": [{"role": "system", "content": prefix}, {"role": "user", "content": "go"}],
    }
    first = read_openrouter_export(base, 0)
    second_obj = json.loads(json.dumps(base))
    second_obj["id"] = "gen-volatile-1"
    second_obj["messages"][0]["content"] = other
    second = read_openrouter_export(second_obj, 1)
    assert isinstance(first, RequestRecord)
    assert isinstance(second, RequestRecord)
    assert first.system_prefix_chars == second.system_prefix_chars == len(prefix)
    assert first.system_prefix_hash != second.system_prefix_hash


# --------------------------------------------------------------------------------------
# the privacy guarantee
# --------------------------------------------------------------------------------------


def _walk_texts(value: object, collected: set[str], *, inside_text_key: bool = False) -> None:
    if isinstance(value, str):
        if inside_text_key:
            collected.add(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_texts(item, collected, inside_text_key=key in TEXT_KEYS)
        return
    if isinstance(value, list):
        for item in value:
            _walk_texts(item, collected, inside_text_key=inside_text_key)


def fixture_texts(path: Path) -> set[str]:
    collected: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _walk_texts(json.loads(line), collected)
    return collected


def text_tokens(texts: set[str]) -> set[str]:
    """Distinctive words from prompt/completion text.

    Tokens are filtered to length >= 5 with at least one non-hex character, so a match
    inside a record can only come from retained text, never from a hex digest.
    """
    tokens: set[str] = set()
    for text in texts:
        for raw in text.replace(";", " ").replace(",", " ").replace(".", " ").split():
            token = raw.strip("?!:'\"").lower()
            if len(token) >= 5 and any(char not in "0123456789abcdef" for char in token):
                tokens.add(token)
    return tokens


def test_fixture_texts_match_the_canonical_dataset() -> None:
    """Guard against fixture/test drift: the declared texts really are in every fixture."""
    for format_id in EXPECTATIONS:
        path = FIXTURES / EXPECTATIONS[format_id]["file"]
        texts = fixture_texts(path)
        for i in range(6):
            assert USERS[i] in texts, f"{path.name} is missing user text {i}"
            assert COMPLETIONS[i] in texts, f"{path.name} is missing completion text {i}"
            if SYSTEMS[i] is not None:
                assert SYSTEMS[i] in texts, f"{path.name} is missing system text {i}"


_EXPLICIT_FORMATS = {
    "generic.jsonl": "generic-jsonl",
    "malformed_openai.jsonl": "openai-jsonl",
    "ambiguous.jsonl": "litellm-jsonl",
}
# unknown.jsonl carries no chat text at all, so it has nothing to leak.
TEXT_FIXTURES = sorted(p for p in FIXTURES.glob("*.jsonl") if p.name != "unknown.jsonl")


@pytest.mark.parametrize("path", TEXT_FIXTURES, ids=lambda p: p.name)
def test_no_prompt_or_completion_text_survives_a_read(path: Path) -> None:
    texts = fixture_texts(path)
    tokens = text_tokens(texts)
    assert len(tokens) >= 15, "the fixture must contain distinctive text for this test to bite"

    format_id = _EXPLICIT_FORMATS.get(path.name) or detect(path)
    mapping = GENERIC_MAPPING if format_id == "generic-jsonl" else None
    stream = read_requests(path, format=format_id, mapping=mapping)
    records: list[RequestRecord] = []
    haystacks: list[str] = []
    while True:
        try:
            records.append(next(stream))
        except StopIteration:
            break
        except IngestError as exc:
            # A refusal message is a public string too, and must not carry text either.
            haystacks.append(str(exc))
            break
    assert records, "the privacy test needs at least one produced record"

    haystacks.extend((repr(stream.report()), repr(stream)))
    for record in records:
        haystacks.append(repr(record))
        for name in record.__slots__:
            value = getattr(record, name)
            if isinstance(value, str):
                haystacks.append(value)
    blob = "\n".join(haystacks).lower()

    for text in texts:
        assert text.lower() not in blob
    leaked = sorted(token for token in tokens if token in blob)
    assert not leaked, f"prompt/completion text leaked into records: {leaked}"


def test_error_messages_never_echo_prompt_text() -> None:
    secret = "hippopotamus-shaped invoice"
    record = {
        "request": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": secret}]},
        "response": {
            "id": "chatcmpl-z",
            "created": 1775030400,
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": "many", "completion_tokens": 1},
        },
    }
    with pytest.raises(MalformedRecordError) as raised:
        read_openai_jsonl(record, 0)
    assert "hippopotamus" not in str(raised.value)


def test_invalid_json_error_does_not_echo_the_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "hippopotamus"} \n', encoding="utf-8"
    )
    with pytest.raises(MalformedRecordError) as raised:
        list(read_requests(path, format="openai-jsonl"))
    message = str(raised.value)
    assert "hippopotamus" not in message
    assert "record 0 (line 1)" in message
    assert "fix:" in message


# --------------------------------------------------------------------------------------
# streaming and bounded memory
# --------------------------------------------------------------------------------------

_LARGE_TARGET_BYTES = 200 * 1024 * 1024
_PEAK_BOUND_BYTES = 16 * 1024 * 1024


def _write_large_fixture(path: Path, target_bytes: int) -> int:
    pad = "kumquat pangolin zeppelin toboggan " * 220
    lines = 0
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        while written < target_bytes:
            row = {
                "request": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are the pangolin triage assistant."},
                        {"role": "user", "content": f"{pad}{lines}"},
                    ],
                },
                "response": {
                    "id": f"chatcmpl-big{lines}",
                    "created": 1775030400 + (lines % 86400),
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": pad}}],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 30},
                },
            }
            line = f"{json.dumps(row)}\n"
            handle.write(line)
            written += len(line)
            lines += 1
    return lines


def test_reader_is_lazy_and_never_materializes_the_source() -> None:
    stream = read_requests(FIXTURES / "openai.jsonl", format="openai-jsonl")
    assert isinstance(stream, Iterator)
    first = next(stream)
    assert first.raw_index == 0
    # Only the consumed record has been counted: nothing was read ahead into a list.
    assert stream.report().parsed == 1
    assert not stream.exhausted
    assert stream.format == "openai-jsonl"
    remaining = list(stream)
    assert len(remaining) == 5
    assert stream.exhausted
    assert stream.report().parsed == 6


def test_large_source_is_processed_with_bounded_memory(tmp_path: Path) -> None:
    """Bounded peak *tracked* memory via tracemalloc over a ~200 MiB source."""
    path = tmp_path / "large.jsonl"
    expected_lines = _write_large_fixture(path, _LARGE_TARGET_BYTES)
    assert path.stat().st_size >= _LARGE_TARGET_BYTES

    tracemalloc.start()
    try:
        stream = read_requests(path, format="openai-jsonl")
        seen = 0
        checksum = 0
        for record in stream:
            seen += 1
            checksum += record.prompt_tokens
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert seen == expected_lines
    assert checksum == expected_lines * 1200
    assert stream.report() == IngestReport(parsed=expected_lines, skipped=0, reasons={})
    assert peak < _PEAK_BOUND_BYTES, f"peak traced memory {peak} exceeded {_PEAK_BOUND_BYTES}"


def test_oversized_record_is_refused_before_it_is_buffered(tmp_path: Path) -> None:
    path = tmp_path / "one_giant_line.jsonl"
    with path.open("wb") as handle:
        handle.write(b'{"request": {"messages": [{"role": "user", "content": "')
        block = b"a" * (1024 * 1024)
        for _ in range(MAX_RECORD_BYTES // len(block) + 2):
            handle.write(block)
    with pytest.raises(MalformedRecordError) as raised:
        list(read_requests(path, format="openai-jsonl"))
    assert "record 0 (line 1)" in str(raised.value)
    assert "fix:" in str(raised.value)


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("openai.jsonl", "openai-jsonl"),
        ("anthropic.jsonl", "anthropic-jsonl"),
        ("litellm.jsonl", "litellm-jsonl"),
        ("helicone.jsonl", "helicone-export"),
        ("openrouter.jsonl", "openrouter-export"),
    ],
)
def test_detect_identifies_each_format(name: str, expected: str) -> None:
    assert detect(FIXTURES / name) == expected


def test_auto_detection_matches_explicit_format() -> None:
    stream = read_requests(FIXTURES / "helicone.jsonl")
    assert stream.format == "helicone-export"
    assert len(list(stream)) == 6


def test_ambiguous_source_refuses_and_names_every_candidate() -> None:
    with pytest.raises(AmbiguousFormatError) as raised:
        detect(FIXTURES / "ambiguous.jsonl")
    assert raised.value.candidates == ("litellm-jsonl", "openrouter-export")
    message = str(raised.value)
    assert "litellm-jsonl" in message
    assert "openrouter-export" in message
    assert "fix:" in message


def test_auto_read_of_an_ambiguous_source_refuses() -> None:
    with pytest.raises(AmbiguousFormatError):
        read_requests(FIXTURES / "ambiguous.jsonl")


def test_unrecognized_source_refuses_with_the_candidate_list() -> None:
    with pytest.raises(UnknownFormatError) as raised:
        detect(FIXTURES / "unknown.jsonl")
    message = str(raised.value)
    assert "generic-jsonl" in message
    assert "fix:" in message


def test_generic_format_is_never_auto_detected() -> None:
    with pytest.raises(UnknownFormatError):
        detect(FIXTURES / "generic.jsonl")


def test_empty_source_refuses(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n\n", encoding="utf-8")
    with pytest.raises(EmptySourceError) as raised:
        detect(path)
    assert "fix:" in str(raised.value)


def test_missing_source_refuses(tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError) as raised:
        read_requests(tmp_path / "nope.jsonl")
    assert "fix:" in str(raised.value)
    with pytest.raises(SourceNotFoundError):
        read_requests(tmp_path)


def test_unsupported_format_id_refuses() -> None:
    with pytest.raises(UnknownFormatError) as raised:
        read_requests(FIXTURES / "openai.jsonl", format="datadog")
    message = str(raised.value)
    assert "fix:" in message
    for format_id in FORMAT_IDS:
        assert format_id in message


# --------------------------------------------------------------------------------------
# malformed records and skips
# --------------------------------------------------------------------------------------


def test_malformed_record_names_its_index_and_field() -> None:
    stream = read_requests(FIXTURES / "malformed_openai.jsonl", format="openai-jsonl")
    assert [record.raw_index for record in (next(stream), next(stream))] == [0, 1]
    with pytest.raises(MalformedRecordError) as raised:
        next(stream)
    assert raised.value.index == 2
    assert raised.value.field == "response.usage.prompt_tokens"
    message = str(raised.value)
    assert "record 2 (line 3)" in message
    assert "fix:" in message


def test_skipped_records_are_counted_with_reasons() -> None:
    stream = read_requests(FIXTURES / "skips_openai.jsonl", format="openai-jsonl")
    records = list(stream)
    assert [record.id for record in records] == ["chatcmpl-s0", "chatcmpl-s3"]
    assert [record.raw_index for record in records] == [0, 4]
    report = stream.report()
    assert report == IngestReport(
        parsed=2,
        skipped=3,
        reasons={"non_chat_request": 1, "blank_line": 1, "missing_usage": 1},
    )
    assert report.total() == 5
    assert set(report.reasons) <= SKIP_REASONS


def test_non_chat_call_types_are_skipped_not_raised() -> None:
    report = read_requests_report(FIXTURES / "litellm_mixed.jsonl")
    assert report == IngestReport(parsed=2, skipped=1, reasons={"non_chat_call_type": 1})


def test_rows_without_message_content_are_skipped_not_hashed_as_empty() -> None:
    stream = read_requests(FIXTURES / "openrouter_no_text.jsonl")
    records = list(stream)
    assert [record.id for record in records] == ["gen-n1"]
    assert stream.report() == IngestReport(
        parsed=1, skipped=2, reasons={"missing_message_content": 2}
    )


def test_report_is_available_mid_iteration_and_after_exhaustion() -> None:
    stream = read_requests(FIXTURES / "skips_openai.jsonl", format="openai-jsonl")
    assert stream.report() == IngestReport(parsed=0, skipped=0, reasons={})
    next(stream)
    assert stream.report().parsed == 1
    list(stream)
    assert stream.exhausted
    assert stream.report().skipped == 3


def test_read_requests_report_consumes_the_whole_source() -> None:
    report = read_requests_report(FIXTURES / "openai.jsonl")
    assert report == IngestReport(parsed=6, skipped=0, reasons={})


def test_timestamps_in_milliseconds_are_refused_rather_than_misread() -> None:
    record = {
        "request": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        "response": {
            "id": "chatcmpl-ms",
            "created": 1775030400000,
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    }
    with pytest.raises(MalformedRecordError) as raised:
        read_openai_jsonl(record, 7)
    assert raised.value.field == "response.created"
    assert "epoch-seconds range" in str(raised.value)
    assert "fix:" in str(raised.value)


@pytest.mark.parametrize(
    ("status_value", "expected"),
    [
        (200, "ok"),
        (408, "timeout"),
        (429, "rate_limited"),
        (499, "cancelled"),
        (500, "error"),
    ],
)
def test_http_status_codes_map_to_the_status_vocabulary(status_value: int, expected: str) -> None:
    record = json.loads((FIXTURES / "helicone.jsonl").read_text(encoding="utf-8").splitlines()[0])
    record["status"] = status_value
    outcome = read_helicone_export(record, 0)
    assert isinstance(outcome, RequestRecord)
    assert outcome.status == expected


def test_unrecognized_http_status_is_refused() -> None:
    record = json.loads((FIXTURES / "helicone.jsonl").read_text(encoding="utf-8").splitlines()[0])
    record["status"] = 999
    with pytest.raises(MalformedRecordError) as raised:
        read_helicone_export(record, 0)
    assert "fix:" in str(raised.value)


def test_litellm_end_before_start_is_refused() -> None:
    record = json.loads((FIXTURES / "litellm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    record["endTime"], record["startTime"] = record["startTime"], record["endTime"]
    with pytest.raises(MalformedRecordError) as raised:
        read_litellm_jsonl(record, 3)
    assert raised.value.field == "endTime"
    assert "fix:" in str(raised.value)


def test_non_text_content_blocks_contribute_no_data_to_the_digest() -> None:
    base = {
        "id": "gen-img",
        "created_at": "2026-04-01T09:00:00Z",
        "model": "gpt-4o-mini",
        "provider_name": "OpenAI",
        "native_tokens_prompt": 100,
        "native_tokens_completion": 5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
    }
    first = read_openrouter_export(base, 0)
    other = json.loads(json.dumps(base))
    other["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64,BBBB"
    second = read_openrouter_export(other, 0)
    assert isinstance(first, RequestRecord)
    assert isinstance(second, RequestRecord)
    assert first.messages_hash == second.messages_hash
    assert first.messages_hash == digest_messages(
        [("user", [("text", "describe this"), ("image_url", "")])]
    )


# --------------------------------------------------------------------------------------
# generic-jsonl mapping
# --------------------------------------------------------------------------------------


def test_generic_format_without_a_mapping_refuses() -> None:
    with pytest.raises(MappingError) as raised:
        read_requests(FIXTURES / "generic.jsonl", format="generic-jsonl")
    message = str(raised.value)
    assert "fix:" in message
    for key in ("id", "timestamp", "model", "provider", "messages", "prompt_tokens"):
        assert key in message


def test_generic_mapping_missing_required_keys_refuses() -> None:
    mapping = {key: value for key, value in GENERIC_MAPPING.items() if key != "prompt_tokens"}
    with pytest.raises(MappingError) as raised:
        read_requests(FIXTURES / "generic.jsonl", format="generic-jsonl", mapping=mapping)
    assert "prompt_tokens" in str(raised.value)
    assert "fix:" in str(raised.value)


def test_generic_mapping_with_unknown_keys_refuses() -> None:
    mapping = dict(GENERIC_MAPPING, cost_usd="spend")
    with pytest.raises(MappingError) as raised:
        read_requests(FIXTURES / "generic.jsonl", format="generic-jsonl", mapping=mapping)
    assert "cost_usd" in str(raised.value)
    assert "fix:" in str(raised.value)


def test_generic_mapping_pointing_at_a_missing_field_names_the_path() -> None:
    mapping = dict(GENERIC_MAPPING, prompt_tokens="counters.input_tokens")
    stream = read_requests(FIXTURES / "generic.jsonl", format="generic-jsonl", mapping=mapping)
    with pytest.raises(MalformedRecordError) as raised:
        next(stream)
    assert raised.value.field == "counters.input_tokens"
    assert "fix:" in str(raised.value)


def test_generic_unmapped_status_is_unknown_not_assumed_ok() -> None:
    mapping = {key: value for key, value in GENERIC_MAPPING.items() if key != "status"}
    records = list(
        read_requests(FIXTURES / "generic.jsonl", format="generic-jsonl", mapping=mapping)
    )
    assert {record.status for record in records} == {"unknown"}


def test_mapping_with_a_non_generic_format_refuses() -> None:
    with pytest.raises(MappingError) as raised:
        read_requests(FIXTURES / "openai.jsonl", format="openai-jsonl", mapping=GENERIC_MAPPING)
    assert "fix:" in str(raised.value)


# --------------------------------------------------------------------------------------
# value-type guards
# --------------------------------------------------------------------------------------


def valid_record_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "id": "chatcmpl-1",
        "timestamp": 1775030400.0,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "messages_hash": "0" * 32,
        "system_prefix_hash": EMPTY_TEXT_HASH,
        "prompt_tokens": 100,
        "cached_prompt_tokens": 64,
        "completion_tokens": 10,
        "latency_ms": 12.5,
        "status": "ok",
        "group_key": None,
        "raw_index": 0,
        "system_prefix_chars": 76,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": " "},
        {"model": ""},
        {"provider": "OpenAI"},
        {"timestamp": 0.0},
        {"timestamp": float("inf")},
        {"timestamp": float("nan")},
        {"messages_hash": "ABCDEF0123456789abcdef0123456789"},
        {"messages_hash": "beef"},
        {"system_prefix_hash": "zz" * 16},
        {"prompt_tokens": -1},
        {"completion_tokens": True},
        {"cached_prompt_tokens": 101},
        {"cached_prompt_tokens": -1},
        {"latency_ms": -0.5},
        {"latency_ms": float("nan")},
        {"status": "success"},
        {"group_key": ""},
        {"raw_index": -1},
        {"system_prefix_chars": -1},
        {"system_prefix_chars": 76.0},
        {"system_prefix_chars": True},
    ],
)
def test_request_record_refuses_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(IngestError) as raised:
        RequestRecord(**valid_record_kwargs(**overrides))  # type: ignore[arg-type]
    assert "fix:" in str(raised.value)


def test_request_record_is_frozen_and_slotted() -> None:
    record = RequestRecord(**valid_record_kwargs())  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        record.prompt_tokens = 1  # type: ignore[misc]
    assert not hasattr(record, "__dict__")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parsed": -1, "skipped": 0, "reasons": {}},
        {"parsed": 0, "skipped": 1, "reasons": {}},
        {"parsed": 0, "skipped": 1, "reasons": {"blank_line": 2}},
        {"parsed": 0, "skipped": 1, "reasons": {"because_reasons": 1}},
        {"parsed": 0, "skipped": 0, "reasons": {"blank_line": 0}},
    ],
)
def test_ingest_report_refuses_inconsistent_counts(kwargs: dict[str, object]) -> None:
    with pytest.raises(IngestError) as raised:
        IngestReport(**kwargs)  # type: ignore[arg-type]
    assert "fix:" in str(raised.value)
