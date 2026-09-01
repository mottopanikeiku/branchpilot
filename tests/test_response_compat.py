"""Byte-compatibility goldens for the client-visible chat-completion response.

Every BranchPilot lever sits between an OpenAI-compatible client and an upstream provider, so
the one contract that must never drift is the response body the client actually receives. This
suite pins that body for every provider adapter and every lever path that exists today, driving
the real ASGI app over a mocked transport: no network call is made, and no gateway internals are
reimplemented here.

Normalization
-------------
``id`` and ``created`` are the only two response fields the gateway mints fresh per request.
Before any comparison they are replaced with the fixed placeholders ``chatcmpl-bp-<normalized>``
and ``0``; their real values are asserted separately against the documented shape (a
``chatcmpl-bp-`` prefixed url-safe token of 24 characters, and a current unix timestamp).
The additive ``branchpilot`` extension key is dropped from both sides before comparison, because
it is the one namespace a lever is allowed to grow. Nothing else is normalized: every other
field path and value is compared against the golden exactly.

Allow-list, stated once
-----------------------
``id``, ``created``, and the top-level ``branchpilot`` key. Any other difference — a new field, a
dropped field, a changed value, at the top level or nested at any depth — fails.

Extending this suite
--------------------
Add one row to ``LEVER_PATHS`` and one golden per provider named
``tests/golden/responses/<provider>-<lever>.json``. Commented placeholder rows for the levers
that are still to be built are kept below so the next change is a row, not a new test file.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from branchpilot.gateway.app import create_app
from branchpilot.gateway.config import GatewayConfig, ModelRoute, UpstreamConfig
from branchpilot.gateway.providers import PROVIDER_IDS
from branchpilot.gateway.upstream import OpenAIUpstream
from branchpilot.strategies import strategy_from_spec

GOLDEN_DIR = Path(__file__).parent / "golden" / "responses"

EXTENSION_KEY = "branchpilot"
NORMALIZED_ID = "chatcmpl-bp-<normalized>"
NORMALIZED_CREATED = 0
VOLATILE_FIELDS = ("created", "id")
_ID_PATTERN = re.compile(r"^chatcmpl-bp-[A-Za-z0-9_-]{24}$")
_CREATED_SKEW_S = 300

_ROUTE_ALIAS = "public-math"
_UPSTREAM_MODEL = "private/model"
_INBOUND_KEY = "inbound-key"


@dataclass(frozen=True)
class LeverPath:
    """One end-to-end path through the gateway whose response shape is pinned by goldens."""

    name: str
    upstream_samples: int
    spec: Mapping[str, Any] = field(repr=False)


LEVER_PATHS: tuple[LeverPath, ...] = (
    LeverPath(
        name="single_sample_passthrough",
        upstream_samples=1,
        spec={"type": "fixed", "samples": 1, "max_samples": 1},
    ),
    LeverPath(
        name="multi_sample_adaptive",
        upstream_samples=3,
        spec={"type": "fixed", "samples": 3, "max_samples": 3},
    ),
    # Placeholders for the levers still to be built. Uncomment a row and add one golden per
    # provider; the parametrization, the golden-coverage test, and every assertion below pick
    # the new path up with no further edits.
    #
    # LeverPath(
    #     name="cache_hit",
    #     upstream_samples=1,
    #     spec={"type": "fixed", "samples": 1, "max_samples": 1},
    # ),
    # LeverPath(
    #     name="routed",
    #     upstream_samples=1,
    #     spec={"type": "fixed", "samples": 1, "max_samples": 1},
    # ),
    # LeverPath(
    #     name="batch_materialized",
    #     upstream_samples=1,
    #     spec={"type": "fixed", "samples": 1, "max_samples": 1},
    # ),
)

CASES = tuple(
    pytest.param(provider, lever, id=f"{provider}-{lever.name}")
    for lever in LEVER_PATHS
    for provider in PROVIDER_IDS
)


def _golden_path(provider: str, lever: LeverPath) -> Path:
    return GOLDEN_DIR / f"{provider}-{lever.name}.json"


def _golden(provider: str, lever: LeverPath) -> dict[str, Any]:
    golden = json.loads(_golden_path(provider, lever).read_text(encoding="utf-8"))
    assert golden["provider"] == provider
    assert golden["lever_path"] == lever.name
    assert len(golden["upstream_responses"]) == lever.upstream_samples
    return golden


def _served(provider: str, lever: LeverPath, golden: Mapping[str, Any]) -> tuple[Any, int]:
    """Drive the real app against the golden's recorded upstream payloads, over no network."""
    scripted: list[Any] = list(golden["upstream_responses"])
    served = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal served
        assert request.url.host.endswith(".invalid")
        payload = scripted[served]
        served += 1
        return httpx.Response(200, json=payload)

    upstream_config = UpstreamConfig(
        name="local",
        base_url="http://upstream.invalid",
        api_key="provider-key",
        provider=provider,
    )
    upstream = OpenAIUpstream(upstream_config, 5.0, transport=httpx.MockTransport(handler))
    spec = dict(lever.spec)
    route = ModelRoute(
        alias=_ROUTE_ALIAS,
        upstream="local",
        upstream_model=_UPSTREAM_MODEL,
        deployment=SimpleNamespace(strategy=strategy_from_spec(dict(spec)), cost=0.0, spec=spec),
        plan=SimpleNamespace(policy=lever.name, family="fixed"),
        extractor="exact-content",
        max_samples=lever.upstream_samples,
        max_completion_tokens=64,
        options={"temperature": 0.8},
        allow_client_overrides=False,
        allowed_strategies=frozenset(),
    )
    config = GatewayConfig(
        inbound_api_keys=(_INBOUND_KEY,),
        request_timeout_s=5.0,
        queue_timeout_s=1.0,
        max_concurrent_sessions=2,
        upstreams={"local": upstream_config},
        models={_ROUTE_ALIAS: route},
    )
    app = create_app(config, strategies={}, upstreams={"local": upstream})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": _ROUTE_ALIAS, "messages": [{"role": "user", "content": "question"}]},
            headers={"authorization": f"Bearer {_INBOUND_KEY}"},
        )
    return response, served


def _key_paths(value: Any, prefix: str = "") -> set[str]:
    """Every recursive field path in a decoded JSON body, list positions included."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            paths.add(path)
            paths |= _key_paths(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths |= _key_paths(item, f"{prefix}[{index}]")
    return paths


def _normalized(body: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(body))
    if "id" in value:
        value["id"] = NORMALIZED_ID
    if "created" in value:
        value["created"] = NORMALIZED_CREATED
    value.pop(EXTENSION_KEY, None)
    return value


def _unknown_paths(value: Any, prefix: str = "") -> set[str]:
    """Field paths the OpenAI SDK types do not declare, found at any depth."""
    paths: set[str] = set()
    if isinstance(value, BaseModel):
        for name in value.model_extra or ():
            paths.add(f"{prefix}.{name}" if prefix else str(name))
        for name in type(value).model_fields:
            child = f"{prefix}.{name}" if prefix else name
            paths |= _unknown_paths(getattr(value, name), child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths |= _unknown_paths(item, f"{prefix}[{index}]")
    return paths


def _root(path: str) -> str:
    return path.split(".", 1)[0].split("[", 1)[0]


def _assert_matches_golden(body: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    live = _normalized(body)
    want = _normalized(expected)
    stray = _key_paths(live) - _key_paths(want)
    missing = _key_paths(want) - _key_paths(live)
    assert not stray, f"response grew field paths the golden does not have: {sorted(stray)}"
    assert not missing, f"response dropped golden field paths: {sorted(missing)}"
    assert live == want


def _assert_single_extension(body: Mapping[str, Any]) -> None:
    """All BranchPilot additions live under one top-level ``branchpilot`` key, or nowhere."""
    completion = ChatCompletion.model_validate(body)
    outside = {path for path in _unknown_paths(completion) if _root(path) != EXTENSION_KEY}
    assert not outside, (
        "non-OpenAI response fields must live under the single top-level "
        f"{EXTENSION_KEY!r} key; found: {sorted(outside)}"
    )


def _assert_volatile_fields(body: Mapping[str, Any]) -> None:
    assert _ID_PATTERN.fullmatch(body["id"]), body["id"]
    created = body["created"]
    assert isinstance(created, int) and not isinstance(created, bool)
    assert abs(created - int(time.time())) < _CREATED_SKEW_S


def test_golden_files_cover_exactly_the_registered_providers_and_levers() -> None:
    expected = {
        f"{provider}-{lever.name}.json" for lever in LEVER_PATHS for provider in PROVIDER_IDS
    }
    assert {path.name for path in GOLDEN_DIR.glob("*.json")} == expected


@pytest.mark.parametrize(("provider", "lever"), CASES)
def test_golden_body_validates_against_the_openai_sdk_type(provider: str, lever: LeverPath) -> None:
    golden = _golden(provider, lever)
    completion = ChatCompletion.model_validate(golden["expected_response"])
    assert completion.object == "chat.completion"
    assert completion.model == _ROUTE_ALIAS
    assert len(completion.choices) == 1
    _assert_single_extension(golden["expected_response"])


@pytest.mark.parametrize(("provider", "lever"), CASES)
def test_client_visible_body_matches_the_golden(provider: str, lever: LeverPath) -> None:
    golden = _golden(provider, lever)
    response, served = _served(provider, lever, golden)
    assert response.status_code == 200
    assert served == lever.upstream_samples
    body = response.json()
    _assert_volatile_fields(body)
    _assert_matches_golden(body, golden["expected_response"])
    ChatCompletion.model_validate(body)


@pytest.mark.parametrize(("provider", "lever"), CASES)
def test_client_visible_body_carries_one_extension_namespace(
    provider: str, lever: LeverPath
) -> None:
    golden = _golden(provider, lever)
    response, _ = _served(provider, lever, golden)
    body = response.json()
    _assert_single_extension(body)
    known = set(ChatCompletion.model_fields) | {EXTENSION_KEY}
    assert set(body) <= known, f"stray top-level fields: {sorted(set(body) - known)}"


@pytest.mark.parametrize(("provider", "lever"), CASES)
def test_branchpilot_headers_match_the_golden(provider: str, lever: LeverPath) -> None:
    golden = _golden(provider, lever)
    response, _ = _served(provider, lever, golden)
    observed = {
        name: value
        for name, value in response.headers.items()
        if name.lower().startswith("x-branchpilot")
    }
    assert observed == golden["expected_headers"]


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        pytest.param(
            {"path": ("branchpilot_debug",), "value": {"cost": "0.01"}},
            "single top-level",
            id="second-top-level-extension-namespace",
        ),
        pytest.param(
            {"path": ("cost_saved_usd",), "value": "0.01"},
            "single top-level",
            id="bare-top-level-field",
        ),
        pytest.param(
            {"path": ("usage", "branchpilot_saving"), "value": 3},
            "single top-level",
            id="nested-usage-field",
        ),
        pytest.param(
            {"path": ("choices", 0, "message", "bp_selected_index"), "value": 1},
            "single top-level",
            id="nested-choice-field",
        ),
    ],
)
def test_a_stray_field_anywhere_fails_the_suite(
    mutation: dict[str, Any], expected_message: str
) -> None:
    """Negative control: the checks the passing tests rely on must reject injected drift."""
    golden = _golden("openai", LEVER_PATHS[0])
    expected = golden["expected_response"]
    mutated = copy.deepcopy(expected)
    target: Any = mutated
    for step in mutation["path"][:-1]:
        target = target[step]
    target[mutation["path"][-1]] = mutation["value"]

    with pytest.raises(AssertionError) as drift:
        _assert_matches_golden(mutated, expected)
    assert "grew field paths" in str(drift.value)

    with pytest.raises(AssertionError) as extension:
        _assert_single_extension(mutated)
    assert expected_message in str(extension.value)


def test_the_documented_allow_list_is_the_only_tolerated_difference() -> None:
    """A fresh id, a fresh created, and a branchpilot key pass; a fourth difference does not."""
    golden = _golden("openai", LEVER_PATHS[0])
    expected = golden["expected_response"]
    allowed = copy.deepcopy(expected)
    allowed["id"] = "chatcmpl-bp-" + "A" * 24
    allowed["created"] = 1730000000
    allowed[EXTENSION_KEY] = {"samples": 1, "selection": "majority"}
    _assert_matches_golden(allowed, expected)
    _assert_single_extension(allowed)
    assert set(VOLATILE_FIELDS) | {EXTENSION_KEY} == {"created", "id", EXTENSION_KEY}

    drifted = copy.deepcopy(allowed)
    drifted["choices"][0]["message"]["content"] = "43"
    with pytest.raises(AssertionError):
        _assert_matches_golden(drifted, expected)
