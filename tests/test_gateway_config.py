from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from branchpilot.gateway import config as gateway_config
from branchpilot.gateway.config import ConfigError, load_gateway_config
from branchpilot.gateway.schemas import MAX_COMPLETION_TOKENS
from branchpilot.strategies import FixedStrategy


def _payload(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "inbound_api_key_envs": ["INBOUND_KEY"],
        "request_timeout_s": 30,
        "queue_timeout_s": 1,
        "max_concurrent_sessions": 4,
        "upstreams": {
            "local": {
                "base_url": "http://inference.internal:8000/v1",
                "api_key_env": "UPSTREAM_KEY",
                "max_connections": 2,
            }
        },
        "models": {
            "math": {
                "upstream": "local",
                "upstream_model": "private/model",
                "plan_path": "plans/deployment.json",
                "extractor": "numeric-strict",
                "max_samples": 2,
                "max_completion_tokens": 128,
                "options": {"temperature": 0.7},
            }
        },
    }
    value.update(changes)
    return value


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def loaded_plan(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    seen: list[Path] = []

    def load(path: Path):
        seen.append(path)
        deployment = SimpleNamespace(
            strategy=FixedStrategy(samples=2, max_samples=3),
            cost=0.0,
            spec={"type": "fixed", "samples": 2, "max_samples": 3},
        )
        plan = SimpleNamespace(policy="fixed-2", family="fixed")
        return deployment, plan

    monkeypatch.setattr(gateway_config, "load_deployment_plan", load)
    return seen


def test_loads_relative_plan_and_resolves_separate_secrets(
    tmp_path: Path, loaded_plan: list[Path]
) -> None:
    path = _write(tmp_path, _payload())
    config = load_gateway_config(
        path, environ={"INBOUND_KEY": "client-secret", "UPSTREAM_KEY": "provider-secret"}
    )

    assert config.inbound_api_keys.matches("client-secret")
    assert config.upstreams["local"].api_key == "provider-secret"
    assert config.upstreams["local"].base_url == "http://inference.internal:8000/v1"
    assert config.models["math"].upstream_model == "private/model"
    assert config.models["math"].max_samples == 2
    assert loaded_plan == [tmp_path / "plans/deployment.json"]
    assert "client-secret" not in repr(config)
    assert "provider-secret" not in repr(config.upstreams["local"])


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///tmp/socket",
        "ftp://inference.internal/v1",
        "https://user:password@inference.internal/v1",
        "https://inference.internal/v1?key=secret",
        "//inference.internal/v1",
    ],
)
def test_rejects_unsafe_upstream_urls(
    tmp_path: Path, loaded_plan: list[Path], base_url: str
) -> None:
    payload = _payload()
    payload["upstreams"]["local"]["base_url"] = base_url  # type: ignore[index]
    with pytest.raises(ConfigError, match="base_url"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )


@pytest.mark.parametrize("missing", ["INBOUND_KEY", "UPSTREAM_KEY"])
def test_missing_or_empty_secret_fails_closed(
    tmp_path: Path, loaded_plan: list[Path], missing: str
) -> None:
    environment = {"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"}
    environment[missing] = " "
    with pytest.raises(ConfigError, match=missing):
        load_gateway_config(_write(tmp_path, _payload()), environ=environment)


@pytest.mark.parametrize("reserved", ["model", "messages", "n", "stream", "extra_headers"])
def test_rejects_reserved_model_options(
    tmp_path: Path, loaded_plan: list[Path], reserved: str
) -> None:
    payload = _payload()
    payload["models"]["math"]["options"] = {reserved: "escape"}  # type: ignore[index]
    with pytest.raises(ConfigError, match="reserved"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )


def test_requires_bounded_route_completion_cap(tmp_path: Path, loaded_plan: list[Path]) -> None:
    missing = _payload()
    del missing["models"]["math"]["max_completion_tokens"]  # type: ignore[index]
    with pytest.raises(ConfigError, match="max_completion_tokens"):
        load_gateway_config(
            _write(tmp_path, missing),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )

    excessive = _payload()
    excessive["models"]["math"]["max_completion_tokens"] = (  # type: ignore[index]
        MAX_COMPLETION_TOKENS + 1
    )
    with pytest.raises(ConfigError, match="max_completion_tokens"):
        load_gateway_config(
            _write(tmp_path, excessive),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )


def test_rejects_excessive_upstream_response_cap(tmp_path: Path, loaded_plan: list[Path]) -> None:
    payload = _payload()
    payload["upstreams"]["local"]["max_response_bytes"] = 67_108_865  # type: ignore[index]
    with pytest.raises(ConfigError, match="max_response_bytes"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )


def test_rejects_unknown_alias_extractor_and_excessive_horizon(
    tmp_path: Path, loaded_plan: list[Path]
) -> None:
    payload = _payload()
    payload["models"]["math"]["upstream"] = "missing"  # type: ignore[index]
    with pytest.raises(ConfigError, match="unknown upstream"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )

    payload = _payload()
    payload["models"]["math"]["extractor"] = "dotted.path:callable"  # type: ignore[index]
    with pytest.raises(ConfigError, match="unknown extractor"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )

    payload = _payload()
    payload["models"]["math"]["max_samples"] = 4  # type: ignore[index]
    with pytest.raises(ConfigError, match="horizon"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )


def test_rejects_unknown_config_fields_and_nonfinite_json(
    tmp_path: Path, loaded_plan: list[Path]
) -> None:
    payload = _payload(literal_api_key="must-never-be-accepted")
    with pytest.raises(ConfigError, match="invalid gateway config"):
        load_gateway_config(
            _write(tmp_path, payload),
            environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"},
        )

    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(_payload()).replace('"temperature": 0.7', '"temperature": NaN'))
    with pytest.raises(ConfigError):
        load_gateway_config(path, environ={"INBOUND_KEY": "client", "UPSTREAM_KEY": "upstream"})
