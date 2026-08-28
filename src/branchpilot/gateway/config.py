from __future__ import annotations

import copy
import json
import math
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from branchpilot.deployment import load_deployment_plan
from branchpilot.gateway.providers import (
    CANONICAL_API_KEY,
    CANONICAL_REQUEST_ID,
    DEFAULT_PROVIDER,
    UnknownProviderError,
    resolve_adapter,
)
from branchpilot.gateway.schemas import MAX_COMPLETION_TOKENS, ChatCompletionRequest

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_OPTIONS = frozenset(
    {"model", "messages", "n", "stream", "extra_body", "extra_headers", "extra_query", "timeout"}
)
_FORBIDDEN_FEATURE_FIELDS = frozenset(
    {
        "audio",
        "function_call",
        "functions",
        "modalities",
        "parallel_tool_calls",
        "tool_choice",
        "tools",
        "web_search_options",
    }
)
_FIXED_EXTRA_RESERVED = (
    _RESERVED_OPTIONS
    | _FORBIDDEN_FEATURE_FIELDS
    | frozenset({CANONICAL_API_KEY, CANONICAL_REQUEST_ID})
    | frozenset(
        {
            "branchpilot",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "temperature",
            "top_logprobs",
            "top_p",
            "user",
        }
    )
)
_BUILTIN_EXTRACTORS = frozenset({"exact-content", "numeric-strict"})
_DEFAULT_MAX_UPSTREAM_RESPONSE_BYTES = 4_194_304
_HARD_MAX_UPSTREAM_RESPONSE_BYTES = 67_108_864


class ConfigError(ValueError):
    """A public, secret-free operator configuration error."""


class InboundAPIKeys:
    """Opaque inbound credentials that remain redacted in nested representations."""

    __slots__ = ("__keys",)

    def __init__(self, keys: Sequence[str]) -> None:
        self.__keys = tuple(keys)

    def matches(self, candidate: str) -> bool:
        return any(secrets.compare_digest(candidate, expected) for expected in self.__keys)

    def __repr__(self) -> str:
        return "InboundAPIKeys(<redacted>)"


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class UpstreamFileConfig(_ConfigModel):
    base_url: StrictStr = Field(min_length=1, max_length=2048)
    api_key_env: StrictStr = Field(min_length=1, max_length=128, pattern=_ENV_NAME.pattern)
    provider: StrictStr = Field(default=DEFAULT_PROVIDER, min_length=1, max_length=64)
    connect_timeout_s: StrictFloat | StrictInt = Field(default=5.0, gt=0, le=3600)
    read_timeout_s: StrictFloat | StrictInt = Field(default=60.0, gt=0, le=3600)
    write_timeout_s: StrictFloat | StrictInt = Field(default=10.0, gt=0, le=3600)
    pool_timeout_s: StrictFloat | StrictInt = Field(default=5.0, gt=0, le=3600)
    max_connections: StrictInt = Field(default=32, ge=1, le=10_000)
    max_response_bytes: StrictInt = Field(
        default=_DEFAULT_MAX_UPSTREAM_RESPONSE_BYTES,
        ge=1,
        le=_HARD_MAX_UPSTREAM_RESPONSE_BYTES,
    )
    fixed_extra_body: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        return value.rstrip("/")

    @field_validator("connect_timeout_s", "read_timeout_s", "write_timeout_s", "pool_timeout_s")
    @classmethod
    def finite_timeout(cls, value: float | int) -> float | int:
        if not math.isfinite(float(value)):
            raise ValueError("timeout must be finite")
        return value

    @field_validator("provider")
    @classmethod
    def registered_provider(cls, value: str) -> str:
        try:
            resolve_adapter(value)
        except UnknownProviderError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("fixed_extra_body")
    @classmethod
    def safe_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        _finite_json(value, "fixed_extra_body")
        forbidden = _FIXED_EXTRA_RESERVED.intersection(value)
        if forbidden:
            raise ValueError(
                f"fixed_extra_body contains reserved or unsupported field(s): "
                f"{', '.join(sorted(forbidden))}"
            )
        return value


class ModelFileConfig(_ConfigModel):
    upstream: StrictStr = Field(min_length=1, max_length=128)
    upstream_model: StrictStr = Field(min_length=1, max_length=256)
    plan_path: StrictStr = Field(min_length=1, max_length=4096)
    extractor: StrictStr = Field(default="numeric-strict", min_length=1, max_length=128)
    max_samples: StrictInt | None = Field(default=None, ge=1)
    max_completion_tokens: StrictInt = Field(ge=1, le=MAX_COMPLETION_TOKENS)
    options: dict[StrictStr, Any] = Field(default_factory=dict)
    allow_client_overrides: StrictBool = False
    allowed_strategies: list[StrictStr] = Field(default_factory=list, max_length=64)

    @field_validator("options")
    @classmethod
    def safe_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        _finite_json(value, "model options")
        reserved = _RESERVED_OPTIONS.intersection(value)
        if reserved:
            raise ValueError(
                f"model options contain reserved field(s): {', '.join(sorted(reserved))}"
            )
        try:
            ChatCompletionRequest.model_validate(
                {
                    "model": "configured-model",
                    "messages": [{"role": "user", "content": "configured request"}],
                    **value,
                }
            )
        except ValidationError as exc:
            raise ValueError("model options contain unsupported or invalid values") from exc
        return value

    @model_validator(mode="after")
    def coherent_overrides(self) -> ModelFileConfig:
        if len(set(self.allowed_strategies)) != len(self.allowed_strategies):
            raise ValueError("allowed_strategies must not contain duplicates")
        if self.allowed_strategies and not self.allow_client_overrides:
            raise ValueError("allowed_strategies require allow_client_overrides=true")
        return self


class GatewayFileConfig(_ConfigModel):
    inbound_api_key_envs: list[StrictStr] = Field(min_length=1, max_length=64)
    request_timeout_s: StrictFloat | StrictInt = Field(default=90.0, gt=0, le=86_400)
    queue_timeout_s: StrictFloat | StrictInt = Field(default=2.0, gt=0, le=3600)
    max_concurrent_sessions: StrictInt = Field(default=32, ge=1, le=10_000)
    body_read_timeout_s: StrictFloat | StrictInt = Field(default=10.0, gt=0, le=3600)
    upstreams: dict[StrictStr, UpstreamFileConfig] = Field(min_length=1, max_length=128)
    models: dict[StrictStr, ModelFileConfig] = Field(min_length=1, max_length=1024)

    @field_validator("inbound_api_key_envs")
    @classmethod
    def valid_inbound_envs(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("inbound_api_key_envs must not contain duplicates")
        if any(not _ENV_NAME.fullmatch(value) for value in values):
            raise ValueError("inbound_api_key_envs contains an invalid environment variable name")
        return values

    @field_validator("request_timeout_s", "queue_timeout_s", "body_read_timeout_s")
    @classmethod
    def finite_gateway_timeout(cls, value: float | int) -> float | int:
        if not math.isfinite(float(value)):
            raise ValueError("timeout must be finite")
        return value


@dataclass(frozen=True, slots=True)
class UpstreamConfig:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0
    write_timeout_s: float = 10.0
    pool_timeout_s: float = 5.0
    max_connections: int = 32
    max_response_bytes: int = _DEFAULT_MAX_UPSTREAM_RESPONSE_BYTES
    fixed_extra_body: Mapping[str, Any] = field(default_factory=dict)
    provider: str = DEFAULT_PROVIDER

    def __post_init__(self) -> None:
        try:
            resolve_adapter(self.provider)
        except UnknownProviderError as exc:
            raise ConfigError(str(exc)) from exc
        object.__setattr__(
            self,
            "fixed_extra_body",
            MappingProxyType(copy.deepcopy(dict(self.fixed_extra_body))),
        )


@dataclass(frozen=True, slots=True)
class ModelRoute:
    alias: str
    upstream: str
    upstream_model: str
    deployment: Any
    plan: Any
    extractor: str
    max_samples: int
    max_completion_tokens: int
    options: Mapping[str, Any]
    allow_client_overrides: bool
    allowed_strategies: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(copy.deepcopy(dict(self.options))))
        object.__setattr__(self, "allowed_strategies", frozenset(self.allowed_strategies))

    @property
    def strategy(self) -> Any:
        return self.deployment.strategy

    @property
    def strategy_name(self) -> str:
        return str(self.deployment.spec.get("type", self.plan.family))

    @property
    def cost(self) -> float:
        return float(self.deployment.cost)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    inbound_api_keys: InboundAPIKeys | Sequence[str] = field(repr=False)
    request_timeout_s: float
    queue_timeout_s: float
    max_concurrent_sessions: int
    upstreams: Mapping[str, UpstreamConfig]
    models: Mapping[str, ModelRoute]
    body_read_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.inbound_api_keys, InboundAPIKeys):
            object.__setattr__(self, "inbound_api_keys", InboundAPIKeys(self.inbound_api_keys))
        object.__setattr__(self, "upstreams", MappingProxyType(dict(self.upstreams)))
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))


def _finite_json(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _secret(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value.strip():
        raise ConfigError(f"required secret environment variable {name!r} is missing or empty")
    return value


def load_gateway_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    known_extractors: set[str] | frozenset[str] | None = None,
) -> GatewayConfig:
    """Load and fully bind an operator JSON config without accepting literal secrets."""
    config_path = Path(path)
    environment = os.environ if environ is None else environ
    try:
        raw = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(
            f"cannot load gateway config {config_path}: invalid JSON or unreadable file"
        ) from exc
    try:
        file_config = GatewayFileConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise ConfigError(f"invalid gateway config: {details}") from exc

    extractor_names = (
        _BUILTIN_EXTRACTORS if known_extractors is None else frozenset(known_extractors)
    )
    inbound_keys = tuple(_secret(environment, name) for name in file_config.inbound_api_key_envs)
    if len(set(inbound_keys)) != len(inbound_keys):
        raise ConfigError("inbound API key environment variables must resolve to distinct values")

    upstreams: dict[str, UpstreamConfig] = {}
    for name, item in file_config.upstreams.items():
        if not name.strip():
            raise ConfigError("upstream aliases cannot be empty")
        upstreams[name] = UpstreamConfig(
            name=name,
            base_url=item.base_url,
            api_key=_secret(environment, item.api_key_env),
            connect_timeout_s=float(item.connect_timeout_s),
            read_timeout_s=float(item.read_timeout_s),
            write_timeout_s=float(item.write_timeout_s),
            pool_timeout_s=float(item.pool_timeout_s),
            max_connections=item.max_connections,
            fixed_extra_body=dict(item.fixed_extra_body),
            max_response_bytes=item.max_response_bytes,
            provider=item.provider,
        )

    models: dict[str, ModelRoute] = {}
    for alias, item in file_config.models.items():
        if not alias.strip():
            raise ConfigError("model aliases cannot be empty")
        if item.upstream not in upstreams:
            raise ConfigError(f"model {alias!r} references unknown upstream {item.upstream!r}")
        if item.extractor not in extractor_names:
            raise ConfigError(f"model {alias!r} references unknown extractor {item.extractor!r}")
        plan_path = Path(item.plan_path)
        if not plan_path.is_absolute():
            plan_path = config_path.parent / plan_path
        try:
            deployment, plan = load_deployment_plan(plan_path)
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigError(f"model {alias!r} has an invalid deployment plan") from exc
        strategy_max = deployment.strategy.max_samples
        cap = strategy_max if item.max_samples is None else item.max_samples
        if cap > strategy_max:
            raise ConfigError(
                f"model {alias!r} max_samples exceeds its deployment strategy horizon"
            )
        configured_tokens = item.options.get(
            "max_completion_tokens", item.options.get("max_tokens")
        )
        if configured_tokens is not None and configured_tokens > item.max_completion_tokens:
            raise ConfigError(f"model {alias!r} options exceed max_completion_tokens")
        models[alias] = ModelRoute(
            alias=alias,
            upstream=item.upstream,
            upstream_model=item.upstream_model,
            deployment=deployment,
            plan=plan,
            extractor=item.extractor,
            max_samples=cap,
            max_completion_tokens=item.max_completion_tokens,
            options=dict(item.options),
            allow_client_overrides=item.allow_client_overrides,
            allowed_strategies=frozenset(item.allowed_strategies),
        )

    return GatewayConfig(
        inbound_api_keys=inbound_keys,
        request_timeout_s=float(file_config.request_timeout_s),
        queue_timeout_s=float(file_config.queue_timeout_s),
        max_concurrent_sessions=file_config.max_concurrent_sessions,
        body_read_timeout_s=float(file_config.body_read_timeout_s),
        upstreams=upstreams,
        models=models,
    )
