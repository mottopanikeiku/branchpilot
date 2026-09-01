"""Convert a LiteLLM proxy YAML config into a BranchPilot gateway config and deployment plan.

Requires the optional ``importers`` extra (LiteLLM configs are YAML and core BranchPilot has
no YAML parser). The core path never imports this module.

Usage
-----
::

    from branchpilot.importers.litellm import convert_litellm_config

    result = convert_litellm_config("litellm.yaml", "out/", force=False)
    print(result.config_path, result.plan_path, result.notes_path)
    print(dict(result.mapped))
    for key in result.unmapped:
        print(key.path, key.equivalent or "not supported", key.reason)

:func:`convert_litellm_config` writes exactly three files into ``output_dir``:

``gateway.json``
    Loadable by :func:`branchpilot.gateway.config.load_gateway_config` once the environment
    variables it names are set. Fields LiteLLM did not specify are omitted so BranchPilot's
    own defaults apply.
``gateway-plan.json``
    Loadable by :func:`branchpilot.deployment.load_deployment_plan`. A plan is mandatory for
    every route, and an imported config has no measured calibration, so the plan pins the
    built-in ``fixed`` strategy at ``samples: 1`` -- one upstream call per request, which is
    exactly what LiteLLM was doing. Its ``selection_source.payload_sha256`` is the SHA-256 of
    the source config, recording where the plan came from.
``MIGRATION-NOTES.md``
    Every unmapped LiteLLM key with its path and either the BranchPilot equivalent or an
    explicit "not supported" plus one sentence of reason, together with every choice this
    import made. Unsupported constructs are never silently dropped.

What is mapped
--------------
=========================================== ==========================================
LiteLLM                                     BranchPilot
=========================================== ==========================================
``model_list[].model_name``                 a ``models`` alias
``model_list[].litellm_params.model``       upstream provider and ``upstream_model``
``model_list[].litellm_params.api_base``    ``upstreams.<name>.base_url``
``model_list[].litellm_params.api_key``     ``upstreams.<name>.api_key_env``
``model_list[].litellm_params.max_tokens``  ``models.<alias>.max_completion_tokens``
``router_settings.timeout``                 ``request_timeout_s``, ``read_timeout_s``
``router_settings.max_parallel_requests``   ``max_concurrent_sessions``, ``max_connections``
``general_settings.master_key``             ``inbound_api_key_envs``
=========================================== ==========================================

LiteLLM's ``general_settings`` budgets have no destination: BranchPilot's store records cost
per principal but no spend cap is configured from this file, so budgets are reported as
unmapped rather than translated into something that would not be enforced.

Secrets
-------
An ``api_key: os.environ/FOO`` becomes ``api_key_env: FOO`` -- the *name* travels, never the
value, and nothing is resolved from the environment during conversion. A literal secret in
the source config is refused with a ``fix:`` clause and no output file is written; it is
never copied and never dropped. Error messages carry the key's path, never its value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from branchpilot.gateway.providers import PROVIDER_IDS
from branchpilot.gateway.schemas import MAX_COMPLETION_TOKENS

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "branchpilot.importers.litellm needs the optional 'importers' extra to read LiteLLM's "
        "YAML config; fix: install branchpilot[importers] (pyyaml==6.0.3), or hand-write a "
        "gateway config instead of importing one"
    ) from exc

CONFIG_FILENAME = "gateway.json"
PLAN_FILENAME = "gateway-plan.json"
NOTES_FILENAME = "MIGRATION-NOTES.md"

ENV_REFERENCE_PREFIX = "os.environ/"
DEFAULT_INBOUND_KEY_ENV = "BRANCHPILOT_GATEWAY_KEY"
DEFAULT_MAX_COMPLETION_TOKENS = 1024
DEFAULT_EXTRACTOR = "exact-content"
PLAN_POLICY = "imported-litellm-fixed-1"
MAX_SOURCE_BYTES = 4_194_304

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LIST_INDEX = re.compile(r"\[\d+\]")
# Mirrors of the gateway config's own bounds: a generated config that exceeds one of these
# would be written and then rejected by load_gateway_config, so the import refuses first.
_MAX_ALIAS_CHARS = 128
_MAX_MODELS = 1024
_MAX_UPSTREAMS = 128
_MAX_UPSTREAM_READ_TIMEOUT_S = 3600.0
_MAX_REQUEST_TIMEOUT_S = 86_400.0
_MAX_CONCURRENCY = 10_000

# LiteLLM's "<prefix>/<model>" prefixes that name a provider BranchPilot has an adapter for.
# Anything else is refused rather than routed through an adapter that speaks a different API.
_PROVIDER_BY_PREFIX: Mapping[str, str] = MappingProxyType(
    {
        "anthropic": "anthropic",
        "bedrock": "bedrock",
        "gemini": "gemini",
        "openai": "openai",
    }
)
_DEFAULT_BASE_URL: Mapping[str, str] = MappingProxyType(
    {
        "anthropic": "https://api.anthropic.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "openai": "https://api.openai.com/v1",
    }
)
_DEFAULT_KEY_ENV: Mapping[str, str] = MappingProxyType(
    {
        "anthropic": "ANTHROPIC_API_KEY",
        "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
)

# Curated destinations and reasons for LiteLLM keys BranchPilot does not carry over, keyed by
# path with list indices normalized away. Anything absent here gets _FALLBACK_REASON.
_UNMAPPED_REASONS: Mapping[str, tuple[str | None, str]] = MappingProxyType(
    {
        "environment_variables": (
            None,
            "BranchPilot reads credentials from the process environment and never copies a "
            "value into a config file.",
        ),
        "general_settings.budget_duration": (
            None,
            "BranchPilot enforces no spend cap from this file, so a budget window has nothing "
            "to reset.",
        ),
        "general_settings.database_url": (
            "the store URL passed to open_store()",
            "BranchPilot takes its database URL at runtime, not from the gateway config.",
        ),
        "general_settings.max_budget": (
            None,
            "BranchPilot's store records cost per principal but the gateway config has no "
            "spend-cap field, so a budget here would not be enforced.",
        ),
        "general_settings.max_parallel_requests": (
            "max_concurrent_sessions",
            "router_settings.max_parallel_requests is the mapped source for that limit; set it "
            "there instead of in general_settings.",
        ),
        "litellm_settings.cache": (
            None,
            "the gateway config has no response-cache switch.",
        ),
        "litellm_settings.callbacks": (
            None,
            "the gateway config has no callback hooks.",
        ),
        "litellm_settings.drop_params": (
            None,
            "the gateway config has no request-field-dropping switch; an unsupported request "
            "field is rejected rather than quietly removed.",
        ),
        "litellm_settings.failure_callback": (
            None,
            "the gateway config has no callback hooks.",
        ),
        "litellm_settings.success_callback": (
            None,
            "the gateway config has no callback hooks.",
        ),
        "model_list[].litellm_params.api_version": (
            None,
            "BranchPilot pins each provider's API version in its adapter rather than per route.",
        ),
        "model_list[].litellm_params.rpm": (
            "upstreams.<name>.max_connections (approximate)",
            "BranchPilot bounds concurrency, not requests per minute, so an rpm quota has no "
            "exact translation.",
        ),
        "model_list[].litellm_params.tpm": (
            None,
            "BranchPilot does not throttle on tokens per minute.",
        ),
        "model_list[].model_info.mode": (
            None,
            "BranchPilot routes chat completions only, so a per-route mode has no effect.",
        ),
        "router_settings.allowed_fails": (
            None,
            "the gateway config has no upstream failure budget.",
        ),
        "router_settings.cooldown_time": (
            None,
            "the gateway config has no upstream cooldown.",
        ),
        "router_settings.num_retries": (
            None,
            "the gateway config exposes no automatic retry count.",
        ),
        "router_settings.redis_host": (
            "the BranchPilot store URL",
            "BranchPilot keeps shared state in its own SQLite or Postgres store, not Redis.",
        ),
        "router_settings.redis_port": (
            "the BranchPilot store URL",
            "BranchPilot keeps shared state in its own SQLite or Postgres store, not Redis.",
        ),
        "router_settings.routing_strategy": (
            None,
            "BranchPilot resolves exactly one upstream per model alias, so there is no router "
            "to strategize.",
        ),
        "router_settings.stream_timeout": (
            None,
            "the gateway does not stream responses, so there is no streaming timeout.",
        ),
    }
)
_FALLBACK_REASON = (
    None,
    "BranchPilot has no equivalent for this LiteLLM setting; review it by hand before "
    "cutting over.",
)


class LiteLLMImportError(ValueError):
    """A public, secret-free LiteLLM import failure."""

    def __init__(self, message: str) -> None:
        if "fix:" not in message:
            raise AssertionError("LiteLLM import errors must carry a 'fix:' clause")
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class UnmappedKey:
    """A LiteLLM key that was read and deliberately not carried over.

    ``equivalent`` is ``None`` when BranchPilot has no equivalent at all; the notes render
    that as "not supported". ``reason`` is one sentence.
    """

    path: str
    equivalent: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class ImportResult:
    """The three files written by :func:`convert_litellm_config` and what they cover.

    ``mapped`` counts the source keys carried over per category: ``models`` and ``upstreams``
    count the emitted routes and upstreams, while ``litellm_params``, ``router_settings``, and
    ``general_settings`` count the individual LiteLLM keys consumed from those sections. A
    category that contributed nothing is absent.
    """

    config_path: Path
    plan_path: Path
    notes_path: Path
    mapped: Mapping[str, int]
    unmapped: Sequence[UnmappedKey]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapped", MappingProxyType(dict(self.mapped)))
        object.__setattr__(self, "unmapped", tuple(self.unmapped))


@dataclass(frozen=True, slots=True)
class _Conversion:
    config: dict[str, Any]
    plan: dict[str, Any]
    mapped: dict[str, int]
    unmapped: tuple[UnmappedKey, ...]
    choices: tuple[str, ...]


def convert_litellm_config(
    source: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> ImportResult:
    """Convert LiteLLM proxy config ``source`` into gateway files under ``output_dir``.

    Raises :class:`LiteLLMImportError` -- always with a ``fix:`` clause and never quoting a
    secret -- when the source cannot be represented, when it holds a literal credential, or
    when an output file already exists and ``force`` is false. Every check runs before the
    first byte is written, so a refusal leaves the output directory untouched.
    """

    source_path = Path(source)
    payload = _read_source(source_path)
    document = _parse_document(payload, source_path)
    conversion = _convert(
        document,
        source_path=source_path,
        digest=hashlib.sha256(payload).hexdigest(),
    )

    directory = Path(output_dir)
    config_path = directory / CONFIG_FILENAME
    plan_path = directory / PLAN_FILENAME
    notes_path = directory / NOTES_FILENAME
    targets = (plan_path, config_path, notes_path)
    if not force:
        present = [str(path) for path in targets if os.path.lexists(path)]
        if present:
            raise LiteLLMImportError(
                f"{', '.join(present)} already exist(s); fix: pass force=True to overwrite the "
                "previous import, or point output_dir at an empty directory"
            )

    notes = _render_notes(conversion, source_path=source_path)
    directory.mkdir(parents=True, exist_ok=True)
    _write(plan_path, _json_text(conversion.plan), force=force)
    _write(config_path, _json_text(conversion.config), force=force)
    _write(notes_path, notes, force=force)

    return ImportResult(
        config_path=config_path,
        plan_path=plan_path,
        notes_path=notes_path,
        mapped=conversion.mapped,
        unmapped=conversion.unmapped,
    )


def _read_source(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise LiteLLMImportError(
            f"cannot read LiteLLM config {path}: {exc.strerror}; fix: pass the path of a "
            "readable LiteLLM proxy YAML config"
        ) from exc
    if not payload.strip():
        raise LiteLLMImportError(
            f"LiteLLM config {path} is empty; fix: pass a config that declares a 'model_list'"
        )
    if len(payload) > MAX_SOURCE_BYTES:
        raise LiteLLMImportError(
            f"LiteLLM config {path} is {len(payload)} bytes, above the {MAX_SOURCE_BYTES} byte "
            "limit; fix: split the config, or import the deployments in batches"
        )
    return payload


class _StrictLoader(yaml.SafeLoader):
    """A safe loader that refuses duplicate and non-string mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: Any) -> dict[str, Any]:
    loader.flatten_mapping(node)
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise LiteLLMImportError(
                f"LiteLLM config key {key!r} is not a string; fix: quote every mapping key so "
                "the config reads as YAML text keys"
            )
        if key in mapping:
            raise LiteLLMImportError(
                f"LiteLLM config declares the key {key!r} twice in one mapping; fix: delete the "
                "duplicate so the imported value is unambiguous"
            )
        mapping[key] = loader.construct_object(value_node)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _parse_document(payload: bytes, path: Path) -> Mapping[str, Any]:
    try:
        document = yaml.load(payload.decode("utf-8"), Loader=_StrictLoader)  # noqa: S506
    except UnicodeDecodeError as exc:
        raise LiteLLMImportError(
            f"LiteLLM config {path} is not UTF-8 text; fix: save the config as UTF-8"
        ) from exc
    except yaml.YAMLError as exc:
        raise LiteLLMImportError(
            f"LiteLLM config {path} is not valid YAML: {exc.__class__.__name__}; fix: run the "
            "config through a YAML linter and correct the reported line"
        ) from exc
    if not isinstance(document, Mapping):
        raise LiteLLMImportError(
            f"LiteLLM config {path} is not a YAML mapping; fix: pass a proxy config whose top "
            "level declares 'model_list'"
        )
    return document


def _convert(
    document: Mapping[str, Any],
    *,
    source_path: Path,
    digest: str,
) -> _Conversion:
    consumed: set[str] = set()
    choices: list[str] = []

    router = _mapping_section(document, "router_settings", source_path)
    general = _mapping_section(document, "general_settings", source_path)

    request_timeout_s: float | None = None
    read_timeout_s: float | None = None
    if "timeout" in router:
        request_timeout_s = _seconds(
            router["timeout"], "router_settings.timeout", maximum=_MAX_REQUEST_TIMEOUT_S
        )
        read_timeout_s = min(request_timeout_s, _MAX_UPSTREAM_READ_TIMEOUT_S)
        consumed.add("router_settings.timeout")
        if read_timeout_s < request_timeout_s:
            choices.append(
                f"router_settings.timeout is {request_timeout_s:g}s, above the "
                f"{_MAX_UPSTREAM_READ_TIMEOUT_S:g}s ceiling on an upstream read timeout, so "
                f"every read_timeout_s was clamped to {read_timeout_s:g}s."
            )

    concurrency: int | None = None
    if "max_parallel_requests" in router:
        concurrency = _bounded_int(
            router["max_parallel_requests"],
            "router_settings.max_parallel_requests",
            minimum=1,
            maximum=_MAX_CONCURRENCY,
        )
        consumed.add("router_settings.max_parallel_requests")

    if "master_key" in general:
        inbound = [_env_reference(general["master_key"], "general_settings.master_key")]
        consumed.add("general_settings.master_key")
    else:
        inbound = [DEFAULT_INBOUND_KEY_ENV]
        choices.append(
            "The source config declares no general_settings.master_key, so "
            f"inbound_api_key_envs names {DEFAULT_INBOUND_KEY_ENV}; set that variable to the "
            "key your clients will send before starting the gateway."
        )

    upstreams, models, model_count, param_count = _routes(
        document,
        source_path=source_path,
        consumed=consumed,
        choices=choices,
        read_timeout_s=read_timeout_s,
        max_connections=concurrency,
    )

    config: dict[str, Any] = {"inbound_api_key_envs": inbound}
    if request_timeout_s is not None:
        config["request_timeout_s"] = request_timeout_s
    if concurrency is not None:
        config["max_concurrent_sessions"] = concurrency
    config["upstreams"] = upstreams
    config["models"] = models

    mapped = {
        "models": model_count,
        "upstreams": len(upstreams),
        "litellm_params": param_count,
        "router_settings": sum(1 for path in consumed if path.startswith("router_settings.")),
        "general_settings": sum(1 for path in consumed if path.startswith("general_settings.")),
    }
    choices.append(
        "The deployment plan pins the built-in 'fixed' strategy at samples: 1, so every "
        "request makes exactly one upstream call, as LiteLLM did. Nothing about latency or "
        "spend changes on cutover; run a calibration and replace gateway-plan.json to start "
        "sampling adaptively."
    )
    choices.append(
        f"The plan reports no measured accuracy or token cost: its selection_source names "
        f"policy {PLAN_POLICY!r} and carries the SHA-256 of {source_path.name} instead of a "
        "benchmark payload, so it is traceable to this import and to no experiment."
    )
    choices.append(
        f"Every route uses the {DEFAULT_EXTRACTOR!r} extractor, which treats the whole "
        "completion as the answer. That is the only choice that cannot reject a response; "
        "pick a stricter extractor per route once you sample more than once."
    )

    return _Conversion(
        config=config,
        plan=_plan(digest),
        mapped={name: count for name, count in mapped.items() if count},
        unmapped=_unmapped(document, consumed),
        choices=tuple(dict.fromkeys(choices)),
    )


def _routes(
    document: Mapping[str, Any],
    *,
    source_path: Path,
    consumed: set[str],
    choices: list[str],
    read_timeout_s: float | None,
    max_connections: int | None,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    entries = document.get("model_list")
    if not isinstance(entries, list) or not entries:
        raise LiteLLMImportError(
            f"LiteLLM config {source_path} declares no non-empty 'model_list'; fix: import a "
            "proxy config that lists at least one deployment"
        )
    if len(entries) > _MAX_MODELS:
        raise LiteLLMImportError(
            f"LiteLLM config {source_path} lists {len(entries)} deployments, above the "
            f"{_MAX_MODELS} route limit; fix: split the config across gateway instances"
        )

    upstreams: dict[str, Any] = {}
    models: dict[str, Any] = {}
    by_target: dict[tuple[str, str, str], str] = {}
    param_count = 0

    for index, entry in enumerate(entries):
        base = f"model_list[{index}]"
        if not isinstance(entry, Mapping):
            raise LiteLLMImportError(
                f"{base} is not a mapping; fix: give every model_list entry a 'model_name' and "
                "a 'litellm_params' block"
            )
        alias = _alias(entry.get("model_name"), f"{base}.model_name")
        if alias in models:
            raise LiteLLMImportError(
                f"{base}.model_name repeats the alias {alias!r}, which LiteLLM load-balances "
                "across deployments; fix: BranchPilot resolves exactly one upstream per alias, "
                "so give each deployment a distinct model_name and route clients explicitly"
            )
        consumed.add(f"{base}.model_name")

        params = entry.get("litellm_params")
        if not isinstance(params, Mapping) or not params:
            raise LiteLLMImportError(
                f"{base}.litellm_params is missing or empty; fix: declare at least "
                "'model: <provider>/<model>' for every deployment"
            )

        provider, upstream_model = _split_model(params.get("model"), f"{base}.litellm_params")
        consumed.add(f"{base}.litellm_params.model")
        if "/" not in str(params["model"]):
            choices.append(
                f"{base}.litellm_params.model carries no provider prefix, so this route was "
                f"imported as provider {provider!r}; prefix the model in LiteLLM "
                f"(model: {provider}/{upstream_model}) if that is wrong."
            )
        param_count += 1

        if "api_base" in params:
            base_url = _base_url(params["api_base"], f"{base}.litellm_params.api_base")
            consumed.add(f"{base}.litellm_params.api_base")
            param_count += 1
        else:
            base_url = _DEFAULT_BASE_URL.get(provider, "")
            if not base_url:
                raise LiteLLMImportError(
                    f"{base}.litellm_params declares no api_base and provider {provider!r} has "
                    "no single public endpoint; fix: set api_base to the regional endpoint, for "
                    "example https://bedrock-runtime.us-east-1.amazonaws.com"
                )
            choices.append(
                f"{base}.litellm_params declares no api_base, so upstream base_url defaults to "
                f"{base_url} for provider {provider!r}."
            )

        if "api_key" in params:
            api_key_env = _env_reference(params["api_key"], f"{base}.litellm_params.api_key")
            consumed.add(f"{base}.litellm_params.api_key")
            param_count += 1
        else:
            api_key_env = _DEFAULT_KEY_ENV[provider]
            choices.append(
                f"{base}.litellm_params declares no api_key, so the upstream reads "
                f"{api_key_env}, the conventional variable for provider {provider!r}."
            )

        if "max_tokens" in params:
            max_completion_tokens = _bounded_int(
                params["max_tokens"],
                f"{base}.litellm_params.max_tokens",
                minimum=1,
                maximum=MAX_COMPLETION_TOKENS,
            )
            consumed.add(f"{base}.litellm_params.max_tokens")
            param_count += 1
        else:
            max_completion_tokens = DEFAULT_MAX_COMPLETION_TOKENS
            choices.append(
                f"{base}.litellm_params declares no max_tokens, so max_completion_tokens "
                f"defaults to {DEFAULT_MAX_COMPLETION_TOKENS}; raise it per route if your "
                "clients ask for longer completions."
            )

        target = (provider, base_url, api_key_env)
        name = by_target.get(target)
        if name is None:
            if len(upstreams) >= _MAX_UPSTREAMS:
                raise LiteLLMImportError(
                    f"{base}.litellm_params needs a distinct upstream beyond the "
                    f"{_MAX_UPSTREAMS} a gateway config can hold; fix: split the deployments "
                    "across gateway instances, or point them at a shared api_base"
                )
            name = _upstream_name(provider, upstreams)
            by_target[target] = name
            upstream: dict[str, Any] = {
                "base_url": base_url,
                "api_key_env": api_key_env,
                "provider": provider,
            }
            if read_timeout_s is not None:
                upstream["read_timeout_s"] = read_timeout_s
            if max_connections is not None:
                upstream["max_connections"] = max_connections
            upstreams[name] = upstream

        models[alias] = {
            "upstream": name,
            "upstream_model": upstream_model,
            "plan_path": PLAN_FILENAME,
            "extractor": DEFAULT_EXTRACTOR,
            "max_samples": 1,
            "max_completion_tokens": max_completion_tokens,
        }

    return upstreams, models, len(entries), param_count


def _plan(digest: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_source": {"benchmark_schema_version": 2, "payload_sha256": digest},
        "family": "fixed",
        "policy": PLAN_POLICY,
        "strategy_spec": {"type": "fixed", "samples": 1, "max_samples": 1},
        "expected_accuracy": 0.0,
        "accuracy_interval": {"lower": 0.0, "upper": 0.0},
        "expected_samples": 1.0,
        "samples_interval": {"lower": 1.0, "upper": 1.0},
        "expected_tokens": 0.0,
        "tokens_interval": {"lower": 0.0, "upper": 0.0},
        "requested_sample_budget": 1.0,
        "conservative": True,
        "budget_satisfied": True,
    }


def _mapping_section(
    document: Mapping[str, Any], name: str, source_path: Path
) -> Mapping[str, Any]:
    value = document.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LiteLLMImportError(
            f"LiteLLM config {source_path} declares {name!r} as "
            f"{type(value).__name__}; fix: make {name!r} a mapping of settings, or delete it"
        )
    return value


def _alias(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteLLMImportError(
            f"{path} must be a non-empty string; fix: name the alias your clients will send as "
            "the 'model' field, for example 'gpt-4o'"
        )
    alias = value.strip()
    if alias != value:
        raise LiteLLMImportError(f"{path} has surrounding whitespace; fix: set it to {alias!r}")
    if "*" in alias:
        raise LiteLLMImportError(
            f"{path} is the wildcard alias {alias!r}, which LiteLLM expands to any model at "
            "request time; fix: BranchPilot routes only the aliases it declares, so give each "
            "model your clients send its own model_list entry"
        )
    if len(alias) > _MAX_ALIAS_CHARS:
        raise LiteLLMImportError(
            f"{path} is {len(alias)} characters, above the {_MAX_ALIAS_CHARS} limit; fix: "
            "shorten the alias"
        )
    if any(character.isspace() or not character.isprintable() for character in alias):
        raise LiteLLMImportError(
            f"{path} contains whitespace or control characters; fix: use a slug such as "
            "'gpt-4o' or 'claude-sonnet'"
        )
    return alias


def _split_model(value: Any, path: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise LiteLLMImportError(
            f"{path}.model must be a non-empty string; fix: set it to '<provider>/<model>', for "
            "example 'openai/gpt-4o'"
        )
    text = value.strip()
    prefix, separator, remainder = text.partition("/")
    if not separator:
        provider, model = "openai", text
    else:
        provider = _PROVIDER_BY_PREFIX.get(prefix, "")
        if not provider:
            raise LiteLLMImportError(
                f"{path}.model names the LiteLLM provider {prefix!r}, which BranchPilot has no "
                f"adapter for; fix: route this deployment through one of: "
                f"{', '.join(PROVIDER_IDS)}, or delete the entry and keep serving it elsewhere"
            )
        model = remainder.strip()
    if not model:
        raise LiteLLMImportError(
            f"{path}.model names a provider but no model; fix: set it to "
            f"'{provider}/<model>', for example '{provider}/gpt-4o'"
        )
    if len(model) > 256:
        raise LiteLLMImportError(
            f"{path}.model names a {len(model)} character model, above the 256 limit; fix: use "
            "the provider's own model id"
        )
    if "*" in model:
        raise LiteLLMImportError(
            f"{path}.model is the wildcard pattern {model!r}, which LiteLLM resolves per "
            f"request; fix: name the exact upstream model, for example '{provider}/gpt-4o'"
        )
    return provider, model


def _base_url(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteLLMImportError(
            f"{path} must be a non-empty URL; fix: set it to the upstream base URL, for example "
            "https://api.openai.com/v1"
        )
    text = value.strip()
    if len(text) > 2048:
        raise LiteLLMImportError(
            f"{path} is {len(text)} characters, above the 2048 limit; fix: shorten the URL"
        )
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LiteLLMImportError(
            f"{path} is not an absolute HTTP(S) URL; fix: set it to a full URL such as "
            "https://api.openai.com/v1"
        )
    if parsed.username is not None or parsed.password is not None:
        raise LiteLLMImportError(
            f"{path} carries credentials in the URL, which would put a secret in the generated "
            "config; fix: drop the userinfo from the URL and pass the credential through "
            "api_key: os.environ/NAME instead"
        )
    if parsed.query or parsed.fragment:
        raise LiteLLMImportError(
            f"{path} carries a query or fragment; fix: set it to the bare base URL, for example "
            "https://api.openai.com/v1"
        )
    return text.rstrip("/")


def _env_reference(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteLLMImportError(
            f"{path} must be a string environment reference; fix: set it to "
            f"'{ENV_REFERENCE_PREFIX}NAME' and export NAME in the deployment environment"
        )
    text = value.strip()
    if not text.startswith(ENV_REFERENCE_PREFIX):
        raise LiteLLMImportError(
            f"{path} holds a literal secret, and BranchPilot never copies a credential into a "
            f"generated file; fix: move the value into an environment variable and set "
            f"{path.rpartition('.')[2]}: {ENV_REFERENCE_PREFIX}NAME in the LiteLLM config, then "
            "re-run the import"
        )
    name = text[len(ENV_REFERENCE_PREFIX) :].strip()
    if not _ENV_NAME.fullmatch(name) or len(name) > 128:
        raise LiteLLMImportError(
            f"{path} does not name a usable environment variable; fix: use "
            f"'{ENV_REFERENCE_PREFIX}NAME' with NAME matching [A-Za-z_][A-Za-z0-9_]*, for "
            f"example '{ENV_REFERENCE_PREFIX}OPENAI_API_KEY'"
        )
    return name


def _seconds(value: Any, path: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiteLLMImportError(
            f"{path} must be a number of seconds; fix: set it to a positive number such as 60"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0 or number > maximum:
        raise LiteLLMImportError(
            f"{path} must be a finite number of seconds in (0, {maximum:g}]; fix: set it to a "
            "positive timeout such as 60"
        )
    return number


def _bounded_int(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiteLLMImportError(
            f"{path} must be an integer; fix: set it to a whole number in [{minimum}, {maximum}]"
        )
    if value < minimum or value > maximum:
        raise LiteLLMImportError(
            f"{path} is {value}, outside [{minimum}, {maximum}]; fix: set it to a value in "
            "that range"
        )
    return value


def _upstream_name(provider: str, taken: Mapping[str, Any]) -> str:
    if provider not in taken:
        return provider
    suffix = 2
    while f"{provider}-{suffix}" in taken:
        suffix += 1
    return f"{provider}-{suffix}"


def _leaf_paths(value: Any, prefix: str) -> Iterator[str]:
    """Yield one path per configured leaf; an empty mapping or list configures nothing."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaf_paths(item, child)
        return
    if isinstance(value, list):
        if not value:
            return
        if all(isinstance(item, Mapping) for item in value):
            for index, item in enumerate(value):
                yield from _leaf_paths(item, f"{prefix}[{index}]")
            return
    yield prefix


def _reason(path: str) -> tuple[str | None, str]:
    """Resolve the curated destination for ``path``, falling back to its nearest section."""

    normalized = _LIST_INDEX.sub("[]", path)
    while normalized:
        curated = _UNMAPPED_REASONS.get(normalized)
        if curated is not None:
            return curated
        normalized = normalized.rpartition(".")[0]
    return _FALLBACK_REASON


def _unmapped(document: Mapping[str, Any], consumed: set[str]) -> tuple[UnmappedKey, ...]:
    keys: list[UnmappedKey] = []
    for path in _leaf_paths(document, ""):
        if path in consumed:
            continue
        equivalent, reason = _reason(path)
        keys.append(UnmappedKey(path=path, equivalent=equivalent, reason=reason))
    return tuple(keys)


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _render_notes(conversion: _Conversion, *, source_path: Path) -> str:
    lines = [
        "# LiteLLM import notes",
        "",
        f"`{_cell(CONFIG_FILENAME)}` and `{_cell(PLAN_FILENAME)}` in this directory were "
        f"generated from `{_cell(source_path.name)}` by `branchpilot.importers.litellm`.",
        "No credential was resolved or copied: every key travels as the *name* of an "
        "environment variable, which the gateway reads at startup.",
        "",
        "## What was mapped",
        "",
        "| Category | LiteLLM keys carried over |",
        "| --- | --- |",
    ]
    for name, count in conversion.mapped.items():
        lines.append(f"| {_cell(name)} | {count} |")
    lines += ["", "## Choices this import made", ""]
    lines += [f"- {_cell(choice)}" for choice in conversion.choices]
    lines += ["", "## Unmapped keys", ""]
    if not conversion.unmapped:
        lines.append("Every key in the source config was carried over.")
    else:
        lines += [
            "Every key below was read and deliberately not carried over. Nothing was dropped "
            "silently.",
            "",
            "| LiteLLM path | BranchPilot equivalent | Why |",
            "| --- | --- | --- |",
        ]
        for key in conversion.unmapped:
            equivalent = "not supported" if key.equivalent is None else f"`{_cell(key.equivalent)}`"
            lines.append(f"| `{_cell(key.path)}` | {equivalent} | {_cell(key.reason)} |")
    lines.append("")
    return "\n".join(lines)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write(path: Path, text: str, *, force: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if force else os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise LiteLLMImportError(
            f"{path} already exists; fix: pass force=True to overwrite the previous import, or "
            "point output_dir at an empty directory"
        ) from exc
    except OSError as exc:
        raise LiteLLMImportError(
            f"cannot write {path}: {exc.strerror}; fix: point output_dir at a writable directory"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


__all__ = [
    "CONFIG_FILENAME",
    "NOTES_FILENAME",
    "PLAN_FILENAME",
    "ImportResult",
    "LiteLLMImportError",
    "UnmappedKey",
    "convert_litellm_config",
]
