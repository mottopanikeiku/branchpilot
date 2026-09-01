from __future__ import annotations

import hashlib
import json
import re
import textwrap
from pathlib import Path

import pytest

from branchpilot.deployment import load_deployment_plan
from branchpilot.gateway.config import load_gateway_config
from branchpilot.importers.litellm import (
    CONFIG_FILENAME,
    NOTES_FILENAME,
    PLAN_FILENAME,
    ImportResult,
    LiteLLMImportError,
    convert_litellm_config,
)

FIXTURES = Path(__file__).parent / "fixtures" / "importers"
PROXY_CONFIG = FIXTURES / "litellm-proxy.yaml"
LITERAL_SECRET_CONFIG = FIXTURES / "litellm-literal-secret.yaml"

# Every key the representative fixture declares that BranchPilot does not carry over. The
# notes must cover exactly this set: nothing missing, nothing invented.
FIXTURE_UNMAPPED = frozenset(
    {
        "model_list[0].litellm_params.rpm",
        "model_list[2].model_info.mode",
        "router_settings.num_retries",
        "router_settings.cooldown_time",
        "router_settings.routing_strategy",
        "general_settings.max_budget",
        "general_settings.budget_duration",
        "litellm_settings.drop_params",
        "litellm_settings.success_callback",
    }
)

# Values the gateway will read at runtime. Conversion must never resolve them, so every one
# of these doubles as a sentinel: none may appear in a generated file.
FIXTURE_ENVIRONMENT = {
    "LITELLM_MASTER_KEY": "sk-sentinel-inbound-9f2a",
    "OPENAI_API_KEY": "sk-sentinel-openai-4c71",
    "ANTHROPIC_API_KEY": "sk-sentinel-anthropic-0b3e",
}

_NOTES_PATH_CELL = re.compile(r"^\| `([^`]+)` \|")


def _source(tmp_path: Path, body: str) -> Path:
    """Write ``body`` as a LiteLLM config and return its path."""

    path = tmp_path / "litellm.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _minimal(extra: str = "") -> str:
    """One openai deployment, plus ``extra`` appended as further top-level sections."""

    return (
        "model_list:\n"
        "  - model_name: gpt-4o\n"
        "    litellm_params:\n"
        "      model: openai/gpt-4o\n"
        "      api_key: os.environ/OPENAI_API_KEY\n" + textwrap.dedent(extra)
    )


def _notes_unmapped(notes: str) -> set[str]:
    """The LiteLLM paths listed in the notes' unmapped table."""

    section = notes.partition("## Unmapped keys")[2]
    matches = (_NOTES_PATH_CELL.match(line) for line in section.splitlines())
    return {match.group(1) for match in matches if match is not None}


@pytest.fixture
def imported(tmp_path: Path) -> ImportResult:
    return convert_litellm_config(PROXY_CONFIG, tmp_path / "out")


def test_result_names_the_three_generated_files(imported: ImportResult, tmp_path: Path) -> None:
    directory = tmp_path / "out"
    assert imported.config_path == directory / CONFIG_FILENAME
    assert imported.plan_path == directory / PLAN_FILENAME
    assert imported.notes_path == directory / NOTES_FILENAME
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        (CONFIG_FILENAME, PLAN_FILENAME, NOTES_FILENAME)
    )


def test_generated_config_loads_with_fake_environment(imported: ImportResult) -> None:
    config = load_gateway_config(imported.config_path, environ=FIXTURE_ENVIRONMENT)

    assert sorted(config.models) == ["claude-sonnet", "gpt-4o", "gpt-4o-mini"]
    assert sorted(config.upstreams) == ["anthropic", "openai"]
    assert config.models["gpt-4o"].upstream_model == "gpt-4o"
    assert config.models["gpt-4o"].max_completion_tokens == 2048
    assert config.models["claude-sonnet"].upstream == "anthropic"
    assert config.request_timeout_s == 120.0
    assert config.max_concurrent_sessions == 16
    assert config.upstreams["openai"].max_connections == 16
    assert config.upstreams["openai"].read_timeout_s == 120.0


def test_deployments_sharing_a_target_collapse_into_one_upstream(imported: ImportResult) -> None:
    payload = json.loads(imported.config_path.read_text(encoding="utf-8"))

    assert set(payload["upstreams"]) == {"openai", "anthropic"}
    assert payload["models"]["gpt-4o"]["upstream"] == "openai"
    assert payload["models"]["gpt-4o-mini"]["upstream"] == "openai"
    assert imported.mapped["upstreams"] == 2
    assert imported.mapped["models"] == 3


def test_generated_plan_loads_and_pins_one_sample(imported: ImportResult) -> None:
    deployment, plan = load_deployment_plan(imported.plan_path)

    assert deployment.spec == {"type": "fixed", "samples": 1, "max_samples": 1}
    assert deployment.strategy.max_samples == 1
    assert plan.family == "fixed"
    assert plan.conservative is True
    assert plan.expected_samples == 1.0


def test_plan_records_the_source_digest(imported: ImportResult) -> None:
    payload = json.loads(imported.plan_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PROXY_CONFIG.read_bytes()).hexdigest()

    assert payload["selection_source"]["payload_sha256"] == digest


def test_env_references_travel_as_names_never_values(imported: ImportResult) -> None:
    payload = json.loads(imported.config_path.read_text(encoding="utf-8"))

    assert payload["inbound_api_key_envs"] == ["LITELLM_MASTER_KEY"]
    assert payload["upstreams"]["openai"]["api_key_env"] == "OPENAI_API_KEY"
    assert payload["upstreams"]["anthropic"]["api_key_env"] == "ANTHROPIC_API_KEY"


def test_no_secret_value_reaches_a_generated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in FIXTURE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    result = convert_litellm_config(PROXY_CONFIG, tmp_path / "out")

    for path in (result.config_path, result.plan_path, result.notes_path):
        text = path.read_text(encoding="utf-8")
        for sentinel in FIXTURE_ENVIRONMENT.values():
            assert sentinel not in text, f"{path.name} leaked {sentinel!r}"
        assert "sk-" not in text


def test_notes_cover_the_unmapped_set_exactly(imported: ImportResult) -> None:
    notes = imported.notes_path.read_text(encoding="utf-8")

    assert {key.path for key in imported.unmapped} == FIXTURE_UNMAPPED
    assert _notes_unmapped(notes) == FIXTURE_UNMAPPED


def test_notes_give_every_unmapped_key_a_destination_or_a_reason(imported: ImportResult) -> None:
    notes = imported.notes_path.read_text(encoding="utf-8")

    for key in imported.unmapped:
        assert key.reason.strip()
        if key.equivalent is None:
            assert f"| `{key.path}` | not supported |" in notes
        else:
            assert f"| `{key.path}` | `{key.equivalent}` |" in notes


def test_notes_record_the_fixed_strategy_choice(imported: ImportResult) -> None:
    notes = imported.notes_path.read_text(encoding="utf-8")

    assert "## Choices this import made" in notes
    assert "'fixed' strategy at samples: 1" in notes


def test_literal_api_key_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "out"

    with pytest.raises(LiteLLMImportError) as error:
        convert_litellm_config(LITERAL_SECRET_CONFIG, output)

    message = str(error.value)
    assert "fix:" in message
    assert "os.environ/NAME" in message
    assert "sk-real-looking-value" not in message
    assert not output.exists()


def test_literal_master_key_is_refused(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        _minimal(
            """
            general_settings:
              master_key: sk-1234567890
            """
        ),
    )

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "general_settings.master_key" in str(error.value)
    assert "sk-1234567890" not in str(error.value)


def test_credentials_in_api_base_are_refused(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        model_list:
          - model_name: gpt-4o
            litellm_params:
              model: openai/gpt-4o
              api_base: https://user:hunter2@proxy.internal/v1
        """,
    )

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "hunter2" not in str(error.value)
    assert not (tmp_path / "out").exists()


def test_existing_output_is_refused_without_force(tmp_path: Path) -> None:
    output = tmp_path / "out"
    convert_litellm_config(PROXY_CONFIG, output)
    stamp = output / NOTES_FILENAME
    stamp.write_text("hand written\n", encoding="utf-8")

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(PROXY_CONFIG, output)

    assert "force=True" in str(error.value)
    assert stamp.read_text(encoding="utf-8") == "hand written\n"


def test_force_overwrites_a_previous_import(tmp_path: Path) -> None:
    output = tmp_path / "out"
    convert_litellm_config(PROXY_CONFIG, output)
    (output / NOTES_FILENAME).write_text("hand written\n", encoding="utf-8")

    result = convert_litellm_config(PROXY_CONFIG, output, force=True)

    assert result.notes_path.read_text(encoding="utf-8").startswith("# LiteLLM import notes")
    load_gateway_config(result.config_path, environ=FIXTURE_ENVIRONMENT)


def test_a_single_pre_existing_file_blocks_the_whole_import(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / PLAN_FILENAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(LiteLLMImportError, match="fix:"):
        convert_litellm_config(PROXY_CONFIG, output)

    assert sorted(path.name for path in output.iterdir()) == [PLAN_FILENAME]


def test_missing_master_key_falls_back_to_a_named_variable(tmp_path: Path) -> None:
    result = convert_litellm_config(_source(tmp_path, _minimal()), tmp_path / "out")
    payload = json.loads(result.config_path.read_text(encoding="utf-8"))

    assert payload["inbound_api_key_envs"] == ["BRANCHPILOT_GATEWAY_KEY"]
    assert "BRANCHPILOT_GATEWAY_KEY" in result.notes_path.read_text(encoding="utf-8")
    load_gateway_config(
        result.config_path,
        environ={"BRANCHPILOT_GATEWAY_KEY": "inbound", "OPENAI_API_KEY": "upstream"},
    )


def test_router_timeout_above_the_read_ceiling_is_clamped(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        _minimal(
            """
            router_settings:
              timeout: 7200
            """
        ),
    )

    result = convert_litellm_config(source, tmp_path / "out")
    payload = json.loads(result.config_path.read_text(encoding="utf-8"))

    assert payload["request_timeout_s"] == 7200.0
    assert payload["upstreams"]["openai"]["read_timeout_s"] == 3600.0
    assert "clamped to 3600s" in result.notes_path.read_text(encoding="utf-8")
    load_gateway_config(
        result.config_path,
        environ={"BRANCHPILOT_GATEWAY_KEY": "inbound", "OPENAI_API_KEY": "upstream"},
    )


def test_empty_sections_are_not_reported_as_unmapped_keys(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        _minimal(
            """
            router_settings: {}
            litellm_settings:
              callbacks: []
            """
        ),
    )

    result = convert_litellm_config(source, tmp_path / "out")

    assert result.unmapped == ()
    assert "Every key in the source config was carried over." in result.notes_path.read_text(
        encoding="utf-8"
    )


def test_a_nested_unmapped_section_inherits_its_curated_reason(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        _minimal(
            """
            environment_variables:
              OPENAI_API_KEY: sk-not-copied-anywhere
            """
        ),
    )

    result = convert_litellm_config(source, tmp_path / "out")

    (key,) = result.unmapped
    assert key.path == "environment_variables.OPENAI_API_KEY"
    assert "never copies a value into a config file" in key.reason
    assert "sk-not-copied-anywhere" not in result.notes_path.read_text(encoding="utf-8")


def test_wildcard_alias_is_refused(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        model_list:
          - model_name: "openai/*"
            litellm_params:
              model: "openai/*"
        """,
    )

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "wildcard" in str(error.value)


def test_repeated_alias_is_refused(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        model_list:
          - model_name: gpt-4o
            litellm_params: {model: openai/gpt-4o}
          - model_name: gpt-4o
            litellm_params: {model: openai/gpt-4o-mini}
        """,
    )

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "model_list[1].model_name" in str(error.value)


def test_provider_without_an_adapter_is_refused(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        model_list:
          - model_name: gpt-4o
            litellm_params:
              model: azure/gpt-4o
              api_base: https://example.openai.azure.com
        """,
    )

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "azure" in str(error.value)


def test_duplicate_yaml_key_is_refused(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        model_list:
          - model_name: gpt-4o
            litellm_params:
              model: openai/gpt-4o
              model: openai/gpt-4o-mini
        """,
    )

    with pytest.raises(LiteLLMImportError, match="fix:"):
        convert_litellm_config(source, tmp_path / "out")


def test_config_without_a_model_list_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path, "general_settings:\n  master_key: os.environ/KEY\n")

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "model_list" in str(error.value)


def test_more_distinct_upstreams_than_a_config_can_hold_is_refused(tmp_path: Path) -> None:
    entries = "".join(
        f"  - model_name: m{index}\n"
        f"    litellm_params:\n"
        f"      model: openai/gpt-4o\n"
        f"      api_base: https://host{index}.example.com/v1\n"
        f"      api_key: os.environ/KEY_{index}\n"
        for index in range(129)
    )
    source = tmp_path / "litellm.yaml"
    source.write_text(f"model_list:\n{entries}", encoding="utf-8")

    with pytest.raises(LiteLLMImportError, match="fix:") as error:
        convert_litellm_config(source, tmp_path / "out")

    assert "128" in str(error.value)
    assert not (tmp_path / "out").exists()
