from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from branchpilot.gateway.providers.anthropic import AnthropicAdapter
from branchpilot.gateway.providers.base import (
    CANONICAL_API_KEY,
    CANONICAL_REQUEST_ID,
    COMPLETION_DETAIL_FIELDS,
    FINISH_REASONS,
    PROMPT_DETAIL_FIELDS,
    CanonicalResponse,
    ProviderAdapter,
    ProviderRequest,
    ProviderRequestError,
    ProviderResponseError,
)
from branchpilot.gateway.providers.bedrock import BedrockAdapter
from branchpilot.gateway.providers.gemini import GeminiAdapter
from branchpilot.gateway.providers.openai import OpenAIAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_PROVIDER = "openai"

# A fixed, in-process registry. Adapters are never resolved from a request-controlled string and
# never imported from a caller-supplied path: an unknown id is a config-load failure.
_ADAPTERS: Mapping[str, ProviderAdapter] = MappingProxyType(
    {
        "anthropic": AnthropicAdapter(),
        "bedrock": BedrockAdapter(),
        "gemini": GeminiAdapter(),
        "openai": OpenAIAdapter(),
    }
)
PROVIDER_IDS: tuple[str, ...] = tuple(sorted(_ADAPTERS))


class UnknownProviderError(LookupError):
    """A provider id that is not in the fixed adapter registry."""


def resolve_adapter(provider: str) -> ProviderAdapter:
    """Return the adapter for a registered provider id, or refuse with the exact fix."""
    adapter = _ADAPTERS.get(provider) if isinstance(provider, str) else None
    if adapter is None:
        raise UnknownProviderError(
            f"unknown upstream provider {provider!r}; fix: set upstreams.<name>.provider to "
            f"one of: {', '.join(PROVIDER_IDS)}"
        )
    return adapter


__all__ = [
    "CANONICAL_API_KEY",
    "CANONICAL_REQUEST_ID",
    "COMPLETION_DETAIL_FIELDS",
    "DEFAULT_PROVIDER",
    "FINISH_REASONS",
    "PROMPT_DETAIL_FIELDS",
    "PROVIDER_IDS",
    "AnthropicAdapter",
    "BedrockAdapter",
    "CanonicalResponse",
    "GeminiAdapter",
    "OpenAIAdapter",
    "ProviderAdapter",
    "ProviderRequest",
    "ProviderRequestError",
    "ProviderResponseError",
    "UnknownProviderError",
    "resolve_adapter",
]
