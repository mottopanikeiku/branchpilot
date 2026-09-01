"""Importers that translate another gateway's configuration into BranchPilot's.

Nothing in this package is reachable from the core path. Every importer names the optional
extra it needs and raises :class:`ImportError` with a ``fix:`` clause when that extra is
absent, so importing :mod:`branchpilot.importers` itself never pulls a third-party parser::

    from branchpilot.importers.litellm import convert_litellm_config

Available importers:

* :mod:`branchpilot.importers.litellm` -- LiteLLM proxy YAML. Needs the ``importers`` extra.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
