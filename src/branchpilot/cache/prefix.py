"""Prefix-cache analysis over a text-free :class:`RequestRecord` stream.

What this module measures
------------------------
Records are clustered by ``system_prefix_hash``. Every cluster reports how many requests
share the prefix, which ``(provider, model)`` pairs used it, how much prefix character mass
was sent in total and how much of that was repeated, and -- where the source log reported
``cached_prompt_tokens`` -- the *realized* cache behavior the provider actually delivered.

Nothing here is estimated from prompt length. :class:`RequestRecord` carries no prefix
token count, and a character count is not a token count. A projection therefore requires a
token count that is either supplied by the caller or measured from the log's own
cache-read counts; otherwise the plan says ``needs_token_counts`` and refuses to guess.

A projection is counted only over *billed* requests -- those whose log reports a positive
``prompt_tokens``. A refused or rate-limited request bills no prompt, so it can neither write
nor read a prefix cache, and counting it would project a saving that was never on the table.
Request status is deliberately not the gate here: a log with no status field ingests as
``unknown``, while billed prompt tokens are a number every log reports.

When the count comes from the log's own cache reads (``token_source="measured"``) it is the
smallest positive ``cached_prompt_tokens`` observed. A cache read covers the shared prefix and
can cover more, so that count is an upper bound on the prefix and the projection built from it
is an upper bound too. Pass ``prefix_tokens`` when an exact count is known.

The volatile-prefix signal
--------------------------
:func:`analyze_prefixes` also reports the failure this module exists to catch. When a
``(provider, model)`` keeps sending a system prefix of a *stable character length* while
its ``system_prefix_hash`` churns from request to request, the prefix contains something
per-request -- an injected timestamp, a uuid, a per-user string -- and the provider's
prefix cache silently never hits. No error is returned by any provider for this; the
operator finds out from the invoice. :class:`VolatilePrefix` reports it per model with a
churn rate, where ``0`` is a perfectly stable prefix and ``1`` is a fresh prefix every
single request.

Provider facts
--------------
Minimum cacheable prefix lengths, cache-write premiums, and explicit breakpoint budgets
are documented provider behavior, not our estimates. Each value below is sourced, and an
undocumented provider yields ``None`` rather than a guess:

* Anthropic -- per-model minimums (512 to 4,096 tokens), 4 explicit breakpoints, and a
  5-minute cache write at 1.25x base input.
  https://platform.claude.com/docs/en/build-with-claude/prompt-caching
* OpenAI -- 2,048 visible input tokens for models older than GPT-5.6 (every model in the
  packaged price book), implicit breakpoints only, and no cache-write charge on those
  models. https://developers.openai.com/api/docs/guides/prompt-caching
* Gemini -- implicit caching only, 2,048 tokens on 2.5 models and 4,096 on 3.x, no
  breakpoints and no write charge. https://ai.google.dev/gemini-api/docs/caching

A ``breakpoints`` count of ``0`` on a cacheable plan is not a downgrade: it is the correct
recommendation for a provider whose caching is implicit, where the lever is prefix
*stability* rather than marker placement.

Money
-----
Every amount is a :class:`decimal.Decimal` produced by :meth:`PriceBook.price`, so this
module adds no arithmetic of its own to the cost path. An unpriced ``(provider, model)``
never raises out of an analysis: the plan reports ``unpriced_model`` and names the fix.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext

from branchpilot.ingest.formats import RequestRecord
from branchpilot.pricing import PriceBook, UnknownModelError

__all__ = [
    "PLAN_STATUSES",
    "PROVIDER_CACHE_DOCS",
    "TOKEN_SOURCES",
    "VOLATILE_CHURN_THRESHOLD",
    "VOLATILE_MIN_PREFIX_CHARS",
    "VOLATILE_MIN_REQUESTS",
    "CachePlan",
    "ModelPrefixUsage",
    "PrefixAnalysis",
    "PrefixAnalysisError",
    "PrefixCluster",
    "VolatilePrefix",
    "analyze_prefixes",
    "cache_write_multiplier",
    "max_cache_breakpoints",
    "minimum_cacheable_tokens",
    "plan_cache",
]

PLAN_STATUSES = frozenset(
    {
        "ok",
        "no_prefix",
        "no_repeats",
        "needs_token_counts",
        "unpriced_model",
        "below_provider_minimum",
        "below_break_even",
        "already_cached",
    }
)
"""Closed vocabulary for :attr:`CachePlan.status`. Only ``ok`` carries a live projection."""

TOKEN_SOURCES = frozenset({"supplied", "measured", "unavailable"})
"""Where :attr:`CachePlan.prefix_tokens` came from. Never ``prompt_tokens``."""

VOLATILE_CHURN_THRESHOLD = Decimal("0.5")
"""Churn at or above this fraction is reported as a volatile prefix."""

VOLATILE_MIN_REQUESTS = 5
"""Requests a ``(model, prefix length)`` group needs before churn is meaningful."""

VOLATILE_MIN_PREFIX_CHARS = 1
"""A request with no system prefix has nothing to churn, so it is never reported."""

PROVIDER_CACHE_DOCS: Mapping[str, str] = {
    "anthropic": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
    "gemini": "https://ai.google.dev/gemini-api/docs/caching",
    "openai": "https://developers.openai.com/api/docs/guides/prompt-caching",
}
"""The page each provider fact below is read from, quoted in operator-facing fixes."""

# Exact per-model minimums. Anthropic and Gemini both publish minimums that differ by
# model, so an exact match is used before any provider-level fallback.
_MODEL_MIN_CACHEABLE_TOKENS: Mapping[tuple[str, str], int] = {
    ("anthropic", "claude-fable-5"): 512,
    ("anthropic", "claude-opus-5"): 512,
    ("anthropic", "claude-opus-4-8"): 1024,
    ("anthropic", "claude-opus-4-7"): 2048,
    ("anthropic", "claude-opus-4-6"): 4096,
    ("anthropic", "claude-opus-4-5"): 4096,
    ("anthropic", "claude-sonnet-5"): 1024,
    ("anthropic", "claude-sonnet-4-6"): 1024,
    ("anthropic", "claude-sonnet-4-5"): 1024,
    ("anthropic", "claude-haiku-4-5"): 4096,
    ("gemini", "gemini-3.5-flash"): 4096,
    ("gemini", "gemini-2.5-flash"): 2048,
    ("gemini", "gemini-2.5-pro"): 2048,
}

# Fallback for a priced model this table does not name: the provider's *largest*
# documented minimum, which can only understate cacheability, never overstate a saving.
_PROVIDER_MIN_CACHEABLE_TOKENS: Mapping[str, int] = {
    "anthropic": 4096,
    "gemini": 4096,
    "openai": 2048,
}

# Cache-write cost as a multiple of the base input rate. Anthropic charges 1.25x for the
# default 5-minute TTL uniformly across active models (2x for the 1-hour TTL, which a
# caller selects by passing write_multiplier explicitly). Every OpenAI model in the
# packaged price book predates GPT-5.6 and carries no cache-write charge, and Gemini
# implicit caching has none either.
_PROVIDER_CACHE_WRITE_MULTIPLIER: Mapping[str, Decimal] = {
    "anthropic": Decimal("1.25"),
    "gemini": Decimal("1"),
    "openai": Decimal("1"),
}

# Explicit `cache_control`-style breakpoints the provider accepts. OpenAI's priced models
# and every Gemini model cache implicitly, so there is no marker to place.
_PROVIDER_MAX_BREAKPOINTS: Mapping[str, int] = {
    "anthropic": 4,
    "gemini": 0,
    "openai": 0,
}

_DEFAULT_WRITE_MULTIPLIER = Decimal("1")
_SHARE_QUANTUM = Decimal("0.000001")
_SHARE_PRECISION = 40
_MONEY_PRECISION = 60


class PrefixAnalysisError(ValueError):
    """A public, text-free prefix-analysis failure. Every message names its fix."""


def minimum_cacheable_tokens(provider: str, model: str) -> int | None:
    """Documented minimum prefix length the provider will cache, or ``None`` if unknown.

    An exact ``(provider, model)`` match wins. Otherwise the provider's largest documented
    minimum is used, so an unrecognized model is treated as harder to cache rather than
    easier. A provider with no published minimum returns ``None``; nothing is guessed.
    """
    _require_identifier(provider, "provider")
    _require_identifier(model, "model")
    exact = _MODEL_MIN_CACHEABLE_TOKENS.get((provider, model))
    if exact is not None:
        return exact
    return _PROVIDER_MIN_CACHEABLE_TOKENS.get(provider)


def cache_write_multiplier(provider: str) -> Decimal:
    """Documented cache-write cost as a multiple of the base input rate.

    ``1`` means the provider charges nothing extra to populate the cache, so the very
    first hit is already profitable.
    """
    _require_identifier(provider, "provider")
    return _PROVIDER_CACHE_WRITE_MULTIPLIER.get(provider, _DEFAULT_WRITE_MULTIPLIER)


def max_cache_breakpoints(provider: str) -> int:
    """Explicit cache breakpoints the provider accepts; ``0`` for implicit-only caching."""
    _require_identifier(provider, "provider")
    return _PROVIDER_MAX_BREAKPOINTS.get(provider, 0)


@dataclass(frozen=True, slots=True)
class ModelPrefixUsage:
    """One ``(provider, model)`` slice of a prefix cluster.

    Prefix caches are never shared across models on any provider, so this slice -- not the
    whole cluster -- is the unit a :class:`CachePlan` is priced against.

    ``requests`` counts every record in the slice. ``billed_requests`` counts the subset whose
    log reports a positive ``prompt_tokens``, which is the only subset a cache can serve.
    """

    provider: str
    model: str
    requests: int
    billed_requests: int
    prompt_tokens: int
    measured_requests: int
    measured_prompt_tokens: int
    cached_prompt_tokens: int
    hit_requests: int
    min_positive_cached_tokens: int | None

    def __post_init__(self) -> None:
        _require_identifier(self.provider, "provider")
        _require_identifier(self.model, "model")
        for name in (
            "requests",
            "billed_requests",
            "prompt_tokens",
            "measured_requests",
            "measured_prompt_tokens",
            "cached_prompt_tokens",
            "hit_requests",
        ):
            _require_count(getattr(self, name), name)
        if self.min_positive_cached_tokens is not None:
            _require_count(self.min_positive_cached_tokens, "min_positive_cached_tokens")
            if self.min_positive_cached_tokens == 0:
                raise PrefixAnalysisError(
                    "ModelPrefixUsage min_positive_cached_tokens cannot be zero; "
                    "fix: use None when no request reported a positive cache read"
                )
        if self.requests < 1:
            raise PrefixAnalysisError(
                "ModelPrefixUsage requests must be at least 1; "
                "fix: do not build a usage slice for a model that sent no request"
            )
        if self.billed_requests > self.requests:
            raise PrefixAnalysisError(
                f"ModelPrefixUsage billed_requests ({self.billed_requests}) exceeds requests "
                f"({self.requests}); fix: count only the requests whose log reports a positive "
                "prompt_tokens as billed"
            )
        if self.hit_requests > self.billed_requests:
            raise PrefixAnalysisError(
                f"ModelPrefixUsage hit_requests ({self.hit_requests}) exceeds billed_requests "
                f"({self.billed_requests}); fix: a cache read is a subset of a billed prompt, "
                "so a request that billed no prompt tokens cannot have read from cache"
            )
        if self.measured_requests > self.requests:
            raise PrefixAnalysisError(
                f"ModelPrefixUsage measured_requests ({self.measured_requests}) exceeds "
                f"requests ({self.requests}); fix: count only requests whose source log "
                "reported cached_prompt_tokens as measured"
            )
        if self.hit_requests > self.measured_requests:
            raise PrefixAnalysisError(
                f"ModelPrefixUsage hit_requests ({self.hit_requests}) exceeds "
                f"measured_requests ({self.measured_requests}); fix: a cache hit is only "
                "observable on a request that reported cached_prompt_tokens"
            )
        if self.cached_prompt_tokens > self.measured_prompt_tokens:
            raise PrefixAnalysisError(
                f"ModelPrefixUsage cached_prompt_tokens ({self.cached_prompt_tokens}) "
                f"exceeds measured_prompt_tokens ({self.measured_prompt_tokens}); fix: "
                "cached tokens are a subset of the prompt tokens billed on the same request"
            )
        if self.measured_prompt_tokens > self.prompt_tokens:
            raise PrefixAnalysisError(
                f"ModelPrefixUsage measured_prompt_tokens ({self.measured_prompt_tokens}) "
                f"exceeds prompt_tokens ({self.prompt_tokens}); fix: measured prompt tokens "
                "are the subset of prompt tokens on requests that reported cache counts"
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)

    @property
    def realized_cached_share(self) -> Decimal | None:
        """Share of measured prompt tokens the provider actually served from cache."""
        return _share(self.cached_prompt_tokens, self.measured_prompt_tokens)

    @property
    def realized_hit_rate(self) -> Decimal | None:
        """Share of measured requests that got any cache read at all."""
        return _share(self.hit_requests, self.measured_requests)

    @property
    def measured_prefix_tokens(self) -> int | None:
        """Provider-reported prefix token count, or ``None`` when the log reports none.

        This is the smallest positive ``cached_prompt_tokens`` observed on this slice: the
        provider's own count of prompt tokens it served from cache. A cache read covers the
        shared prefix and can cover more, so this is an upper bound on the prefix, and a
        projection built from it is an upper bound too. It is never derived from
        ``prompt_tokens``; pass ``prefix_tokens`` when an exact count is known.
        """
        return self.min_positive_cached_tokens

    @property
    def unrealized_hits(self) -> int:
        """Billed repeats that were sent but not served from cache, so still on the table.

        Counted over :attr:`billed_requests`: a request that billed no prompt could not have
        used a cache, so counting it would project a saving that never existed.
        """
        return max(self.billed_requests - 1 - self.hit_requests, 0)


@dataclass(frozen=True, slots=True)
class PrefixCluster:
    """Every request that shared one ``system_prefix_hash``."""

    system_prefix_hash: str
    prefix_chars: int
    requests: int
    usage: tuple[ModelPrefixUsage, ...]
    first_seen: float
    last_seen: float

    def __post_init__(self) -> None:
        _require_identifier(self.system_prefix_hash, "system_prefix_hash")
        _require_count(self.prefix_chars, "prefix_chars")
        _require_count(self.requests, "requests")
        if self.requests < 1:
            raise PrefixAnalysisError(
                "PrefixCluster requests must be at least 1; "
                "fix: do not build a cluster for a prefix no request used"
            )
        if not self.usage:
            raise PrefixAnalysisError(
                "PrefixCluster usage must hold at least one model slice; "
                "fix: build the cluster from records, which always carry a provider and model"
            )
        keys = [slice_.key for slice_ in self.usage]
        if keys != sorted(keys):
            raise PrefixAnalysisError(
                "PrefixCluster usage must be sorted by (provider, model); "
                "fix: sort the slices so a cluster is comparable regardless of input order"
            )
        if len(set(keys)) != len(keys):
            raise PrefixAnalysisError(
                "PrefixCluster usage has a duplicate (provider, model); "
                "fix: merge the counts for a pair into exactly one slice"
            )
        total = sum(slice_.requests for slice_ in self.usage)
        if total != self.requests:
            raise PrefixAnalysisError(
                f"PrefixCluster usage slices sum to {total} requests but the cluster "
                f"reports {self.requests}; fix: every request must land in exactly one slice"
            )
        for name in ("first_seen", "last_seen"):
            _require_timestamp(getattr(self, name), name)
        if self.last_seen < self.first_seen:
            raise PrefixAnalysisError(
                "PrefixCluster last_seen precedes first_seen; "
                "fix: take first_seen as the minimum and last_seen as the maximum timestamp"
            )

    @property
    def has_prefix(self) -> bool:
        """False for traffic with no system prefix, which has nothing to cache."""
        return self.prefix_chars > 0

    @property
    def models(self) -> tuple[tuple[str, str], ...]:
        return tuple(slice_.key for slice_ in self.usage)

    @property
    def distinct_models(self) -> int:
        return len(self.usage)

    @property
    def total_prefix_chars(self) -> int:
        """Prefix characters sent across every request in the cluster."""
        return self.prefix_chars * self.requests

    @property
    def repeated_prefix_chars(self) -> int:
        """Prefix characters a warm cache could have covered: everything after the first."""
        return self.prefix_chars * (self.requests - 1)

    @property
    def prompt_tokens(self) -> int:
        return sum(slice_.prompt_tokens for slice_ in self.usage)

    @property
    def billed_requests(self) -> int:
        """Requests whose log reported a positive ``prompt_tokens``."""
        return sum(slice_.billed_requests for slice_ in self.usage)

    @property
    def measured_requests(self) -> int:
        return sum(slice_.measured_requests for slice_ in self.usage)

    @property
    def measured_prompt_tokens(self) -> int:
        return sum(slice_.measured_prompt_tokens for slice_ in self.usage)

    @property
    def cached_prompt_tokens(self) -> int:
        return sum(slice_.cached_prompt_tokens for slice_ in self.usage)

    @property
    def hit_requests(self) -> int:
        return sum(slice_.hit_requests for slice_ in self.usage)

    @property
    def realized_cached_share(self) -> Decimal | None:
        """Realized cached share over the whole cluster, or ``None`` when unmeasured."""
        return _share(self.cached_prompt_tokens, self.measured_prompt_tokens)

    @property
    def realized_hit_rate(self) -> Decimal | None:
        return _share(self.hit_requests, self.measured_requests)

    def slice_for(self, provider: str, model: str) -> ModelPrefixUsage:
        """Return one model slice, or raise naming the pairs this cluster does hold."""
        for slice_ in self.usage:
            if slice_.key == (provider, model):
                return slice_
        known = ", ".join(f"{name}/{ident}" for name, ident in self.models)
        raise PrefixAnalysisError(
            f"cluster {self.system_prefix_hash} has no slice for {provider}/{model}; "
            f"pairs in this cluster: {known}; fix: ask for one of those pairs"
        )


@dataclass(frozen=True, slots=True)
class VolatilePrefix:
    """A prefix whose length holds steady while its hash churns: caching never hits.

    ``churn_rate`` is ``(distinct_hashes - 1) / (requests - 1)``: exactly ``0`` when every
    request reused one prefix, and exactly ``1`` when every request sent a different one.
    """

    provider: str
    model: str
    prefix_chars: int
    requests: int
    distinct_hashes: int
    churn_rate: Decimal
    first_seen: float
    last_seen: float

    def __post_init__(self) -> None:
        _require_identifier(self.provider, "provider")
        _require_identifier(self.model, "model")
        _require_count(self.prefix_chars, "prefix_chars")
        _require_count(self.requests, "requests")
        _require_count(self.distinct_hashes, "distinct_hashes")
        if self.requests < 2:
            raise PrefixAnalysisError(
                "VolatilePrefix needs at least 2 requests to have a churn rate; "
                "fix: report churn only for a group that repeated"
            )
        if not 1 <= self.distinct_hashes <= self.requests:
            raise PrefixAnalysisError(
                f"VolatilePrefix distinct_hashes ({self.distinct_hashes}) must be between 1 "
                f"and requests ({self.requests}); fix: count the distinct prefix hashes seen "
                "within this group"
            )
        _require_rate(self.churn_rate, "churn_rate")
        for name in ("first_seen", "last_seen"):
            _require_timestamp(getattr(self, name), name)
        if self.last_seen < self.first_seen:
            raise PrefixAnalysisError(
                "VolatilePrefix last_seen precedes first_seen; "
                "fix: take first_seen as the minimum and last_seen as the maximum timestamp"
            )

    def message(self) -> str:
        """Operator-facing report. Names the measurement, then the fix."""
        return (
            f"{self.provider}/{self.model} sent {self.requests} requests with a "
            f"{self.prefix_chars}-character system prefix but {self.distinct_hashes} "
            f"different prefixes (churn {self.churn_rate}); the prefix cache cannot hit and "
            f"no provider reports this as an error; fix: the prefix length is stable while "
            f"its content is not, so something per-request is embedded in it -- move the "
            f"injected timestamp, uuid, or per-user string out of the system prefix and into "
            f"a later message, then re-run the audit. See "
            f"{PROVIDER_CACHE_DOCS.get(self.provider, 'your provider cache documentation')}"
        )


@dataclass(frozen=True, slots=True)
class CachePlan:
    """What prefix caching is worth for one cluster on one model, and what it needs.

    ``projected_saving`` is ``None`` whenever the plan could not be priced -- absent token
    counts or an unpriced model -- and never a placeholder zero standing in for a real
    number. A priced plan that is simply not worth taking reports ``Decimal("0")`` with a
    status explaining why.

    ``break_even_hits`` is the smallest number of cache reads that repays the write premium:
    ``0`` when the provider charges no premium, and ``None`` when no number of reads could,
    because the price book's cache-read rate is not below its input rate.
    """

    system_prefix_hash: str
    provider: str
    model: str
    status: str
    requests: int
    billed_requests: int
    unrealized_hits: int
    breakpoints: int
    min_cacheable_tokens: int | None
    prefix_tokens: int | None
    token_source: str
    write_premium: Decimal | None
    read_saving: Decimal | None
    projected_saving: Decimal | None
    break_even_hits: int | None
    fix: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.system_prefix_hash, "system_prefix_hash")
        _require_identifier(self.provider, "provider")
        _require_identifier(self.model, "model")
        if self.status not in PLAN_STATUSES:
            expected = ", ".join(sorted(PLAN_STATUSES))
            raise PrefixAnalysisError(
                f"CachePlan status {self.status!r} is not a known status; "
                f"fix: use one of: {expected}"
            )
        if self.token_source not in TOKEN_SOURCES:
            expected = ", ".join(sorted(TOKEN_SOURCES))
            raise PrefixAnalysisError(
                f"CachePlan token_source {self.token_source!r} is not a known source; "
                f"fix: use one of: {expected}"
            )
        for name in ("requests", "billed_requests", "unrealized_hits", "breakpoints"):
            _require_count(getattr(self, name), name)
        if self.billed_requests > self.requests:
            raise PrefixAnalysisError(
                f"CachePlan billed_requests ({self.billed_requests}) exceeds requests "
                f"({self.requests}); fix: count only the requests whose log reports a positive "
                "prompt_tokens as billed"
            )
        for name in ("min_cacheable_tokens", "prefix_tokens", "break_even_hits"):
            value = getattr(self, name)
            if value is not None:
                _require_count(value, name)
        for name in ("write_premium", "read_saving", "projected_saving"):
            value = getattr(self, name)
            if value is not None:
                _require_money(value, name)
        if (self.prefix_tokens is None) != (self.token_source == "unavailable"):
            raise PrefixAnalysisError(
                "CachePlan prefix_tokens and token_source disagree; fix: set token_source to "
                "'unavailable' exactly when prefix_tokens is None"
            )
        if self.status == "needs_token_counts" and self.projected_saving is not None:
            raise PrefixAnalysisError(
                "CachePlan status is 'needs_token_counts' but a saving was projected; "
                "fix: a prefix token count is never inferred from prompt_tokens -- leave "
                "projected_saving as None"
            )
        if (self.status == "ok") != (self.fix is None):
            raise PrefixAnalysisError(
                f"CachePlan status {self.status!r} and fix disagree; fix: give every status "
                "other than 'ok' a fix clause, and status 'ok' none"
            )
        if self.fix is not None and "fix:" not in self.fix:
            raise PrefixAnalysisError(
                "CachePlan fix must name the operator's next step in a 'fix:' clause; "
                "fix: write the fix as '<reason>; fix: <what to do>'"
            )

    @property
    def actionable(self) -> bool:
        """True only for a priced plan with money left on the table."""
        return self.status == "ok"

    def message(self) -> str:
        """One operator-facing line. Always carries a ``fix:`` unless the plan is ready."""
        head = (
            f"{self.provider}/{self.model} prefix {self.system_prefix_hash}: "
            f"{self.requests} requests, {self.billed_requests} billed, "
            f"{self.unrealized_hits} uncached repeats"
        )
        if self.status == "ok":
            return (
                f"{head}; projected saving {self.projected_saving} after a "
                f"{self.write_premium} cache write, break-even at "
                f"{self.break_even_hits} hits, {self.breakpoints} breakpoint(s)"
            )
        return f"{head}; status {self.status}; {self.fix}"


@dataclass(frozen=True, slots=True)
class PrefixAnalysis:
    """One pass over a record stream: clusters, and the volatile prefixes among them."""

    records: int
    clusters: tuple[PrefixCluster, ...]
    volatile: tuple[VolatilePrefix, ...]

    def __post_init__(self) -> None:
        _require_count(self.records, "records")
        total = sum(cluster.requests for cluster in self.clusters)
        if total != self.records:
            raise PrefixAnalysisError(
                f"PrefixAnalysis clusters hold {total} requests but the analysis reports "
                f"{self.records}; fix: every record must land in exactly one cluster"
            )
        hashes = [cluster.system_prefix_hash for cluster in self.clusters]
        if len(set(hashes)) != len(hashes):
            raise PrefixAnalysisError(
                "PrefixAnalysis has two clusters for one system_prefix_hash; "
                "fix: merge them -- a prefix hash identifies exactly one cluster"
            )

    def cluster(self, system_prefix_hash: str) -> PrefixCluster:
        """Return one cluster by prefix hash, or raise naming how many exist."""
        for cluster in self.clusters:
            if cluster.system_prefix_hash == system_prefix_hash:
                return cluster
        raise PrefixAnalysisError(
            f"no cluster for prefix hash {system_prefix_hash!r}; the analysis holds "
            f"{len(self.clusters)} clusters; fix: read the hash from a record's "
            "system_prefix_hash, or from PrefixAnalysis.clusters"
        )

    @property
    def cacheable_clusters(self) -> tuple[PrefixCluster, ...]:
        """Clusters that have a system prefix at all and sent it more than once."""
        return tuple(
            cluster for cluster in self.clusters if cluster.has_prefix and cluster.requests > 1
        )

    def plans(
        self,
        price_book: PriceBook,
        *,
        prefix_tokens: Mapping[str, int] | None = None,
        write_multipliers: Mapping[str, Decimal] | None = None,
    ) -> tuple[CachePlan, ...]:
        """One plan per (cluster, provider, model), best-paying first.

        ``prefix_tokens`` maps a ``system_prefix_hash`` to a known prefix token count and
        overrides the count measured from the log. ``write_multipliers`` maps a provider to
        a cache-write multiple, for selecting a non-default cache TTL.
        """
        if not isinstance(price_book, PriceBook):
            raise PrefixAnalysisError(
                "plans() needs a PriceBook; fix: pass PriceBook.load() or "
                "PriceBook.load(path) so every amount comes from a sourced rate card"
            )
        supplied = _check_token_overrides(prefix_tokens)
        multipliers = _check_multiplier_overrides(write_multipliers)
        built = [
            plan_cache(
                cluster,
                slice_,
                price_book,
                prefix_tokens=supplied.get(cluster.system_prefix_hash),
                write_multiplier=multipliers.get(slice_.provider),
            )
            for cluster in self.clusters
            for slice_ in cluster.usage
        ]
        built.sort(
            key=lambda plan: (
                -(plan.projected_saving or Decimal(0)),
                plan.system_prefix_hash,
                plan.provider,
                plan.model,
            )
        )
        return tuple(built)


def plan_cache(
    cluster: PrefixCluster,
    usage: ModelPrefixUsage,
    price_book: PriceBook,
    *,
    prefix_tokens: int | None = None,
    write_multiplier: Decimal | None = None,
) -> CachePlan:
    """Price prefix caching for one model slice of one cluster.

    ``prefix_tokens`` is a known token count for the prefix. When omitted, the provider's
    own cache-read counts in the log are used if present. When neither exists the plan
    reports ``needs_token_counts``: a character count is not a token count, and
    ``prompt_tokens`` is never used to invent one.

    A projection assumes one cache write covers the repeats this cluster observed, so it is an
    upper bound when those repeats are spread beyond the provider's cache lifetime.
    """
    if not isinstance(cluster, PrefixCluster) or not isinstance(usage, ModelPrefixUsage):
        raise PrefixAnalysisError(
            "plan_cache() needs a PrefixCluster and one of its ModelPrefixUsage slices; "
            "fix: pass cluster and cluster.slice_for(provider, model)"
        )
    if not isinstance(price_book, PriceBook):
        raise PrefixAnalysisError(
            "plan_cache() needs a PriceBook; fix: pass PriceBook.load() or "
            "PriceBook.load(path) so every amount comes from a sourced rate card"
        )
    if usage not in cluster.usage:
        raise PrefixAnalysisError(
            f"usage slice {usage.provider}/{usage.model} does not belong to cluster "
            f"{cluster.system_prefix_hash}; fix: take the slice from cluster.usage"
        )
    if prefix_tokens is not None:
        _require_count(prefix_tokens, "prefix_tokens")
        if prefix_tokens == 0:
            raise PrefixAnalysisError(
                "prefix_tokens cannot be zero; fix: omit prefix_tokens when the prefix "
                "token count is unknown, so the plan can say so"
            )
    multiplier = cache_write_multiplier(usage.provider)
    if write_multiplier is not None:
        _require_multiplier(write_multiplier)
        multiplier = write_multiplier

    minimum = minimum_cacheable_tokens(usage.provider, usage.model)
    docs = PROVIDER_CACHE_DOCS.get(usage.provider, "your provider's cache documentation")
    build = _PlanBuilder(cluster, usage, minimum)

    if not cluster.has_prefix:
        return build.refuse(
            "no_prefix",
            "this traffic carries no system prefix, so there is no prefix to cache; fix: "
            "move the instructions every request repeats into a leading system message",
        )
    if usage.billed_requests < 2:
        return build.refuse(
            "no_repeats",
            f"{usage.provider}/{usage.model} sent this prefix on {usage.requests} request(s), "
            f"of which {usage.billed_requests} billed a prompt, and a cache only pays on a "
            "billed repeat; fix: nothing to do until the prefix is reused on a request the "
            "provider bills -- a refused or rate-limited request never reaches the cache",
        )

    tokens = prefix_tokens if prefix_tokens is not None else usage.measured_prefix_tokens
    source = "supplied" if prefix_tokens is not None else "measured"
    if tokens is None:
        return build.refuse(
            "needs_token_counts",
            "this log reports no cached prompt tokens, so the prefix token count is "
            "unknown and is never inferred from prompt_tokens; fix: re-export the log with "
            "the provider's cached-token field, or pass a counted prefix_tokens",
        )

    try:
        rates = _CacheRates.load(price_book, usage.provider, usage.model, tokens, multiplier)
    except UnknownModelError:
        return build.refuse(
            "unpriced_model",
            f"the price book has no rate for {usage.provider}/{usage.model}, and no price "
            f"is ever estimated from a similar model; fix: add the pair to a price book "
            f"file and pass it with --price-book FILE",
            tokens=tokens,
            source=source,
        )

    if minimum is not None and tokens < minimum:
        return build.refuse(
            "below_provider_minimum",
            f"the prefix is {tokens} tokens but {usage.provider} caches nothing shorter "
            f"than {minimum} tokens and returns no error when it silently declines; fix: "
            f"extend the shared prefix past {minimum} tokens, or accept that this traffic "
            f"is not cacheable. See {docs}",
            tokens=tokens,
            source=source,
            rates=rates,
            projected=Decimal(0),
        )

    breakpoints = 1 if max_cache_breakpoints(usage.provider) >= 1 else 0
    if usage.unrealized_hits == 0:
        return build.refuse(
            "already_cached",
            f"every repeat on {usage.provider}/{usage.model} was already served from "
            "cache, so there is no saving left here; fix: nothing to do -- keep the prefix "
            "stable and this stays true",
            tokens=tokens,
            source=source,
            rates=rates,
            projected=Decimal(0),
            breakpoints=breakpoints,
        )

    premium_due = rates.write_premium if usage.hit_requests == 0 else Decimal(0)
    projected = rates.read_saving * usage.unrealized_hits - premium_due
    if projected <= 0:
        if rates.break_even_hits is None:
            unprofitable = (
                f"a cache read on {usage.provider}/{usage.model} costs as much as a fresh "
                f"read of the same {tokens} tokens, so no number of hits recovers the "
                f"{rates.write_premium} cache write; fix: check the cached_input rate in the "
                f"price book against {docs}, and leave this prefix uncached until it is lower"
            )
        else:
            unprofitable = (
                f"{usage.unrealized_hits} uncached repeat(s) do not recover the "
                f"{rates.write_premium} cache write on {usage.provider}/{usage.model}; fix: "
                f"leave this prefix uncached until it is reused at least "
                f"{rates.break_even_hits} times inside the cache lifetime"
            )
        return build.refuse(
            "below_break_even",
            unprofitable,
            tokens=tokens,
            source=source,
            rates=rates,
            projected=Decimal(0),
            breakpoints=breakpoints,
        )
    return build.finish(
        "ok",
        tokens=tokens,
        source=source,
        rates=rates,
        projected=projected,
        breakpoints=breakpoints,
        fix=None,
    )


def analyze_prefixes(
    records: Iterable[RequestRecord],
    *,
    churn_threshold: Decimal = VOLATILE_CHURN_THRESHOLD,
    min_requests: int = VOLATILE_MIN_REQUESTS,
    min_prefix_chars: int = VOLATILE_MIN_PREFIX_CHARS,
) -> PrefixAnalysis:
    """Cluster a record stream by system prefix and flag volatile prefixes.

    One streaming pass; memory is bounded by the number of distinct prefixes, never by the
    number of records. The result depends only on the multiset of records, not their order.
    """
    _require_rate(churn_threshold, "churn_threshold")
    _require_count(min_requests, "min_requests")
    if min_requests < 2:
        raise PrefixAnalysisError(
            "min_requests must be at least 2; fix: churn is undefined for a single "
            "request -- pass min_requests=2 or higher"
        )
    _require_count(min_prefix_chars, "min_prefix_chars")
    if min_prefix_chars < 1:
        raise PrefixAnalysisError(
            "min_prefix_chars must be at least 1; fix: a request with no system prefix has "
            "nothing to churn -- pass min_prefix_chars=1 or higher"
        )

    clusters: dict[str, _Cluster] = {}
    lengths: dict[tuple[str, str, int], _Length] = {}
    seen = 0
    for record in records:
        if not isinstance(record, RequestRecord):
            raise PrefixAnalysisError(
                f"analyze_prefixes() takes RequestRecord values, not "
                f"{type(record).__name__}; fix: pass read_requests(path) or another "
                "iterable of RequestRecord"
            )
        seen += 1
        cluster = clusters.get(record.system_prefix_hash)
        if cluster is None:
            cluster = clusters[record.system_prefix_hash] = _Cluster(record)
        cluster.add(record)
        if record.system_prefix_chars >= min_prefix_chars:
            key = (record.provider, record.model, record.system_prefix_chars)
            group = lengths.get(key)
            if group is None:
                group = lengths[key] = _Length(record)
            group.add(record)

    built = [cluster.build() for cluster in clusters.values()]
    built.sort(
        key=lambda cluster: (
            -cluster.requests,
            -cluster.repeated_prefix_chars,
            cluster.system_prefix_hash,
        )
    )
    flagged = [
        signal
        for signal in (group.build() for group in lengths.values())
        if signal is not None and signal.requests >= min_requests
        if signal.churn_rate >= churn_threshold
    ]
    flagged.sort(
        key=lambda signal: (
            -signal.requests,
            -signal.churn_rate,
            signal.provider,
            signal.model,
            signal.prefix_chars,
        )
    )
    return PrefixAnalysis(records=seen, clusters=tuple(built), volatile=tuple(flagged))


# --------------------------------------------------------------------------------------
# streaming accumulators -- private, mutable, and never returned
# --------------------------------------------------------------------------------------


class _Usage:
    """Running counts for one ``(provider, model)`` inside one cluster."""

    __slots__ = (
        "billed_requests",
        "cached_tokens",
        "hit_requests",
        "measured_prompt_tokens",
        "measured_requests",
        "min_positive_cached",
        "model",
        "prompt_tokens",
        "provider",
        "requests",
    )

    def __init__(self, record: RequestRecord) -> None:
        self.provider = record.provider
        self.model = record.model
        self.requests = 0
        self.billed_requests = 0
        self.prompt_tokens = 0
        self.measured_requests = 0
        self.measured_prompt_tokens = 0
        self.cached_tokens = 0
        self.hit_requests = 0
        self.min_positive_cached: int | None = None

    def add(self, record: RequestRecord) -> None:
        self.requests += 1
        self.prompt_tokens += record.prompt_tokens
        if record.prompt_tokens > 0:
            self.billed_requests += 1
        cached = record.cached_prompt_tokens
        if cached is None:
            return
        self.measured_requests += 1
        self.measured_prompt_tokens += record.prompt_tokens
        self.cached_tokens += cached
        if cached > 0:
            self.hit_requests += 1
            if self.min_positive_cached is None or cached < self.min_positive_cached:
                self.min_positive_cached = cached

    def build(self) -> ModelPrefixUsage:
        return ModelPrefixUsage(
            provider=self.provider,
            model=self.model,
            requests=self.requests,
            billed_requests=self.billed_requests,
            prompt_tokens=self.prompt_tokens,
            measured_requests=self.measured_requests,
            measured_prompt_tokens=self.measured_prompt_tokens,
            cached_prompt_tokens=self.cached_tokens,
            hit_requests=self.hit_requests,
            min_positive_cached_tokens=self.min_positive_cached,
        )


class _Cluster:
    """Running counts for one ``system_prefix_hash``."""

    __slots__ = ("first_seen", "last_seen", "prefix_chars", "prefix_hash", "requests", "usage")

    def __init__(self, record: RequestRecord) -> None:
        self.prefix_hash = record.system_prefix_hash
        # Equal hashes mean equal prefix text, hence an equal character count. The maximum
        # is taken so a 128-bit digest collision reports the larger prefix instead of
        # depending on which record arrived first.
        self.prefix_chars = 0
        self.requests = 0
        self.first_seen = record.timestamp
        self.last_seen = record.timestamp
        self.usage: dict[tuple[str, str], _Usage] = {}

    def add(self, record: RequestRecord) -> None:
        self.requests += 1
        self.prefix_chars = max(self.prefix_chars, record.system_prefix_chars)
        self.first_seen = min(self.first_seen, record.timestamp)
        self.last_seen = max(self.last_seen, record.timestamp)
        key = (record.provider, record.model)
        slot = self.usage.get(key)
        if slot is None:
            slot = self.usage[key] = _Usage(record)
        slot.add(record)

    def build(self) -> PrefixCluster:
        return PrefixCluster(
            system_prefix_hash=self.prefix_hash,
            prefix_chars=self.prefix_chars,
            requests=self.requests,
            usage=tuple(self.usage[key].build() for key in sorted(self.usage)),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


class _Length:
    """Running counts for one ``(provider, model, prefix length)`` churn group."""

    __slots__ = ("chars", "first_seen", "hashes", "last_seen", "model", "provider", "requests")

    def __init__(self, record: RequestRecord) -> None:
        self.provider = record.provider
        self.model = record.model
        self.chars = record.system_prefix_chars
        self.requests = 0
        self.hashes: set[str] = set()
        self.first_seen = record.timestamp
        self.last_seen = record.timestamp

    def add(self, record: RequestRecord) -> None:
        self.requests += 1
        self.hashes.add(record.system_prefix_hash)
        self.first_seen = min(self.first_seen, record.timestamp)
        self.last_seen = max(self.last_seen, record.timestamp)

    def build(self) -> VolatilePrefix | None:
        if self.requests < 2:
            return None
        churn = _share(len(self.hashes) - 1, self.requests - 1)
        if churn is None:
            return None
        return VolatilePrefix(
            provider=self.provider,
            model=self.model,
            prefix_chars=self.chars,
            requests=self.requests,
            distinct_hashes=len(self.hashes),
            churn_rate=churn,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


# --------------------------------------------------------------------------------------
# pricing and plan assembly
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CacheRates:
    """The three numbers a projection needs, all from :meth:`PriceBook.price`."""

    read_saving: Decimal
    write_premium: Decimal
    break_even_hits: int | None

    @classmethod
    def load(
        cls,
        price_book: PriceBook,
        provider: str,
        model: str,
        tokens: int,
        multiplier: Decimal,
    ) -> _CacheRates:
        fresh = price_book.price(provider, model, tokens_in=tokens, tokens_out=0)
        cached = price_book.price(provider, model, tokens_in=tokens, tokens_out=0, cached_in=tokens)
        read_saving = fresh - cached
        with localcontext() as ctx:
            ctx.prec = _MONEY_PRECISION
            write_premium = fresh * (multiplier - Decimal(1))
        if read_saving <= 0:
            return cls(read_saving=read_saving, write_premium=write_premium, break_even_hits=None)
        with localcontext() as ctx:
            ctx.prec = _MONEY_PRECISION
            ratio = write_premium / read_saving
        return cls(
            read_saving=read_saving,
            write_premium=write_premium,
            break_even_hits=math.ceil(ratio),
        )


class _PlanBuilder:
    """Assembles a :class:`CachePlan`, so each exit names only what it actually knows."""

    __slots__ = ("cluster", "minimum", "usage")

    def __init__(self, cluster: PrefixCluster, usage: ModelPrefixUsage, minimum: int | None):
        self.cluster = cluster
        self.usage = usage
        self.minimum = minimum

    def refuse(
        self,
        status: str,
        fix: str,
        *,
        tokens: int | None = None,
        source: str = "unavailable",
        rates: _CacheRates | None = None,
        projected: Decimal | None = None,
        breakpoints: int = 0,
    ) -> CachePlan:
        return self.finish(
            status,
            tokens=tokens,
            source=source,
            rates=rates,
            projected=projected,
            breakpoints=breakpoints,
            fix=fix,
        )

    def finish(
        self,
        status: str,
        *,
        tokens: int | None,
        source: str,
        rates: _CacheRates | None,
        projected: Decimal | None,
        breakpoints: int,
        fix: str | None,
    ) -> CachePlan:
        return CachePlan(
            system_prefix_hash=self.cluster.system_prefix_hash,
            provider=self.usage.provider,
            model=self.usage.model,
            status=status,
            requests=self.usage.requests,
            billed_requests=self.usage.billed_requests,
            unrealized_hits=self.usage.unrealized_hits,
            breakpoints=breakpoints,
            min_cacheable_tokens=self.minimum,
            prefix_tokens=tokens,
            token_source=source if tokens is not None else "unavailable",
            write_premium=None if rates is None else rates.write_premium,
            read_saving=None if rates is None else rates.read_saving,
            projected_saving=projected,
            break_even_hits=None if rates is None else rates.break_even_hits,
            fix=fix,
        )


# --------------------------------------------------------------------------------------
# value guards
# --------------------------------------------------------------------------------------


def _share(numerator: int, denominator: int) -> Decimal | None:
    """A ratio quantized for deterministic comparison, or ``None`` with nothing measured."""
    if denominator <= 0:
        return None
    with localcontext() as ctx:
        ctx.prec = _SHARE_PRECISION
        return (Decimal(numerator) / Decimal(denominator)).quantize(_SHARE_QUANTUM)


def _require_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PrefixAnalysisError(
            f"{name} must be a non-empty string; fix: pass the exact {name} that appears in "
            "the records being analyzed"
        )


def _require_count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrefixAnalysisError(f"{name} must be an integer; fix: convert {name} to int")
    if value < 0:
        raise PrefixAnalysisError(
            f"{name} cannot be negative; fix: pass a non-negative count for {name}"
        )


def _require_timestamp(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefixAnalysisError(
            f"{name} must be a real number of unix epoch seconds; "
            f"fix: take {name} from RequestRecord.timestamp"
        )
    if not math.isfinite(value) or value <= 0:
        raise PrefixAnalysisError(
            f"{name} must be a finite positive unix epoch value; "
            f"fix: take {name} from RequestRecord.timestamp"
        )


def _require_money(value: object, name: str) -> None:
    if not isinstance(value, Decimal):
        raise PrefixAnalysisError(
            f"{name} must be a decimal.Decimal; fix: a float in a cost path is never "
            f"accepted -- build {name} from PriceBook.price"
        )
    if not value.is_finite():
        raise PrefixAnalysisError(f"{name} must be finite; fix: build {name} from PriceBook.price")


def _require_rate(value: object, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise PrefixAnalysisError(
            f"{name} must be a finite decimal.Decimal; fix: pass Decimal('0.5'), never a float"
        )
    if not Decimal(0) <= value <= Decimal(1):
        raise PrefixAnalysisError(
            f"{name} must be between 0 and 1 inclusive; fix: express {name} as a fraction "
            "of requests, for example Decimal('0.5')"
        )


def _require_multiplier(value: object) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise PrefixAnalysisError(
            "write_multiplier must be a finite decimal.Decimal; fix: pass Decimal('1.25') "
            "for a 5-minute Anthropic cache write, never a float"
        )
    if value < Decimal(1):
        raise PrefixAnalysisError(
            f"write_multiplier {value} is below 1; fix: a cache write never costs less than "
            "a fresh read -- pass Decimal('1') when the provider charges no premium"
        )


def _check_token_overrides(prefix_tokens: Mapping[str, int] | None) -> Mapping[str, int]:
    if prefix_tokens is None:
        return {}
    if not isinstance(prefix_tokens, Mapping):
        raise PrefixAnalysisError(
            "prefix_tokens must be a mapping of system_prefix_hash to token count; "
            "fix: pass {record.system_prefix_hash: counted_tokens}"
        )
    for prefix_hash, tokens in prefix_tokens.items():
        _require_identifier(prefix_hash, "prefix_tokens key")
        _require_count(tokens, f"prefix_tokens[{prefix_hash!r}]")
        if tokens == 0:
            raise PrefixAnalysisError(
                f"prefix_tokens[{prefix_hash!r}] cannot be zero; fix: omit the entry when "
                "the prefix token count is unknown, so the plan can say so"
            )
    return prefix_tokens


def _check_multiplier_overrides(
    write_multipliers: Mapping[str, Decimal] | None,
) -> Mapping[str, Decimal]:
    if write_multipliers is None:
        return {}
    if not isinstance(write_multipliers, Mapping):
        raise PrefixAnalysisError(
            "write_multipliers must be a mapping of provider to cache-write multiple; "
            "fix: pass {'anthropic': Decimal('2')} for the 1-hour cache TTL"
        )
    for provider, multiplier in write_multipliers.items():
        _require_identifier(provider, "write_multipliers key")
        _require_multiplier(multiplier)
    return write_multipliers
