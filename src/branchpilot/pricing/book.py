from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, Inexact, InvalidOperation, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
STALENESS_HORIZON_DAYS = 90
TOKENS_PER_RATE_UNIT = Decimal(1_000_000)

PACKAGED_PRICE_BOOK = Path(__file__).with_name("prices.json")

_CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RATE_FIELDS = ("input", "output", "cached_input", "batch_input", "batch_output")
_ENTRY_FIELDS = frozenset(
    (*_RATE_FIELDS, "provider", "model", "currency", "effective_date", "source_url")
)
_DOCUMENT_FIELDS = frozenset({"schema_version", "entries"})
_SUGGESTION_LIMIT = 5
# Wide enough that no realistic token count rounds; the Inexact trap catches the rest.
_MONEY_PRECISION = 60


class PriceBookError(ValueError):
    """A public, content-free price book error. Every message names its fix."""


class UnknownModelError(PriceBookError):
    """Raised when a (provider, model) pair has no configured price."""


@dataclass(frozen=True, slots=True)
class PriceEntry:
    """One sourced rate card, quoted per 1,000,000 tokens."""

    provider: str
    model: str
    input: Decimal
    output: Decimal
    cached_input: Decimal
    batch_input: Decimal
    batch_output: Decimal
    currency: str
    effective_date: date
    source_url: str

    def __post_init__(self) -> None:
        _require_identifier(self.provider, "provider")
        _require_identifier(self.model, "model")
        label = f"{self.provider}/{self.model}"
        for name in _RATE_FIELDS:
            _require_rate(getattr(self, name), name, label)
        if self.cached_input > self.input:
            raise PriceBookError(
                f"price entry {label} has cached_input {self.cached_input} above input "
                f"{self.input}; fix: a cache read is never more expensive than a fresh read — "
                f"set cached_input to at most {self.input} in the price book"
            )
        if self.batch_input > self.input:
            raise PriceBookError(
                f"price entry {label} has batch_input {self.batch_input} above input "
                f"{self.input}; fix: the batch lane is never more expensive than the "
                f"synchronous lane — set batch_input to at most {self.input}"
            )
        if self.batch_output > self.output:
            raise PriceBookError(
                f"price entry {label} has batch_output {self.batch_output} above output "
                f"{self.output}; fix: the batch lane is never more expensive than the "
                f"synchronous lane — set batch_output to at most {self.output}"
            )
        if not isinstance(self.currency, str) or not _CURRENCY_CODE.match(self.currency):
            raise PriceBookError(
                f"price entry {label} has currency {self.currency!r}; fix: set currency to a "
                f"3-letter uppercase ISO-4217 code such as 'USD'"
            )
        if type(self.effective_date) is not date:
            raise PriceBookError(
                f"price entry {label} has a non-date effective_date; fix: set effective_date to "
                f"an ISO-8601 day, 'YYYY-MM-DD'"
            )
        _require_absolute_url(self.source_url, label)

    def rates(self, *, batch: bool) -> tuple[Decimal, Decimal]:
        """Return the (input, output) rate pair for the requested lane."""
        if batch:
            return self.batch_input, self.batch_output
        return self.input, self.output

    def age_days(self, today: date) -> int:
        return (today - self.effective_date).days

    def to_dict(self) -> dict[str, str]:
        payload = {"provider": self.provider, "model": self.model}
        payload.update({name: str(getattr(self, name)) for name in _RATE_FIELDS})
        payload["currency"] = self.currency
        payload["effective_date"] = self.effective_date.isoformat()
        payload["source_url"] = self.source_url
        return payload


@dataclass(frozen=True, slots=True)
class StalenessWarning:
    """A price entry older than the staleness horizon. The caller decides what to do."""

    entry: PriceEntry
    age_days: int
    horizon_days: int

    def __post_init__(self) -> None:
        if not isinstance(self.entry, PriceEntry):
            raise TypeError("staleness warning entry must be a PriceEntry")
        for name in ("age_days", "horizon_days"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"staleness warning {name} must be an integer")
        if self.age_days <= self.horizon_days:
            raise ValueError("a staleness warning must be older than its horizon")

    def message(self) -> str:
        return (
            f"price for {self.entry.provider}/{self.entry.model} is {self.age_days} days old "
            f"(horizon {self.horizon_days} days, effective "
            f"{self.entry.effective_date.isoformat()}); fix: re-check {self.entry.source_url} and "
            f"update the entry, or pass a current --price-book FILE"
        )


@dataclass(frozen=True, slots=True)
class PriceBook:
    """An immutable, exact-arithmetic rate card set."""

    entries: tuple[PriceEntry, ...]
    _by_key: Mapping[tuple[str, str], PriceEntry] = field(init=False, repr=False, compare=False)
    _by_provider: Mapping[str, tuple[str, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise PriceBookError(
                "price book has no entries; fix: add at least one entry to the 'entries' array of "
                "your price book file"
            )
        by_key: dict[tuple[str, str], PriceEntry] = {}
        models: dict[str, list[str]] = {}
        for entry in entries:
            if not isinstance(entry, PriceEntry):
                raise TypeError("price book entries must all be PriceEntry values")
            key = (entry.provider, entry.model)
            if key in by_key:
                raise PriceBookError(
                    f"price book has duplicate entries for {entry.provider}/{entry.model}; "
                    f"fix: keep exactly one entry per (provider, model) pair and delete the "
                    f"redundant one"
                )
            by_key[key] = entry
            models.setdefault(entry.provider, []).append(entry.model)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))
        object.__setattr__(
            self,
            "_by_provider",
            MappingProxyType({name: tuple(sorted(ids)) for name, ids in models.items()}),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceBook:
        """Load a price book. ``None`` loads the packaged, sourced rate cards."""
        resolved = PACKAGED_PRICE_BOOK if path is None else Path(path)
        document = _read_document(resolved)
        raw_entries = document["entries"]
        return cls(
            entries=tuple(
                _parse_entry(item, index, resolved) for index, item in enumerate(raw_entries)
            )
        )

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_provider))

    def models(self, provider: str) -> tuple[str, ...]:
        """Configured model ids for one provider; empty when the provider is unknown."""
        return self._by_provider.get(provider, ())

    def entry(self, provider: str, model: str) -> PriceEntry:
        """Return the rate card for a pair, or raise. This never guesses a substitute."""
        _require_identifier(provider, "provider")
        _require_identifier(model, "model")
        found = self._by_key.get((provider, model))
        if found is None:
            raise UnknownModelError(self._unknown_message(provider, model))
        return found

    def price(
        self,
        provider: str,
        model: str,
        *,
        tokens_in: int,
        tokens_out: int,
        cached_in: int = 0,
        batch: bool = False,
    ) -> Decimal:
        """Exact cost for one request's usage, in the entry's currency."""
        entry = self.entry(provider, model)
        _require_token_count(tokens_in, "tokens_in")
        _require_token_count(tokens_out, "tokens_out")
        _require_token_count(cached_in, "cached_in")
        if not isinstance(batch, bool):
            raise PriceBookError(
                "batch must be True or False; fix: pass batch=True only for the asynchronous "
                "batch lane"
            )
        if cached_in > tokens_in:
            raise PriceBookError(
                f"cached_in {cached_in} exceeds tokens_in {tokens_in} for "
                f"{entry.provider}/{entry.model}; fix: cached_in counts the cache-read subset of "
                f"tokens_in — pass cached_in at most {tokens_in}, and check that your log mapping "
                f"is not reporting total prompt tokens as cached"
            )
        rate_in, rate_out = entry.rates(batch=batch)
        fresh_in = tokens_in - cached_in
        try:
            with localcontext() as ctx:
                ctx.prec = _MONEY_PRECISION
                ctx.traps[Inexact] = True
                billed = (
                    Decimal(fresh_in) * rate_in
                    + Decimal(cached_in) * entry.cached_input
                    + Decimal(tokens_out) * rate_out
                )
                return billed / TOKENS_PER_RATE_UNIT
        except Inexact as exc:
            raise PriceBookError(
                f"token counts for {entry.provider}/{entry.model} are too large to price exactly "
                f"({tokens_in} in, {tokens_out} out); fix: price usage in smaller windows and sum "
                f"the resulting Decimal values"
            ) from exc

    def staleness_warnings(self, *, today: date | None = None) -> tuple[StalenessWarning, ...]:
        """Entries older than the horizon. Library code reports; it never prints."""
        if today is None:
            today = date.today()
        elif type(today) is not date:
            raise PriceBookError(
                "today must be a datetime.date; fix: pass today=date.today() or omit the argument"
            )
        warnings = [
            StalenessWarning(
                entry=entry,
                age_days=entry.age_days(today),
                horizon_days=STALENESS_HORIZON_DAYS,
            )
            for entry in self.entries
            if entry.age_days(today) > STALENESS_HORIZON_DAYS
        ]
        warnings.sort(key=lambda warning: (-warning.age_days, warning.entry.provider))
        return tuple(warnings)

    def _unknown_message(self, provider: str, model: str) -> str:
        configured = self.models(provider)
        if not configured:
            known = ", ".join(self.providers)
            return (
                f"price book has no provider {provider!r}; configured providers: {known}; "
                f"fix: correct the provider name, or add {provider!r} to a price book file and "
                f"pass it with --price-book FILE"
            )
        candidates = difflib.get_close_matches(
            model, configured, n=_SUGGESTION_LIMIT, cutoff=0.4
        ) or list(configured[:_SUGGESTION_LIMIT])
        return (
            f"price book has no model {model!r} for provider {provider!r}; closest configured "
            f"models: {', '.join(candidates)}; fix: use one of those model ids, or add "
            f"{model!r} to a price book file and pass it with --price-book FILE. No price is "
            f"ever estimated from a similar model"
        )


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise PriceBookError(
            f"{name} must be a non-empty string; fix: set {name} to the exact id that appears in "
            f"your logs, for example provider='anthropic', model='claude-sonnet-4-6'"
        )
    if value != value.strip():
        raise PriceBookError(
            f"{name} {value!r} has surrounding whitespace; fix: set {name} to {value.strip()!r}"
        )


def _require_rate(value: Any, name: str, label: str) -> None:
    if not isinstance(value, Decimal):
        raise PriceBookError(
            f"price entry {label} field {name} is not exact; fix: build PriceEntry rates with "
            f"decimal.Decimal('<rate>') so money arithmetic stays exact"
        )
    if not value.is_finite():
        raise PriceBookError(
            f"price entry {label} field {name} is not finite; fix: set {name} to a finite decimal "
            f"rate per 1M tokens, for example '3.00'"
        )
    if value < 0:
        raise PriceBookError(
            f"price entry {label} field {name} is negative ({value}); fix: set {name} to a "
            f"non-negative rate per 1M tokens"
        )


def _require_absolute_url(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise PriceBookError(
            f"price entry {label} has no source_url; fix: set source_url to the provider pricing "
            f"page you read the rate from, for example "
            f"'https://platform.claude.com/docs/en/about-claude/pricing'"
        )
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise PriceBookError(
            f"price entry {label} has source_url {value!r}; fix: set source_url to an absolute "
            f"http(s) URL naming the page the rate was read from"
        )


def _require_token_count(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PriceBookError(
            f"{name} must be a whole number of tokens; fix: pass an int — provider usage counts "
            f"are integers, so convert with int(...) before pricing rather than carrying an "
            f"inexact number into a cost path"
        )
    if value < 0:
        raise PriceBookError(
            f"{name} is negative ({value}); fix: pass a non-negative token count; a missing count "
            f"is 0, not a negative sentinel"
        )


def _read_document(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PriceBookError(
            f"price book {path} does not exist; fix: create the file, or pass an existing path "
            f"with --price-book FILE, or omit the flag to use the packaged rate cards"
        ) from exc
    except OSError as exc:
        raise PriceBookError(
            f"price book {path} cannot be read ({exc.strerror}); fix: check the path and its "
            f"permissions, then retry"
        ) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PriceBookError(
            f"price book {path} is not valid JSON (line {exc.lineno}, column {exc.colno}); "
            f"fix: correct the JSON syntax at that position"
        ) from exc
    if not isinstance(document, Mapping):
        raise PriceBookError(
            f"price book {path} must be a JSON object; fix: wrap the rate cards as "
            f'{{"schema_version": {SCHEMA_VERSION}, "entries": [...]}}'
        )
    _require_exact_fields(document, _DOCUMENT_FIELDS, f"price book {path}")
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise PriceBookError(
            f"price book {path} has a non-integer schema_version; fix: set schema_version to "
            f"{SCHEMA_VERSION}"
        )
    if version != SCHEMA_VERSION:
        raise PriceBookError(
            f"price book {path} declares schema_version {version} but this build reads "
            f"{SCHEMA_VERSION}; fix: set schema_version to {SCHEMA_VERSION}, or use a build that "
            f"reads version {version}"
        )
    entries = document["entries"]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise PriceBookError(
            f"price book {path} field 'entries' must be a JSON array; fix: set 'entries' to a "
            f"list of rate card objects"
        )
    return document


def _require_exact_fields(mapping: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    keys = set(mapping)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing:
        raise PriceBookError(
            f"{label} is missing required field(s) {', '.join(missing)}; fix: add "
            f"{', '.join(missing)} to it"
        )
    if unexpected:
        raise PriceBookError(
            f"{label} has unknown field(s) {', '.join(unexpected)}; fix: delete "
            f"{', '.join(unexpected)}; schema_version {SCHEMA_VERSION} defines only "
            f"{', '.join(sorted(expected))}"
        )


def _parse_entry(raw: Any, index: int, path: Path) -> PriceEntry:
    label = f"price book {path} entry {index}"
    if not isinstance(raw, Mapping):
        raise PriceBookError(
            f"{label} is not a JSON object; fix: replace it with an object carrying "
            f"{', '.join(sorted(_ENTRY_FIELDS))}"
        )
    _require_exact_fields(raw, _ENTRY_FIELDS, label)
    return PriceEntry(
        provider=_parse_text(raw["provider"], "provider", label),
        model=_parse_text(raw["model"], "model", label),
        input=_parse_rate(raw["input"], "input", label),
        output=_parse_rate(raw["output"], "output", label),
        cached_input=_parse_rate(raw["cached_input"], "cached_input", label),
        batch_input=_parse_rate(raw["batch_input"], "batch_input", label),
        batch_output=_parse_rate(raw["batch_output"], "batch_output", label),
        currency=_parse_text(raw["currency"], "currency", label),
        effective_date=_parse_day(raw["effective_date"], label),
        source_url=_parse_text(raw["source_url"], "source_url", label),
    )


def _parse_text(value: Any, name: str, label: str) -> str:
    if not isinstance(value, str):
        raise PriceBookError(
            f"{label} field {name} must be a JSON string; fix: quote the value of {name}"
        )
    return value


def _parse_rate(value: Any, name: str, label: str) -> Decimal:
    if not isinstance(value, str):
        raise PriceBookError(
            f'{label} field {name} must be a quoted JSON string such as "3.00"; fix: quote the '
            f"rate — unquoted JSON numbers cannot be read back exactly, and every rate in a cost "
            f"path must be exact"
        )
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise PriceBookError(
            f"{label} field {name} is not a decimal rate ({value!r}); fix: set {name} to a "
            f'quoted decimal per 1M tokens, for example "3.00"'
        ) from exc
    _require_rate(rate, name, label)
    return rate


def _parse_day(value: Any, label: str) -> date:
    if not isinstance(value, str) or not _ISO_DAY.match(value):
        raise PriceBookError(
            f"{label} field effective_date must be an ISO-8601 day 'YYYY-MM-DD', got {value!r}; "
            f"fix: set effective_date to the day you read the rate from source_url, for example "
            f'"2026-08-27"'
        )
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PriceBookError(
            f"{label} field effective_date {value!r} is not a real calendar day; fix: set "
            f"effective_date to an existing date in 'YYYY-MM-DD' form"
        ) from exc
