from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from branchpilot.features import PrefixState, observed_state
from branchpilot.policy import MAX_SAMPLES, Decision
from branchpilot.schema import Sample


@runtime_checkable
class StoppingStrategy(Protocol):
    """Structural interface for an observed-prefix stopping strategy."""

    max_samples: int

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision: ...


def _require_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_max_samples(value: object) -> int:
    maximum = _require_integer(value, "max_samples")
    if maximum < 1 or maximum > MAX_SAMPLES:
        raise ValueError(f"max_samples must be in [1, {MAX_SAMPLES}]")
    return maximum


def _validate_bounded_count(value: object, name: str, maximum: int) -> int:
    count = _require_integer(value, name)
    if count < 1 or count > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return count


def _state_and_horizon(
    strategy_max_samples: int,
    question: str,
    samples: Sequence[Sample],
    prompt_tokens: int,
    max_samples: int | None,
) -> tuple[tuple[Sample, ...], PrefixState, int]:
    horizon = (
        strategy_max_samples
        if max_samples is None
        else _require_integer(max_samples, "max_samples")
    )
    if horizon < 1 or horizon > strategy_max_samples:
        raise ValueError(f"max_samples must be in [1, {strategy_max_samples}]")
    observed = tuple(samples)
    state = observed_state(question, observed, horizon, prompt_tokens)
    return observed, state, horizon


def _rule_decision(state: PrefixState, count: int, stop: bool) -> Decision:
    # These are finite binary rule scores, not estimates from a learned value model.
    q_stop = 1.0 if stop else 0.0
    q_continue = 0.0 if stop else 1.0
    return Decision(
        action="stop" if stop else "continue",
        q_stop=q_stop,
        q_continue=q_continue,
        sample_count=count,
        majority_answer=state.majority_answer,
    )


@dataclass(frozen=True, slots=True)
class FixedStrategy:
    """Stop after a fixed number of observed samples, or at the session horizon."""

    samples: int
    max_samples: int

    def __post_init__(self) -> None:
        maximum = _validate_max_samples(self.max_samples)
        _validate_bounded_count(self.samples, "samples", maximum)

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        observed, state, horizon = _state_and_horizon(
            self.max_samples, question, samples, prompt_tokens, max_samples
        )
        count = len(observed)
        stop = count >= min(self.samples, horizon)
        return _rule_decision(state, count, stop)

    def to_spec(self) -> dict[str, object]:
        return {
            "type": "fixed",
            "samples": self.samples,
            "max_samples": self.max_samples,
        }


@dataclass(frozen=True, slots=True)
class VoteConfidenceStrategy:
    """Stop when the leading vote share reaches a fixed observed-prefix threshold."""

    threshold: float
    max_samples: int
    minimum: int = 2

    def __post_init__(self) -> None:
        maximum = _validate_max_samples(self.max_samples)
        if type(self.threshold) is not float:
            raise TypeError("threshold must be a float")
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be within (0, 1]")
        _validate_bounded_count(self.minimum, "minimum", maximum)

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        observed, state, horizon = _state_and_horizon(
            self.max_samples, question, samples, prompt_tokens, max_samples
        )
        count = len(observed)
        reached_threshold = count >= self.minimum and state.top_votes / count >= self.threshold
        return _rule_decision(state, count, reached_threshold or count >= horizon)

    def to_spec(self) -> dict[str, object]:
        return {
            "type": "vote_confidence",
            "threshold": self.threshold,
            "minimum": self.minimum,
            "max_samples": self.max_samples,
        }


@dataclass(frozen=True, slots=True)
class ConsecutiveAgreementStrategy:
    """Stop after a run of identical parsed answer keys, or at the horizon."""

    streak: int
    max_samples: int

    def __post_init__(self) -> None:
        maximum = _validate_max_samples(self.max_samples)
        _validate_bounded_count(self.streak, "streak", maximum)

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        observed, state, horizon = _state_and_horizon(
            self.max_samples, question, samples, prompt_tokens, max_samples
        )
        run = 0
        previous: str | None = None
        for sample in observed:
            if sample.answer is None:
                run = 0
                previous = None
            elif sample.answer == previous:
                run += 1
            else:
                run = 1
                previous = sample.answer
        count = len(observed)
        return _rule_decision(state, count, run >= self.streak or count >= horizon)

    def to_spec(self) -> dict[str, object]:
        return {
            "type": "consecutive_agreement",
            "streak": self.streak,
            "max_samples": self.max_samples,
        }


def _require_exact_keys(spec: dict[str, object], expected: frozenset[str]) -> None:
    actual = frozenset(spec)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"extra keys: {', '.join(extra)}")
        raise ValueError("invalid strategy spec (" + "; ".join(details) + ")")


def strategy_from_spec(spec: object) -> StoppingStrategy:
    """Construct a strategy from a strict, version-stable JSON object."""

    if type(spec) is not dict:
        raise TypeError("strategy spec must be a dictionary")
    kind = spec.get("type")
    if type(kind) is not str:
        raise TypeError("strategy spec type must be a string")

    if kind == "fixed":
        _require_exact_keys(spec, frozenset({"type", "samples", "max_samples"}))
        return FixedStrategy(samples=spec["samples"], max_samples=spec["max_samples"])
    if kind == "vote_confidence":
        _require_exact_keys(spec, frozenset({"type", "threshold", "minimum", "max_samples"}))
        return VoteConfidenceStrategy(
            threshold=spec["threshold"],
            minimum=spec["minimum"],
            max_samples=spec["max_samples"],
        )
    if kind == "consecutive_agreement":
        _require_exact_keys(spec, frozenset({"type", "streak", "max_samples"}))
        return ConsecutiveAgreementStrategy(streak=spec["streak"], max_samples=spec["max_samples"])
    raise ValueError(f"unknown strategy type: {kind}")
