from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from branchpilot.policy import Decision
from branchpilot.schema import Sample


class _ObservedPolicy(Protocol):
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


@dataclass(frozen=True, slots=True)
class PilotResult:
    answer: str | None
    sample_count: int
    samples: tuple[Sample, ...]
    decisions: tuple[Decision, ...]
    completion_tokens: int
    cost: float


class PilotSession:
    """Collect samples until an observed policy decides to stop."""

    def __init__(
        self,
        policy: _ObservedPolicy,
        question: str,
        cost: float,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            raise TypeError("cost must be a real number")
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and non-negative")
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
            raise TypeError("prompt_tokens must be an integer")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens cannot be negative")

        policy_max_samples = policy.max_samples
        if not isinstance(policy_max_samples, int) or isinstance(policy_max_samples, bool):
            raise TypeError("policy.max_samples must be an integer")
        if policy_max_samples < 1:
            raise ValueError("policy.max_samples must be positive")
        if max_samples is None:
            max_samples = policy_max_samples
        if not isinstance(max_samples, int) or isinstance(max_samples, bool):
            raise TypeError("max_samples must be an integer")
        if max_samples < 1 or max_samples > policy_max_samples:
            raise ValueError(f"max_samples must be in [1, {policy_max_samples}]")

        self.policy = policy
        self.question = question
        self.cost = float(cost)
        self.prompt_tokens = prompt_tokens
        self.max_samples = max_samples
        self._samples: list[Sample] = []
        self._decisions: list[Decision] = []

    @property
    def samples(self) -> tuple[Sample, ...]:
        return tuple(self._samples)

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def stopped(self) -> bool:
        return bool(self._decisions and self._decisions[-1].action == "stop")

    @property
    def should_continue(self) -> bool:
        return not self.stopped

    def observe(self, sample: Sample) -> Decision:
        if self.stopped:
            raise RuntimeError("cannot observe a sample after the session has stopped")
        if len(self._samples) >= self.max_samples:
            raise RuntimeError("policy failed to stop at the session horizon")

        self._samples.append(sample)
        decision = self.policy.decide_observed(
            self.question,
            self.samples,
            self.cost,
            prompt_tokens=self.prompt_tokens,
            max_samples=self.max_samples,
        )
        self._decisions.append(decision)
        return decision

    def run(self, sampler: Callable[[int], Sample]) -> PilotResult:
        while self.should_continue:
            self.observe(sampler(len(self._samples) + 1))
        return self.result()

    async def run_async(self, async_sampler: Callable[[int], Awaitable[Sample]]) -> PilotResult:
        while self.should_continue:
            self.observe(await async_sampler(len(self._samples) + 1))
        return self.result()

    def result(self) -> PilotResult:
        if not self.stopped:
            raise RuntimeError("result is unavailable before the session has stopped")
        final = self._decisions[-1]
        return PilotResult(
            answer=final.majority_answer,
            sample_count=len(self._samples),
            samples=self.samples,
            decisions=self.decisions,
            completion_tokens=sum(sample.token_count for sample in self._samples),
            cost=self.cost,
        )
