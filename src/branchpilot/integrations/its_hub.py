from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from branchpilot.policy import Decision
from branchpilot.runtime import PilotSession
from branchpilot.schema import Sample

try:
    from its_hub.api import (
        AbstractLanguageModel,
        AbstractOrchestrator,
        AbstractScalingAlgorithm,
        AbstractScalingResult,
        ChatMessage,
        ChatMessages,
        GenerationUsage,
    )
except ImportError as error:
    raise ImportError(
        "branchpilot.integrations.its_hub requires the optional 'its-hub>=1.2.0' "
        "dependency; install BranchPilot with its its-hub integration dependencies"
    ) from error


ResponseConverter = Callable[[dict, int], Sample]


@dataclass
class BranchPilotScalingResult(AbstractScalingResult):
    """Full its_hub result for a sequential BranchPilot scaling run."""

    responses: list[dict]
    response_counts: Counter[str]
    selected_index: int
    usage: GenerationUsage
    decisions: tuple[Decision, ...]

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


class BranchPilotAlgorithm(AbstractScalingAlgorithm):
    """Expose an observed-prefix BranchPilot strategy as an its_hub algorithm."""

    def __init__(
        self,
        strategy: Any,
        cost: float,
        response_to_sample: ResponseConverter,
        orchestrator: AbstractOrchestrator,
    ) -> None:
        if not callable(getattr(strategy, "decide_observed", None)):
            raise TypeError("strategy must provide a callable decide_observed method")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise TypeError("cost must be a real number")
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and non-negative")
        if not callable(response_to_sample):
            raise TypeError("response_to_sample must be callable")
        if not callable(getattr(orchestrator, "agenerate", None)):
            raise TypeError("orchestrator must provide a callable agenerate method")

        self.strategy = strategy
        self.cost = float(cost)
        self.response_to_sample = response_to_sample
        self.orchestrator = orchestrator

    @property
    def max_samples(self) -> int:
        return self.strategy.max_samples

    def decide_observed(
        self,
        question: str,
        samples: Sequence[Sample],
        cost: float,
        *,
        prompt_tokens: int = 0,
        max_samples: int | None = None,
    ) -> Decision:
        return self.strategy.decide_observed(
            question,
            samples,
            cost,
            prompt_tokens=prompt_tokens,
            max_samples=max_samples,
        )

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | BranchPilotScalingResult:
        if type(budget) is not int:
            raise TypeError("budget must be an integer")
        if budget < 1:
            raise ValueError("budget must be positive")
        if type(return_response_only) is not bool:
            raise TypeError("return_response_only must be a boolean")

        policy_max_samples = self.max_samples
        if type(policy_max_samples) is not int:
            raise TypeError("strategy.max_samples must be an integer")
        if policy_max_samples < 1:
            raise ValueError("strategy.max_samples must be positive")
        horizon = min(budget, policy_max_samples)

        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        question = chat_messages.to_prompt()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("prompt_or_messages cannot be empty")

        usage = GenerationUsage()
        responses: list[dict] = []
        session: PilotSession | None = None

        while session is None or session.should_continue:
            prior_prompt_tokens = usage.prompt_tokens
            prior_completion_tokens = usage.completion_tokens
            generated = await self.orchestrator.agenerate(
                lm,
                chat_messages.to_batch(1),
                tools=tools,
                tool_choice=tool_choice,
                usage_accumulator=usage,
            )
            if not isinstance(generated, list):
                raise TypeError("orchestrator.agenerate must return a list")
            if len(generated) != 1:
                raise ValueError(
                    "orchestrator.agenerate must return exactly one response for each call"
                )
            response = generated[0]
            if not isinstance(response, dict):
                raise TypeError("orchestrator response must be a dictionary")

            completion_delta = _usage_delta(
                usage.completion_tokens,
                prior_completion_tokens,
                "completion_tokens",
            )
            prompt_delta = _usage_delta(
                usage.prompt_tokens,
                prior_prompt_tokens,
                "prompt_tokens",
            )
            sample = self.response_to_sample(response, completion_delta)
            if not isinstance(sample, Sample):
                raise TypeError("response_to_sample must return a branchpilot.schema.Sample")

            if session is None:
                session = PilotSession(
                    self,
                    question,
                    self.cost,
                    prompt_tokens=prompt_delta,
                    max_samples=horizon,
                )

            responses.append(response)
            session.observe(sample)

        pilot_result = session.result()
        response_counts = Counter(
            _answer_key(sample.answer, index) for index, sample in enumerate(pilot_result.samples)
        )
        if pilot_result.answer is None:
            selected_index = 0
        else:
            try:
                selected_index = next(
                    index
                    for index, sample in enumerate(pilot_result.samples)
                    if sample.answer == pilot_result.answer
                )
            except StopIteration as error:
                raise RuntimeError(
                    "strategy selected an answer absent from the observed responses"
                ) from error

        result = BranchPilotScalingResult(
            responses=responses,
            response_counts=response_counts,
            selected_index=selected_index,
            usage=usage,
            decisions=pilot_result.decisions,
        )
        return result.the_one if return_response_only else result


def _usage_delta(current: object, previous: object, name: str) -> int:
    if type(current) is not int:
        raise TypeError(f"usage {name} must be an integer")
    if type(previous) is not int:
        raise TypeError(f"usage {name} must be an integer")
    delta = current - previous
    if delta < 0:
        raise ValueError(f"usage {name} cannot decrease between generation calls")
    return delta


def _answer_key(answer: str | None, index: int) -> str:
    return answer if answer is not None else f"<unparsed:{index}>"
