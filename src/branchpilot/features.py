from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from branchpilot.schema import Rollout, Sample

FEATURE_NAMES = (
    "progress",
    "top_vote_share",
    "vote_margin",
    "vote_entropy",
    "answer_diversity",
    "parse_rate",
    "mean_logprob",
    "logprob_std",
    "logprob_coverage",
    "latest_agrees",
    "mean_completion_length",
    "completion_length_std",
    "prompt_tokens",
    "prompt_character_length",
    "prompt_numeric_density",
)


@dataclass(frozen=True, slots=True)
class PrefixState:
    features: np.ndarray
    majority_answer: str | None
    top_votes: int
    runner_up_votes: int


AnswerKey = tuple[bool, str | int]


def _answer_key(answer: str | None, index: int) -> AnswerKey:
    return (True, answer) if answer is not None else (False, index)


def observed_state(
    question: str,
    samples: Sequence[Sample],
    max_samples: int,
    prompt_tokens: int = 0,
) -> PrefixState:
    """Build a policy state from exactly the samples observed at inference time."""
    count = len(samples)
    if not question:
        raise ValueError("question cannot be empty")
    if count < 1 or count > max_samples:
        raise ValueError(f"observed sample count must be in [1, {max_samples}], got {count}")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens cannot be negative")

    votes: dict[AnswerKey, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        votes[_answer_key(sample.answer, index)].append(index)

    def vote_rank(item: tuple[AnswerKey, list[int]]) -> tuple[int, float, int]:
        _, indices = item
        logprobs = [samples[index].mean_logprob for index in indices]
        finite = [value for value in logprobs if value is not None and math.isfinite(value)]
        confidence = float(np.mean(finite)) if finite else -math.inf
        return (len(indices), confidence, -indices[0])

    ranked = sorted(votes.items(), key=vote_rank, reverse=True)
    winner, winner_indices = ranked[0]
    top_votes = len(winner_indices)
    runner_up_votes = len(ranked[1][1]) if len(ranked) > 1 else 0
    majority = str(winner[1]) if winner[0] else None

    probabilities = np.asarray([len(indices) / count for _, indices in ranked], dtype=np.float32)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    entropy_scale = math.log(max(2, count))

    finite_logprobs = np.asarray(
        [sample.mean_logprob for sample in samples if sample.mean_logprob is not None],
        dtype=np.float32,
    )
    mean_logprob = float(np.mean(finite_logprobs)) if finite_logprobs.size else 0.0
    logprob_std = float(np.std(finite_logprobs)) if finite_logprobs.size else 0.0
    lengths = np.asarray([sample.token_count for sample in samples], dtype=np.float32)
    latest_key = _answer_key(samples[-1].answer, count - 1)
    numeric_characters = len(re.findall(r"[\d+*/=-]", question))
    parsed = sum(sample.answer is not None for sample in samples)

    features = np.asarray(
        [
            count / max_samples,
            top_votes / count,
            (top_votes - runner_up_votes) / count,
            entropy / entropy_scale,
            len(ranked) / count,
            parsed / count,
            mean_logprob,
            logprob_std,
            finite_logprobs.size / count,
            float(latest_key == winner),
            float(np.mean(lengths)) / 256.0,
            float(np.std(lengths)) / 256.0,
            min(prompt_tokens / 512.0, 4.0),
            min(len(question) / 512.0, 4.0),
            numeric_characters / max(1, len(question)),
        ],
        dtype=np.float32,
    )
    return PrefixState(features, majority, top_votes, runner_up_votes)


def prefix_state(rollout: Rollout, count: int, max_samples: int | None = None) -> PrefixState:
    if count < 1 or count > len(rollout.samples):
        raise ValueError(f"count must be in [1, {len(rollout.samples)}], got {count}")
    horizon = max_samples or len(rollout.samples)
    return observed_state(
        rollout.question,
        rollout.samples[:count],
        horizon,
        rollout.prompt_tokens,
    )


def prefix_correct(rollout: Rollout, count: int) -> bool:
    answer = prefix_state(rollout, count).majority_answer
    if "answer_correctness" in rollout.metadata:
        return answer is not None and rollout.metadata["answer_correctness"].get(answer, False) is True
    return answer == rollout.gold
