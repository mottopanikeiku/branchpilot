from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from branchpilot.schema import Rollout

FEATURE_NAMES = (
    "progress",
    "top_vote_share",
    "vote_margin",
    "vote_entropy",
    "answer_diversity",
    "mean_logprob",
    "logprob_std",
    "latest_agrees",
    "mean_completion_length",
    "completion_length_std",
    "prompt_length",
    "prompt_numeric_density",
)


@dataclass(frozen=True, slots=True)
class PrefixState:
    features: np.ndarray
    majority_answer: str | None
    top_votes: int
    runner_up_votes: int


def _answer_key(answer: str | None, index: int) -> str:
    return answer if answer is not None else f"<unparsed:{index}>"


def prefix_state(rollout: Rollout, count: int, max_samples: int | None = None) -> PrefixState:
    if count < 1 or count > len(rollout.samples):
        raise ValueError(f"count must be in [1, {len(rollout.samples)}], got {count}")
    horizon = max_samples or len(rollout.samples)
    prefix = rollout.samples[:count]

    votes: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(prefix):
        votes[_answer_key(sample.answer, index)].append(index)

    def vote_rank(item: tuple[str, list[int]]) -> tuple[int, float, str]:
        key, indices = item
        logprobs = [prefix[index].mean_logprob for index in indices]
        finite = [value for value in logprobs if value is not None and math.isfinite(value)]
        confidence = float(np.mean(finite)) if finite else -math.inf
        return (len(indices), confidence, key)

    ranked = sorted(votes.items(), key=vote_rank, reverse=True)
    winner, winner_indices = ranked[0]
    top_votes = len(winner_indices)
    runner_up_votes = len(ranked[1][1]) if len(ranked) > 1 else 0
    majority = None if winner.startswith("<unparsed:") else winner

    probabilities = np.asarray([len(indices) / count for _, indices in ranked], dtype=np.float32)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    entropy_scale = math.log(max(2, count))

    logprobs = np.asarray(
        [sample.mean_logprob if sample.mean_logprob is not None else -12.0 for sample in prefix],
        dtype=np.float32,
    )
    lengths = np.asarray([sample.token_count for sample in prefix], dtype=np.float32)
    latest_key = _answer_key(prefix[-1].answer, count - 1)
    numeric_characters = len(re.findall(r"[\d+*/=-]", rollout.question))

    features = np.asarray(
        [
            count / horizon,
            top_votes / count,
            (top_votes - runner_up_votes) / count,
            entropy / entropy_scale,
            len(ranked) / count,
            float(np.mean(logprobs)),
            float(np.std(logprobs)),
            float(latest_key == winner),
            float(np.mean(lengths)) / 256.0,
            float(np.std(lengths)) / 256.0,
            min(len(rollout.question) / 512.0, 4.0),
            numeric_characters / max(1, len(rollout.question)),
        ],
        dtype=np.float32,
    )
    return PrefixState(features, majority, top_votes, runner_up_votes)


def prefix_correct(rollout: Rollout, count: int) -> bool:
    return prefix_state(rollout, count).majority_answer == rollout.gold
