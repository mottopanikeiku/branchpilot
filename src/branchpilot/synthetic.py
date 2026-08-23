from __future__ import annotations

import math
import random

from branchpilot.schema import Rollout, Sample


def make_synthetic_rollouts(
    count: int,
    max_samples: int = 8,
    seed: int = 17,
) -> list[Rollout]:
    """Generate a correlated reasoning environment for a zero-GPU quickstart."""
    if count < 1 or max_samples < 1:
        raise ValueError("count and max_samples must be positive")
    rng = random.Random(seed)
    rollouts: list[Rollout] = []
    for index in range(count):
        difficulty = rng.betavariate(1.7, 1.9)
        operands = 2 + int(difficulty * 7)
        values = [rng.randint(2, 30) for _ in range(operands)]
        gold_value = sum(values)
        gold = str(gold_value)
        clauses = " plus ".join(str(value) for value in values)
        question = f"A ledger contains {clauses} credits. What is the total?"

        probability_correct = 0.96 - 0.72 * difficulty
        common_error = str(gold_value + max(1, round(difficulty * operands)))
        samples: list[Sample] = []
        for sample_index in range(max_samples):
            correct = rng.random() < probability_correct
            if correct:
                answer = gold
            elif rng.random() < 0.72:
                answer = common_error
            else:
                answer = str(gold_value + rng.choice((-3, -2, -1, 1, 2, 3)))
            reasoning_steps = max(1, round(2 + difficulty * 8 + rng.gauss(0, 1.2)))
            token_count = max(8, 10 + reasoning_steps * 9 + rng.randint(-5, 8))
            confidence = probability_correct if correct else 1.0 - probability_correct
            mean_logprob = math.log(max(0.02, confidence)) + rng.gauss(0, 0.12)
            text = (
                f"I combine the ledger entries in {reasoning_steps} steps. "
                f"The final total is #### {answer}"
            )
            samples.append(Sample(text, answer, token_count, mean_logprob))

        rollouts.append(
            Rollout(
                uid=f"synthetic-{seed}-{index}",
                question=question,
                gold=gold,
                samples=tuple(samples),
                prompt_tokens=18 + operands * 3,
                metadata={"difficulty": round(difficulty, 6), "source": "synthetic-v1"},
            )
        )
    return rollouts
