from pathlib import Path

import numpy as np
import pytest

from branchpilot.features import FEATURE_NAMES, observed_state, prefix_state
from branchpilot.schema import Rollout, Sample, read_jsonl, write_jsonl


def _rollout() -> Rollout:
    return Rollout(
        uid="case-1",
        question="What is 17 + 25?",
        gold="42",
        samples=(
            Sample("#### 41", "41", 12, -0.9),
            Sample("#### 42", "42", 10, -0.1),
            Sample("unparsed", None, 8, None),
        ),
        prompt_tokens=9,
    )


def test_tied_vote_uses_sequence_confidence() -> None:
    state = prefix_state(_rollout(), 2, 3)
    assert state.majority_answer == "42"
    assert state.top_votes == 1
    assert state.runner_up_votes == 1
    assert state.features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(state.features).all()


def test_tied_vote_without_confidence_is_label_invariant() -> None:
    first = observed_state(
        "question",
        (Sample("z", "z", 1), Sample("a", "a", 1)),
        2,
    )
    renamed = observed_state(
        "question",
        (Sample("a", "a", 1), Sample("z", "z", 1)),
        2,
    )
    assert first.majority_answer == "z"
    assert renamed.majority_answer == "a"


def test_unparsed_samples_never_form_false_consensus() -> None:
    rollout = Rollout(
        uid="unparsed",
        question="unknown",
        gold="1",
        samples=(Sample("x", None, 1), Sample("y", None, 1)),
    )
    state = prefix_state(rollout, 2)
    assert state.majority_answer is None
    assert state.top_votes == 1
    assert state.features[FEATURE_NAMES.index("parse_rate")] == 0.0
    assert state.features[FEATURE_NAMES.index("logprob_coverage")] == 0.0


def test_live_and_offline_state_construction_are_identical() -> None:
    rollout = _rollout()
    offline = prefix_state(rollout, 2, 3)
    live = observed_state(rollout.question, rollout.samples[:2], 3, rollout.prompt_tokens)
    np.testing.assert_array_equal(live.features, offline.features)
    assert live.majority_answer == offline.majority_answer


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [_rollout()])
    assert read_jsonl(path) == [_rollout()]


def test_numeric_json_answer_is_normalized_to_string() -> None:
    sample = Sample.from_dict(
        {"text": "#### 42", "answer": 42, "token_count": 2, "mean_logprob": -0.2}
    )
    assert sample.answer == "42"


def test_non_finite_logprob_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        Sample("#### 42", "42", 2, float("nan"))
