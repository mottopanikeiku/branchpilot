from dataclasses import FrozenInstanceError, replace

import pytest

from branchpilot.integrity import (
    dataset_fingerprint,
    profile_rollouts,
    prompt_fingerprint,
    rollout_fingerprint,
    validate_disjoint,
    validate_unique,
)
from branchpilot.schema import Rollout, Sample


def _rollout(
    uid: str = "case-1",
    question: str = "What is 6 × 7?",
    gold: str = "42",
    samples: tuple[Sample, ...] | None = None,
    *,
    prompt_tokens: int = 9,
    metadata: dict | None = None,
) -> Rollout:
    return Rollout(
        uid=uid,
        question=question,
        gold=gold,
        samples=samples
        or (
            Sample("Reasoning. #### 42", "42", 5, -0.1),
            Sample("Alternative. #### 41", "41", 6, -0.5),
        ),
        prompt_tokens=prompt_tokens,
        metadata=metadata or {},
    )


def test_fingerprints_are_canonical_and_ignore_uid_and_metadata() -> None:
    first = Rollout.from_dict(
        {
            "uid": "first",
            "question": "  What is 6\t×\n7?  ",
            "gold": "42",
            "samples": [
                {
                    "text": "Reasoning. #### 42",
                    "answer": "42",
                    "token_count": 5,
                    "mean_logprob": -0.1,
                }
            ],
            "prompt_tokens": 9,
            "metadata": {"source": "a", "nested": {"x": 1, "y": 2}},
        }
    )
    second = Rollout.from_dict(
        {
            "metadata": {"nested": {"y": 999, "x": 0}, "source": "b"},
            "prompt_tokens": 9,
            "samples": [
                {
                    "mean_logprob": -0.1,
                    "token_count": 5,
                    "answer": "42",
                    "text": "Reasoning. #### 42",
                }
            ],
            "gold": "42",
            "question": "What is 6 × 7?",
            "uid": "second",
        }
    )

    assert prompt_fingerprint(first) == prompt_fingerprint(second)
    assert rollout_fingerprint(first) == rollout_fingerprint(second)
    assert len(prompt_fingerprint(first)) == 64
    assert len(rollout_fingerprint(first)) == 64


def test_prompt_and_rollout_fingerprints_change_for_question_and_exact_gold_changes() -> None:
    original = _rollout()
    changed_question = replace(original, question="What is 6 × 8?")
    changed_gold = replace(original, gold="42.0")

    assert prompt_fingerprint(original) != prompt_fingerprint(changed_question)
    assert prompt_fingerprint(original) != prompt_fingerprint(changed_gold)
    assert rollout_fingerprint(original) != rollout_fingerprint(changed_question)
    assert rollout_fingerprint(original) != rollout_fingerprint(changed_gold)


def test_prompt_fingerprint_does_not_normalize_gold_answer_text() -> None:
    original = _rollout(gold=" 42 ")

    assert prompt_fingerprint(original) != prompt_fingerprint(replace(original, gold="42"))


@pytest.mark.parametrize(
    "changed_sample",
    [
        Sample("Different text. #### 42", "42", 5, -0.1),
        Sample("Reasoning. #### 42", "42.0", 5, -0.1),
        Sample("Reasoning. #### 42", "42", 7, -0.1),
        Sample("Reasoning. #### 42", "42", 5, -0.2),
    ],
)
def test_rollout_fingerprint_covers_all_sample_content(changed_sample: Sample) -> None:
    original = _rollout()
    changed = replace(original, samples=(changed_sample, original.samples[1]))

    assert rollout_fingerprint(original) != rollout_fingerprint(changed)
    assert prompt_fingerprint(original) == prompt_fingerprint(changed)


def test_rollout_fingerprint_covers_prompt_tokens_and_sample_order() -> None:
    original = _rollout()

    assert rollout_fingerprint(original) != rollout_fingerprint(replace(original, prompt_tokens=10))
    assert rollout_fingerprint(original) != rollout_fingerprint(
        replace(original, samples=tuple(reversed(original.samples)))
    )


def test_dataset_fingerprint_is_deterministic_and_order_sensitive() -> None:
    first = _rollout(uid="first")
    second = _rollout(
        uid="second",
        question="What is 8 × 8?",
        gold="64",
        samples=(Sample("#### 64", "64", 4, -0.05),),
    )

    assert dataset_fingerprint([first, second]) == dataset_fingerprint(iter([first, second]))
    assert dataset_fingerprint([first, second]) != dataset_fingerprint([second, first])
    assert dataset_fingerprint([first, second]) == dataset_fingerprint(
        [replace(first, metadata={"path": "/tmp/input.jsonl"}), second]
    )


def test_validate_unique_reports_duplicate_uid() -> None:
    first = _rollout(uid="reused")
    second = _rollout(
        uid="reused",
        question="Different question",
        gold="9",
        samples=(Sample("#### 9", "9", 1),),
    )

    with pytest.raises(ValueError, match=r"duplicate UID 'reused'.*records 0.*and 1"):
        validate_unique([first, second])


def test_validate_unique_detects_duplicate_prompt_under_changed_uids() -> None:
    first = _rollout(uid="first")
    second = replace(
        first,
        uid="second",
        samples=(Sample("A different completion", None, 3),),
    )

    with pytest.raises(ValueError, match=r"duplicate prompt fingerprint.*'first'.*'second'"):
        validate_unique([first, second])


def test_validate_unique_reports_duplicate_rollout_under_changed_uids() -> None:
    first = _rollout(uid="first")
    second = replace(first, uid="second", metadata={"different": True})

    with pytest.raises(ValueError) as error:
        validate_unique([first, second])

    message = str(error.value)
    assert "duplicate prompt fingerprint" in message
    assert "duplicate rollout fingerprint" in message
    assert "UID 'first'" in message
    assert "UID 'second'" in message


def test_validate_unique_accepts_distinct_records() -> None:
    validate_unique(
        [
            _rollout(uid="first"),
            _rollout(
                uid="second",
                question="A distinct prompt",
                gold="1",
                samples=(Sample("#### 1", "1", 1),),
            ),
        ]
    )


@pytest.mark.parametrize(
    ("right", "expected"),
    [
        (
            _rollout(
                uid="left",
                question="No shared prompt",
                gold="0",
                samples=(Sample("#### 0", "0", 1),),
            ),
            "overlapping UID",
        ),
        (
            replace(
                _rollout(uid="left"),
                uid="prompt-copy",
                samples=(Sample("different", None, 1),),
            ),
            "overlapping prompt fingerprint",
        ),
        (
            replace(_rollout(uid="left"), uid="rollout-copy", metadata={"new": "metadata"}),
            "overlapping rollout fingerprint",
        ),
    ],
)
def test_validate_disjoint_detects_every_overlap_identity(right: Rollout, expected: str) -> None:
    with pytest.raises(ValueError) as error:
        validate_disjoint([_rollout(uid="left")], [right])

    assert expected in str(error.value)
    assert "left record 0" in str(error.value)
    assert "right record 0" in str(error.value)


def test_validate_disjoint_accepts_separate_splits() -> None:
    validate_disjoint(
        [_rollout(uid="left")],
        [
            _rollout(
                uid="right",
                question="Separate prompt",
                gold="7",
                samples=(Sample("#### 7", "7", 2),),
            )
        ],
    )


def test_profile_rollouts_reports_boundaries_coverage_and_provenance() -> None:
    first = _rollout(
        uid="first",
        question="Shared prompt",
        gold="1",
        samples=(Sample("one", "1", 1, -0.1),),
    )
    second = _rollout(
        uid="second",
        question="Other prompt",
        gold="2",
        samples=(Sample("none", None, 2), Sample("two", "2", 3)),
    )
    third = _rollout(
        uid="third",
        question="Shared prompt",
        gold="1",
        samples=(
            Sample("one again", "1", 4, -0.2),
            Sample("unparsed", None, 5, -0.3),
            Sample("three", "3", 6),
            Sample("still unparsed", None, 7),
        ),
    )

    profile = profile_rollouts([first, second, third])

    assert profile.record_count == 3
    assert profile.sample_count == 7
    assert profile.horizon_min == 1
    assert profile.horizon_median == 2.0
    assert profile.horizon_max == 4
    assert profile.prompt_uniqueness == pytest.approx(2 / 3)
    assert profile.parse_rate == pytest.approx(4 / 7)
    assert profile.logprob_coverage == pytest.approx(3 / 7)
    assert profile.completion_token_total == 28
    assert profile.dataset_fingerprint == dataset_fingerprint([first, second, third])
    assert profile.unique_uid_count == 3
    assert profile.unique_prompt_count == 2
    assert profile.unique_rollout_count == 3
    assert profile.to_dict()["dataset_fingerprint"] == profile.dataset_fingerprint

    with pytest.raises(FrozenInstanceError):
        profile.record_count = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dataset_fingerprint([]),
        lambda: validate_unique([]),
        lambda: validate_disjoint([], [_rollout()]),
        lambda: validate_disjoint([_rollout()], []),
        lambda: profile_rollouts([]),
    ],
)
def test_empty_inputs_are_rejected(operation) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        operation()
