from branchpilot.answers import extract_answer, normalize_answer


def test_normalizes_supported_numeric_forms() -> None:
    assert normalize_answer("42") == "42"
    assert normalize_answer("-1,500.00") == "-1500"
    assert normalize_answer("+1.25") == "5/4"
    assert normalize_answer("-6 / 8") == "-3/4"
    assert normalize_answer(r"\frac{-6}{8}") == "-3/4"


def test_extracts_all_supported_explicit_delimiters_and_forms() -> None:
    numeric_forms = (
        ("42", "42"),
        ("-1,500.00", "-1500"),
        ("+1.25", "5/4"),
        ("-6 / 8", "-3/4"),
    )
    for raw, expected in numeric_forms:
        assert extract_answer(f"Reasoning 12. #### {raw}") == expected
        assert extract_answer(f"Reasoning 12. <answer>{raw}</answer>") == expected
        assert extract_answer(rf"Reasoning 12. \boxed{{{raw}}}") == expected

    assert extract_answer(r"Reasoning 12. \boxed{\frac{-6}{8}}") == "-3/4"


def test_latest_valid_explicit_answer_wins_in_source_order() -> None:
    text = r"<answer>10</answer> then \boxed{20}, correction #### 30"
    assert extract_answer(text) == "30"

    reverse_delimiter_order = r"#### 10" + "\n" + r"then \boxed{20} and <answer>30</answer>"
    assert extract_answer(reverse_delimiter_order) == "30"


def test_strict_mode_rejects_bare_and_truncated_numeric_reasoning() -> None:
    rejected = (
        "First compute 12 + 18 = 30. Therefore the answer is 30.",
        "The calculation stops midway at 12 /",
        "Reasoning only, then ####",
        "Reasoning only, then <answer>42",
        r"Reasoning only, then \boxed{42",
    )
    for text in rejected:
        assert extract_answer(text) is None


def test_strict_mode_rejects_malformed_explicit_answers() -> None:
    rejected = (
        "#### answer is 42",
        "<answer>42 and 7</answer>",
        "<answer>42</answer",
        r"\boxed{42 and 7}",
        "#### 1/0",
    )
    for text in rejected:
        assert extract_answer(text) is None


def test_trailing_numbers_do_not_replace_a_complete_explicit_answer() -> None:
    text = "The result is #### 42\nLater verification used 99 samples."
    assert extract_answer(text) == "42"
    assert extract_answer(text, strict=False) == "42"


def test_permissive_mode_explicitly_enables_last_number_fallback() -> None:
    text = "First compute 10, then the undelimited result is -12.5."
    assert extract_answer(text) is None
    assert extract_answer(text, strict=False) == "-25/2"


def test_returns_none_without_numeric_answer_in_both_modes() -> None:
    text = "I cannot determine the value."
    assert extract_answer(text) is None
    assert extract_answer(text, strict=False) is None
