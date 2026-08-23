from branchpilot.answers import extract_answer, normalize_answer


def test_extracts_explicit_final_answer_before_reasoning_numbers() -> None:
    text = "First compute 12 + 18 = 30. Check it twice. #### 30"
    assert extract_answer(text) == "30"


def test_normalizes_equivalent_numeric_forms() -> None:
    assert normalize_answer("1,500.00") == "1500"
    assert normalize_answer("1.5") == "3/2"
    assert extract_answer(r"Therefore, the result is \boxed{\frac{3}{2}}.") == "3/2"


def test_returns_none_without_numeric_answer() -> None:
    assert extract_answer("I cannot determine the value.") is None
