from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
_XML_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?")
_LATEX_FRACTION = re.compile(r"\\frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}")


def normalize_answer(raw: str) -> str | None:
    """Canonicalize a numeric answer so decimal, fraction, and comma forms agree."""
    value = raw.strip().replace("$", "").replace(",", "")
    latex_fraction = _LATEX_FRACTION.fullmatch(value)
    if latex_fraction:
        value = f"{latex_fraction.group(1)}/{latex_fraction.group(2)}"
    value = value.rstrip(". ")
    try:
        if "/" in value:
            numerator, denominator = value.split("/", maxsplit=1)
            fraction = Fraction(int(numerator.strip()), int(denominator.strip()))
        else:
            fraction = Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer, preferring explicit answer delimiters."""
    candidates: list[str] = []
    for pattern in (_HASH_ANSWER, _XML_ANSWER, _BOXED):
        matches = pattern.findall(text)
        if matches:
            candidates.append(matches[-1])
    candidates.append(text)

    for candidate in candidates:
        latex_matches = _LATEX_FRACTION.findall(candidate)
        if latex_matches:
            numerator, denominator = latex_matches[-1]
            normalized = normalize_answer(f"{numerator}/{denominator}")
            if normalized is not None:
                return normalized
        numbers = _NUMBER.findall(candidate)
        if numbers:
            normalized = normalize_answer(numbers[-1])
            if normalized is not None:
                return normalized
    return None
