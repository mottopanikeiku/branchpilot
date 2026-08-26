from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_BOXED_OPEN = r"\boxed{"
_HASH_ANSWER = re.compile(r"(?<!#)####(?!#)[ \t]*([^\r\n]*)")
_XML_TAG = re.compile(r"</?answer>", re.IGNORECASE)
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


def _explicit_answers(text: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []

    def add(position: int, raw: str) -> None:
        normalized = normalize_answer(raw)
        if normalized is not None:
            candidates.append((position, normalized))

    for match in _HASH_ANSWER.finditer(text):
        add(match.start(), match.group(1))

    xml_stack: list[tuple[int, int]] = []
    for tag in _XML_TAG.finditer(text):
        if tag.group()[1] != "/":
            xml_stack.append((tag.start(), tag.end()))
        elif xml_stack:
            start, content_start = xml_stack.pop()
            add(start, text[content_start : tag.start()])

    brace_stack: list[tuple[int, int] | None] = []
    index = 0
    while index < len(text):
        if text.startswith(_BOXED_OPEN, index):
            brace_stack.append((index, index + len(_BOXED_OPEN)))
            index += len(_BOXED_OPEN)
            continue
        if text[index] == "{":
            brace_stack.append(None)
        elif text[index] == "}" and brace_stack:
            boxed = brace_stack.pop()
            if boxed is not None:
                start, content_start = boxed
                add(start, text[content_start:index])
        index += 1

    return candidates


def extract_answer(text: str, *, strict: bool = True) -> str | None:
    """Extract the latest complete, explicitly delimited numeric answer."""
    explicit = _explicit_answers(text)
    if explicit:
        return max(explicit, key=lambda candidate: candidate[0])[1]
    if strict:
        return None

    candidates: list[tuple[int, str]] = []
    latex_spans: list[tuple[int, int]] = []
    for match in _LATEX_FRACTION.finditer(text):
        normalized = normalize_answer(match.group())
        if normalized is not None:
            candidates.append((match.start(), normalized))
            latex_spans.append(match.span())

    latex_index = 0
    for match in _NUMBER.finditer(text):
        while latex_index < len(latex_spans) and latex_spans[latex_index][1] <= match.start():
            latex_index += 1
        if (
            latex_index < len(latex_spans)
            and latex_spans[latex_index][0] <= match.start()
            and match.end() <= latex_spans[latex_index][1]
        ):
            continue
        normalized = normalize_answer(match.group())
        if normalized is not None:
            candidates.append((match.start(), normalized))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]
