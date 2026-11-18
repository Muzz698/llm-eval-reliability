"""Prompt templates and answer parsing. The templates differ only in surface format, not in what
is asked; a perfectly reliable test-taker would give the same answer under all of them."""
from __future__ import annotations

import re

from .items import LETTERS


def _subject(item: dict) -> str:
    return item["subject"].replace("_", " ")


# Every template ends with the same instruction, so templates differ only in layout and framing.
INSTRUCTION = "Answer with the letter of the correct option only."


def t_plain(item: dict, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
    return f"{item['question']}\n{opts}\n{INSTRUCTION}"


def t_mmlu(item: dict, choices: list[str]) -> str:
    """The framing of the original MMLU paper (Hendrycks et al., 2021), zero-shot."""
    opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
    return (f"The following is a multiple choice question about {_subject(item)}.\n\n"
            f"{item['question']}\n{opts}\n\n{INSTRUCTION}")


def t_paren(item: dict, choices: list[str]) -> str:
    opts = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(choices))
    return f"Question: {item['question']}\n\nOptions:\n{opts}\n\n{INSTRUCTION}"


TEMPLATES = {"plain": t_plain, "mmlu": t_mmlu, "paren": t_paren}

_START = re.compile(r"^\W*\(?\s*([ABCD])\s*(?:[).:*\]]|\n|$)")
_EXPLICIT = re.compile(r"answer(?: is)?\W*\(?([ABCD])\b", re.IGNORECASE)
_PAREN = re.compile(r"\(([ABCD])\)")


def parse_letter(text: str) -> str | None:
    """Return the chosen letter, or None if the response does not clearly name one.
    Deliberately strict: a bare 'A' inside a sentence ('A cell...') is not an answer."""
    text = text.strip()
    for pattern in (_START, _EXPLICIT, _PAREN):
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    return None
