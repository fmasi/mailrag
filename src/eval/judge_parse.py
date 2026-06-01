# src/eval/judge_parse.py
"""Relevance-judge prompt (shared rubric) + robust 0-3 grade parser."""
from __future__ import annotations

import re

RUBRIC = """
Grade how well the EMAIL answers the QUERY on a 0-3 scale:
3 = directly and fully answers the query
2 = relevant and useful, partially answers
1 = marginally related, does not really answer
0 = irrelevant
""".strip()


def build_judge_prompt(query: str, email_text: str) -> str:
    return (
        f"{RUBRIC}\n\n"
        f"QUERY:\n{query}\n\n"
        f"EMAIL:\n{email_text}\n\n"
        "Respond with ONLY the single integer grade (0, 1, 2, or 3)."
    )


_GRADE_KEYWORD = re.compile(r"grade\D{0,5}([0-3])", re.IGNORECASE)
_JSON_GRADE = re.compile(r'"?grade"?\s*[:=]\s*([0-3])', re.IGNORECASE)
_ANY_DIGIT = re.compile(r"[0-9]")


def parse_grade(text: str) -> int:
    """Extract a 0-3 grade from model output; clamp >3 to 3; default 0."""
    if text is None:
        return 0
    s = text.strip()
    for pat in (_JSON_GRADE, _GRADE_KEYWORD):
        m = pat.search(s)
        if m:
            return min(int(m.group(1)), 3)
    m = _ANY_DIGIT.search(s)
    if m:
        return min(int(m.group(0)), 3)
    return 0


ANSWER_RUBRIC = """
Grade how well the ANSWER answers the QUESTION, using the REFERENCE answer email
as the ground truth, on a 0-3 scale:
3 = fully correct and complete per the reference
2 = mostly correct; a minor omission or imprecision
1 = partially correct, or mixes correct and incorrect claims
0 = incorrect, unsupported, or "I don't know"
""".strip()


def build_answer_judge_prompt(query: str, answer: str, reference: str) -> str:
    """Prompt to grade a generated ANSWER (0-3) against the gold REFERENCE email."""
    return (
        f"{ANSWER_RUBRIC}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"REFERENCE (ground truth email):\n{reference}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Respond with ONLY the single integer grade (0, 1, 2, or 3)."
    )
