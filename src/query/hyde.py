# src/query/hyde.py
"""HyDE / query-expansion for retrieval (issue #16).

Bridge the query->document vocabulary gap: instead of searching with the user's
question, generate a hypothetical answer and search with that (it "looks like" a
real answer email, so it matches better). Pure logic (prompt + combine) lives here
and is unit-tested; the LLM call is a thin glue function. See
docs/superpowers/specs/2026-05-30-hyde-query-side-design.md.
"""
from __future__ import annotations

from src.llm.client import chat


def build_hyde_prompt(query: str) -> str:
    """Prompt the model to write a short, plausible answer email to the question."""
    return (
        "Write a short, plausible email that directly ANSWERS the question below, as if "
        "you were the person replying in the relevant email thread. 2-4 sentences. State "
        "concrete specifics (names, dates, decisions) a real answer would contain. No "
        "greeting, no preamble, no explanation - just the answer text.\n\n"
        f"QUESTION:\n{query}"
    )


def combine_query(query: str, hypothetical: str, mode: str) -> str:
    """Build the search string from the query + its hypothetical answer.

    mode="pure"    -> the hypothetical alone
    mode="augment" -> query + newline + hypothetical
    empty/whitespace hypothetical, or any other mode -> the raw query (fail-safe).
    """
    h = (hypothetical or "").strip()
    if not h:
        return query
    if mode == "pure":
        return h
    if mode == "augment":
        return f"{query}\n{h}"
    return query


def generate_hypothetical(client, model: str, query: str) -> str:
    """One LLM call: the hypothetical answer text for `query` (stripped)."""
    return chat(client, model, build_hyde_prompt(query)).strip()
