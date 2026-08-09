# src/query/hyde.py
"""HyDE / query-expansion for retrieval (issue #16).

Bridge the query->document vocabulary gap: instead of searching with the user's
question, generate a hypothetical answer and search with that (it "looks like" a
real answer email, so it matches better). Pure logic (prompt + combine) lives here
and is unit-tested; the LLM call is a thin glue function.
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


def build_hyde_prompt_anchored(query: str) -> str:
    """Anchored HyDE prompt: reshape the question into answer-surface-form while
    KEEPING the query's real anchors (names, products, terms) and inventing nothing.

    The from-scratch :func:`build_hyde_prompt` hurts retrieval on an entity-rich
    corpus because it fabricates competing specifics (wrong names/dates) that drag
    the search toward sibling threads. This variant preserves the query's known-good
    signal verbatim and only adds the declarative "reply email" vocabulary that real
    answers use, with no new fabricated entities. See EXPERIMENTS §12.
    """
    return (
        "Rewrite the question below as the opening of a plausible answer email, as if "
        "you were replying with the answer in the relevant email thread. RULES:\n"
        "- Keep EVERY name, company, product, project, and specific term from the "
        "question EXACTLY as written - do not drop, rename, or paraphrase them.\n"
        "- Do NOT invent any new names, people, dates, or numbers that are not in the "
        "question. If a specific is unknown, write around it generically.\n"
        "- 1-3 sentences, answer-shaped (declarative, as a reply would phrase it), using "
        "the question's own vocabulary. No greeting, no preamble - just the answer text.\n\n"
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


def generate_hypothetical(client, model: str, query: str, anchored: bool = False) -> str:
    """One LLM call: the hypothetical answer text for `query` (stripped).

    anchored=True uses :func:`build_hyde_prompt_anchored` (preserve query anchors,
    invent nothing); the default keeps the original from-scratch HyDE prompt.
    """
    prompt = build_hyde_prompt_anchored(query) if anchored else build_hyde_prompt(query)
    return chat(client, model, prompt).strip()
