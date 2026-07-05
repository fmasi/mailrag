"""Stdio MCP server over the mailrag email-RAG query pipeline.

The server exposes two tools, both thin wrappers over the *existing* retrieval
stack — nothing about ranking, fusion or answering is reimplemented here:

* ``search_email(query, top_k)`` — hybrid (dense+sparse RRF) retrieval expanded
  into attributed email threads (``HybridSearcher.search_threads``). Returns the
  ranked threads as structured JSON (no LLM call).
* ``answer_question(query, k)`` — the full RAG answer path: retrieve threads then
  ground an answer over the top-``k`` with ``answer_from_threads`` (one LLM call).

Configuration mirrors the CLI ``mailrag ask`` path:

* Collection — ``MAILRAG_COLLECTION`` env, else the most recent onboarding
  manifest (``latest_manifest_collection``). A clear error is raised when neither
  is available, so the server never crashes on a missing corpus.
* Qdrant URL — ``MAILRAG_QDRANT_URL`` (a dedicated override that wins so the
  container-oriented ``QDRANT_URL`` from ``.env`` is not inherited on the host,
  the issue-#29 gotcha), else ``QDRANT_URL``, else ``http://localhost:6333``.

The LLM used by ``answer_question`` is the unified ``Settings.llm`` stack
configured through the usual ``RAG_*`` env vars (see ``src.llm.client``).
"""

from __future__ import annotations

import os
from typing import List, Optional

from src.llm.answer import answer_from_threads
from src.onboard import latest_manifest_collection
from src.query.hybrid import build_hybrid_searcher

SERVER_NAME = "mailrag"
DEFAULT_QDRANT_URL = "http://localhost:6333"

# Cache built searchers by (collection, qdrant_url) so repeated tool calls in one
# server session reuse the same Qdrant client / index wiring.
_SEARCHER_CACHE: dict = {}


def resolve_collection(collection: Optional[str] = None) -> str:
    """Resolve the Qdrant collection to query.

    Precedence: explicit arg > ``$MAILRAG_COLLECTION`` > latest onboarding
    manifest. Raises ``ValueError`` with an actionable message when none is
    available (so the MCP client sees a clear error rather than a crash).
    """
    coll = collection or os.environ.get("MAILRAG_COLLECTION") or latest_manifest_collection()
    if not coll:
        raise ValueError(
            "no email collection configured: set MAILRAG_COLLECTION or run "
            "`mailrag onboard` first to build one"
        )
    return coll


def resolve_qdrant_url(qdrant_url: Optional[str] = None) -> str:
    """Resolve the Qdrant URL.

    Precedence: explicit arg > ``$MAILRAG_QDRANT_URL`` > ``$QDRANT_URL`` >
    ``http://localhost:6333``. ``MAILRAG_QDRANT_URL`` exists so a host-side MCP
    server can target ``localhost`` without inheriting a container-oriented
    ``QDRANT_URL`` (issue #29).
    """
    return (
        qdrant_url
        or os.environ.get("MAILRAG_QDRANT_URL")
        or os.environ.get("QDRANT_URL")
        or DEFAULT_QDRANT_URL
    ).strip() or DEFAULT_QDRANT_URL


def get_searcher(
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    *,
    factory=build_hybrid_searcher,
):
    """Build (and cache) a hybrid searcher for the resolved collection/URL.

    ``factory`` is injectable for tests so no live Qdrant is needed. Reuses the
    existing ``build_hybrid_searcher`` wiring in hybrid mode (dense+sparse RRF).
    """
    coll = resolve_collection(collection)
    url = resolve_qdrant_url(qdrant_url)
    key = (coll, url)
    searcher = _SEARCHER_CACHE.get(key)
    if searcher is None:
        searcher = factory(coll, mode="hybrid", qdrant_url=url)
        _SEARCHER_CACHE[key] = searcher
    return searcher


def _thread_to_dict(ctx) -> dict:
    """Serialize a ``ThreadContext`` into a JSON-friendly result row."""
    return {
        "thread_id": ctx.thread_id,
        "subject": ctx.subject,
        "num_emails": len(ctx.emails),
        "text": ctx.text,
    }


def search_email(query: str, top_k: int = 5, *, searcher=None) -> List[dict]:
    """Retrieve the email threads most relevant to ``query``.

    Runs hybrid retrieval and expands the hits into attributed threads, returning
    up to ``top_k`` of them as structured rows (``thread_id``, ``subject``,
    ``num_emails``, ``text``). No LLM call. ``searcher`` is injectable for tests.

    Raises ``ValueError`` on invalid input or an unconfigured corpus.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    searcher = searcher or get_searcher()
    contexts = searcher.search_threads(query)
    return [_thread_to_dict(c) for c in contexts[:top_k]]


def answer_question(query: str, k: int = 3, *, searcher=None) -> dict:
    """Answer ``query`` with a grounded RAG answer over retrieved email threads.

    Retrieves threads then grounds a single-LLM-call answer over the top-``k``
    (``answer_from_threads``). Returns ``{"answer": str, "sources": [...]}`` where
    each source is the ``thread_id``/``subject`` that fed the answer.

    Raises ``ValueError`` on invalid input or an unconfigured corpus.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be >= 1")
    searcher = searcher or get_searcher()
    contexts = searcher.search_threads(query)
    answer = answer_from_threads(query, contexts, k=k)
    sources = [{"thread_id": c.thread_id, "subject": c.subject} for c in contexts[:k]]
    return {"answer": answer, "sources": sources}


def build_server():
    """Construct the ``FastMCP`` server with the mailrag tools registered.

    Imported lazily so importing this module (e.g. for unit tests of the pure
    query functions) does not require the ``mcp`` SDK at import time.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(SERVER_NAME)

    @server.tool(name="search_email")
    def _tool_search_email(query: str, top_k: int = 5) -> List[dict]:
        """Search the indexed email corpus and return the most relevant threads.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of threads to return (default 5).
        """
        return search_email(query, top_k)

    @server.tool(name="answer_question")
    def _tool_answer_question(query: str, k: int = 3) -> dict:
        """Answer a question using a grounded RAG answer over the email corpus.

        Args:
            query: The question to answer.
            k: Number of retrieved threads to ground the answer on (default 3).
        """
        return answer_question(query, k)

    return server


def serve() -> None:
    """Run the MCP server over stdio (blocking). Entry point for the launcher."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
