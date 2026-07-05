"""Stdio MCP server over the mailrag email-RAG QUERY/read surface.

A single, **multi-collection** server. Every tool is a thin wrapper over the
*existing* retrieval stack / attachment store — nothing about ranking, fusion,
answering or extraction is reimplemented here:

* ``list_collections()`` — discover the indexed corpora by asking Qdrant, marking
  which one is the resolved default.
* ``search_email(query, collection, top_k, mode)`` — hybrid (dense+sparse RRF),
  dense-only or sparse-only retrieval expanded into attributed email threads
  (``HybridSearcher.search_threads``). Returns ranked threads as JSON (no LLM).
* ``answer_question(query, collection, k)`` — the full RAG answer path: retrieve
  threads then ground an answer over the top-``k`` (``answer_from_threads``, one
  LLM call).
* ``list_attachments(thread_id, message_id, collection)`` — parity with the CLI
  ``attachments list``: the attachment rows for a thread or message.
* ``get_attachment(sha256, ocr)`` — parity with ``attachments get --text``: the
  EXTRACTED TEXT (+ metadata) for one attachment. Never returns raw bytes.

Every query tool takes an OPTIONAL ``collection`` so one running server can serve
any indexed corpus; searchers are cached per ``(collection, url, mode)``.

Configuration mirrors the CLI ``mailrag ask`` path:

* Collection — ``collection`` arg > ``MAILRAG_COLLECTION`` env > the most recent
  onboarding manifest (``latest_manifest_collection``). A clear error is raised
  when neither is available, so the server never crashes on a missing corpus.
* Qdrant URL — ``MAILRAG_QDRANT_URL`` (a dedicated override that wins so the
  container-oriented ``QDRANT_URL`` from ``.env`` is not inherited on the host,
  the issue-#29 gotcha), else ``QDRANT_URL``, else ``http://localhost:6333``.
* Attachment store — ``RAG_ATTACH_STORE`` env, else ``~/.mailrag/attachments``.

The LLM used by ``answer_question`` is the unified ``Settings.llm`` stack
configured through the usual ``RAG_*`` env vars (see ``src.llm.client``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.attachments.store import AttachmentStore
from src.llm.answer import answer_from_threads
from src.onboard import latest_manifest_collection
from src.query.hybrid import build_hybrid_searcher

SERVER_NAME = "mailrag"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_ATTACH_STORE = "~/.mailrag/attachments"
VALID_MODES = ("hybrid", "dense", "sparse")

# Cache built searchers by (collection, qdrant_url, mode) so repeated tool calls
# in one server session reuse the same Qdrant client / index wiring.
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
    ``QDRANT_URL`` from ``.env`` (issue #29).
    """
    return (
        qdrant_url
        or os.environ.get("MAILRAG_QDRANT_URL")
        or os.environ.get("QDRANT_URL")
        or DEFAULT_QDRANT_URL
    ).strip() or DEFAULT_QDRANT_URL


def resolve_attach_store(store: Optional[str] = None) -> str:
    """Resolve the attachment store root.

    Precedence: explicit arg > ``$RAG_ATTACH_STORE`` > ``~/.mailrag/attachments``
    (the same default as the CLI ``attachments`` verbs). The store is corpus-wide,
    shared across all collections.
    """
    return store or os.environ.get("RAG_ATTACH_STORE") or DEFAULT_ATTACH_STORE


def get_searcher(
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    mode: str = "hybrid",
    *,
    factory=build_hybrid_searcher,
):
    """Build (and cache) a searcher for the resolved collection/URL/mode.

    ``mode`` selects the retrieval leg: ``hybrid`` (dense+sparse RRF, default),
    ``dense`` (dense-only) or ``sparse`` (sparse-only) — all natively supported by
    ``build_hybrid_searcher``. ``factory`` is injectable for tests so no live
    Qdrant is needed. Searchers are cached per ``(collection, url, mode)``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    coll = resolve_collection(collection)
    url = resolve_qdrant_url(qdrant_url)
    key = (coll, url, mode)
    searcher = _SEARCHER_CACHE.get(key)
    if searcher is None:
        searcher = factory(coll, mode=mode, qdrant_url=url)
        _SEARCHER_CACHE[key] = searcher
    return searcher


def _thread_to_dict(ctx) -> Dict[str, Any]:
    """Serialize a ``ThreadContext`` into a JSON-friendly result row."""
    return {
        "thread_id": ctx.thread_id,
        "subject": ctx.subject,
        "num_emails": len(ctx.emails),
        "text": ctx.text,
    }


def list_collections(qdrant_url: Optional[str] = None, *, client=None) -> List[Dict[str, Any]]:
    """List the indexed corpora available on the resolved Qdrant instance.

    Asks Qdrant (``get_collections``) for every collection name and, for each,
    reports a cheap ``points_count`` when available (``None`` if that per-collection
    call fails). The row whose name matches the resolved default
    (``$MAILRAG_COLLECTION`` or the latest onboarding manifest) is flagged
    ``is_default``.

    ``client`` is injectable for tests. Raises ``ValueError`` with a clear message
    when Qdrant is unreachable, rather than crashing.
    """
    url = resolve_qdrant_url(qdrant_url)
    if client is None:
        from src.config.qdrant import get_qdrant_client

        try:
            client = get_qdrant_client(url=url)
        except Exception as exc:  # pragma: no cover - exercised via injected client
            raise ValueError(f"cannot reach Qdrant at {url}: {exc}") from exc
    try:
        collections = client.get_collections().collections
    except Exception as exc:
        raise ValueError(f"cannot list collections from Qdrant at {url}: {exc}") from exc

    try:
        default = os.environ.get("MAILRAG_COLLECTION") or latest_manifest_collection()
    except Exception:
        default = None

    rows: List[Dict[str, Any]] = []
    for desc in collections:
        name = desc.name
        points: Optional[int] = None
        try:
            points = client.get_collection(name).points_count
        except Exception:
            points = None
        rows.append({"name": name, "points_count": points, "is_default": name == default})
    return rows


def search_email(
    query: str,
    collection: Optional[str] = None,
    top_k: int = 5,
    mode: str = "hybrid",
    *,
    searcher=None,
) -> List[Dict[str, Any]]:
    """Retrieve the email threads most relevant to ``query``.

    Runs retrieval (``mode``: ``hybrid``/``dense``/``sparse``) and expands the hits
    into attributed threads, returning up to ``top_k`` of them as structured rows
    (``thread_id``, ``subject``, ``num_emails``, ``text``). No LLM call.

    ``collection`` selects the corpus (defaults to the resolved collection).
    ``searcher`` is injectable for tests. Raises ``ValueError`` on invalid input or
    an unconfigured corpus.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    searcher = searcher or get_searcher(collection, mode=mode)
    contexts = searcher.search_threads(query)
    return [_thread_to_dict(c) for c in contexts[:top_k]]


def answer_question(
    query: str,
    collection: Optional[str] = None,
    k: int = 3,
    *,
    searcher=None,
) -> Dict[str, Any]:
    """Answer ``query`` with a grounded RAG answer over retrieved email threads.

    Retrieves threads then grounds a single-LLM-call answer over the top-``k``
    (``answer_from_threads``). Returns ``{"answer": str, "sources": [...]}`` where
    each source is the ``thread_id``/``subject`` that fed the answer.

    ``collection`` selects the corpus (defaults to the resolved collection). Raises
    ``ValueError`` on invalid input or an unconfigured corpus.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be >= 1")
    searcher = searcher or get_searcher(collection)
    contexts = searcher.search_threads(query)
    answer = answer_from_threads(query, contexts, k=k)
    sources = [{"thread_id": c.thread_id, "subject": c.subject} for c in contexts[:k]]
    return {"answer": answer, "sources": sources}


def _meta_to_dict(meta) -> Dict[str, Any]:
    """Serialize an ``AttachmentMeta`` into a JSON-friendly result row."""
    return {
        "sha256": meta.sha256,
        "filename": meta.filename,
        "mime": meta.mime,
        "size": meta.size,
        "thread_id": meta.thread_id,
        "message_id": meta.message_id,
        "inline": meta.inline,
    }


def list_attachments(
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
    collection: Optional[str] = None,
    *,
    store=None,
) -> List[Dict[str, Any]]:
    """List the attachments belonging to a thread or message.

    Parity with the CLI ``attachments list``: at least one of ``thread_id`` /
    ``message_id`` must be given. Returns a row per attachment
    (``sha256``, ``filename``, ``mime``, ``size``, ``thread_id``, ``message_id``,
    ``inline``). The store is corpus-wide, so ``collection`` is accepted for API
    symmetry but not required. ``store`` is injectable for tests.

    Raises ``ValueError`` when neither identifier is supplied.
    """
    if not thread_id and not message_id:
        raise ValueError("one of thread_id or message_id is required")
    owns = store is None
    if store is None:
        store = AttachmentStore(resolve_attach_store())
    try:
        metas = store.list_for(thread_id=thread_id, message_id=message_id)
        return [_meta_to_dict(m) for m in metas]
    finally:
        if owns:
            store.close()


def get_attachment(
    sha256: str,
    ocr: Optional[str] = None,
    *,
    store=None,
) -> Dict[str, Any]:
    """Return the extracted TEXT (and metadata) for one attachment.

    Parity with the CLI ``attachments get --text``: extracts (or reads the cached)
    text for ``sha256`` and returns ``{sha256, filename, mime, size, text,
    text_status}``. ``ocr`` selects the extraction backend (``llm`` | ``tesseract``
    | ``cloud``), like the CLI ``--extractor`` flag; defaults to
    ``$RAG_ATTACH_EXTRACTOR`` or ``llm``. Raw bytes are never returned over MCP.

    ``store`` is injectable for tests. Raises ``ValueError`` on empty ``sha256`` or
    an unknown attachment.
    """
    if not sha256 or not sha256.strip():
        raise ValueError("sha256 must be a non-empty string")
    owns = store is None
    if store is None:
        store = AttachmentStore(resolve_attach_store())
    try:
        try:
            fetched = store.fetch(sha256, extractor=ocr)
        except KeyError as exc:
            raise ValueError(f"unknown attachment {sha256}") from exc
        return {
            "sha256": fetched["sha256"],
            "filename": fetched["filename"],
            "mime": fetched["mime"],
            "size": fetched["size"],
            "text": fetched["text"],
            "text_status": fetched["text_status"],
        }
    finally:
        if owns:
            store.close()


def build_server():
    """Construct the ``FastMCP`` server with the mailrag tools registered.

    Imported lazily so importing this module (e.g. for unit tests of the pure
    query/store functions) does not require the ``mcp`` SDK at import time.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(SERVER_NAME)

    @server.tool(name="list_collections")
    def _tool_list_collections() -> List[Dict[str, Any]]:
        """Discover the indexed email corpora available on the Qdrant instance.

        Returns one row per collection: ``{name, points_count, is_default}``. Use
        ``name`` as the ``collection`` argument of the other tools.
        """
        return list_collections()

    @server.tool(name="search_email")
    def _tool_search_email(
        query: str,
        collection: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """Search the indexed email corpus and return the most relevant threads.

        Args:
            query: Natural-language search query.
            collection: Corpus to search (default: server-resolved collection).
            top_k: Maximum number of threads to return (default 5).
            mode: Retrieval leg — ``hybrid`` (default), ``dense`` or ``sparse``.
        """
        return search_email(query, collection=collection, top_k=top_k, mode=mode)

    @server.tool(name="answer_question")
    def _tool_answer_question(
        query: str,
        collection: Optional[str] = None,
        k: int = 3,
    ) -> Dict[str, Any]:
        """Answer a question using a grounded RAG answer over the email corpus.

        Args:
            query: The question to answer.
            collection: Corpus to answer from (default: server-resolved collection).
            k: Number of retrieved threads to ground the answer on (default 3).
        """
        return answer_question(query, collection=collection, k=k)

    @server.tool(name="list_attachments")
    def _tool_list_attachments(
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List the attachments of an email thread or message.

        Args:
            thread_id: Thread whose attachments to list (one of thread_id/message_id).
            message_id: Message whose attachments to list (one of thread_id/message_id).
            collection: Accepted for symmetry; the attachment store is corpus-wide.
        """
        return list_attachments(thread_id=thread_id, message_id=message_id, collection=collection)

    @server.tool(name="get_attachment")
    def _tool_get_attachment(sha256: str, ocr: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted text and metadata for one attachment by sha256.

        Args:
            sha256: Content hash of the attachment (from ``list_attachments``).
            ocr: Extraction backend — ``llm`` | ``tesseract`` | ``cloud``
                (default: ``$RAG_ATTACH_EXTRACTOR`` or ``llm``).
        """
        return get_attachment(sha256, ocr=ocr)

    return server


def serve() -> None:
    """Run the MCP server over stdio (blocking). Entry point for the launcher."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
