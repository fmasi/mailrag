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
from src.mcp_server import usage
from src.mcp_server.grep import grep_email as _grep_email
from src.onboard import latest_manifest_collection
from src.query.hybrid import build_hybrid_searcher

SERVER_NAME = "mailrag"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_ATTACH_STORE = "~/.mailrag/attachments"
VALID_MODES = ("hybrid", "dense", "sparse")

# search_email output bounding (issue #84): default to a compact snippet window
# around the query match plus metadata, never the full thread body. A hard cap
# guarantees a single call can never emit an unbounded payload.
DEFAULT_SEARCH_MAX_CHARS = 500
HARD_SEARCH_MAX_CHARS = 4000

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


def _thread_snippet(text: str, query: str, max_chars: int) -> str:
    """A bounded window of ``text`` centred on the first ``query`` term hit.

    Keeps the payload small (issue #84): when ``text`` fits within ``max_chars``
    it is returned whole; otherwise a ``max_chars``-wide window is centred on the
    earliest matching query token (falling back to the head of the text when no
    token matches), with ``…`` markers where content was elided.
    """
    text = text or ""
    if len(text) <= max_chars:
        return text
    # Find the earliest position any query token appears (case-insensitive).
    lowered = text.lower()
    best = -1
    for token in {t for t in query.lower().split() if len(t) >= 3}:
        pos = lowered.find(token)
        if pos != -1 and (best == -1 or pos < best):
            best = pos
    if best == -1:
        window = text[:max_chars]
        return window + "…"
    start = max(0, best - max_chars // 2)
    end = start + max_chars
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def _thread_meta(ctx) -> Dict[str, Any]:
    """Envelope metadata for a thread's constituent emails (bounded output).

    Surfaces subject/date/from/to/message-id and attachment filenames so an
    agent can decide whether to pull the full body via ``get_thread`` — without
    the tool ever emitting the whole thread text by default (issue #84).
    """
    emails = list(getattr(ctx, "emails", []) or [])
    first = emails[0] if emails else None
    last = emails[-1] if emails else None
    attachments: List[str] = []
    for e in emails:
        for name in getattr(e, "attachment_names", []) or []:
            if name not in attachments:
                attachments.append(name)
    return {
        "date": getattr(first, "date", "") if first else "",
        "last_date": getattr(last, "date", "") if last else "",
        "from": getattr(first, "sender", "") if first else "",
        "to": getattr(first, "to", "") if first else "",
        "message_ids": [getattr(e, "message_id", "") for e in emails],
        "attachment_names": attachments,
    }


def _thread_to_dict(
    ctx, query: str = "", max_chars: int = DEFAULT_SEARCH_MAX_CHARS
) -> Dict[str, Any]:
    """Serialize a ``ThreadContext`` into a **bounded** JSON result row.

    Returns a snippet window (``snippet``) plus metadata instead of the full
    thread body (issue #84). The full body is available on request via
    ``get_thread(thread_id, ...)`` / ``search_email(..., full=True)``.
    """
    row = {
        "thread_id": ctx.thread_id,
        "subject": ctx.subject,
        "num_emails": len(ctx.emails),
        "snippet": _thread_snippet(ctx.text, query, max_chars),
    }
    row.update(_thread_meta(ctx))
    return row


def _thread_to_full_dict(ctx) -> Dict[str, Any]:
    """Serialize a ``ThreadContext`` with the **full** thread text (opt-in path)."""
    row = {
        "thread_id": ctx.thread_id,
        "subject": ctx.subject,
        "num_emails": len(ctx.emails),
        "text": ctx.text,
    }
    row.update(_thread_meta(ctx))
    return row


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
    max_chars: int = DEFAULT_SEARCH_MAX_CHARS,
    full: bool = False,
    *,
    searcher=None,
) -> List[Dict[str, Any]]:
    """Retrieve the email threads most relevant to ``query`` (bounded output).

    Runs retrieval (``mode``: ``hybrid``/``dense``/``sparse``) and expands the hits
    into attributed threads, returning up to ``top_k`` **bounded** rows. Each row
    carries a ``snippet`` window (± context centred on the query match, capped by
    ``max_chars``) plus metadata — subject, date, from, to, message-ids and
    attachment names — instead of the full thread body (issue #84).

    Set ``full=True`` (or call ``get_thread``) to retrieve the whole thread text
    when you actually need it. ``max_chars`` is clamped to ``[1, {hard}]`` so a
    single call can never emit an unbounded payload.

    ``collection`` selects the corpus (defaults to the resolved collection).
    ``searcher`` is injectable for tests. Raises ``ValueError`` on invalid input or
    an unconfigured corpus.
    """.replace("{hard}", str(HARD_SEARCH_MAX_CHARS))
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    max_chars = min(max_chars, HARD_SEARCH_MAX_CHARS)
    searcher = searcher or get_searcher(collection, mode=mode)
    contexts = searcher.search_threads(query)
    if full:
        return [_thread_to_full_dict(c) for c in contexts[:top_k]]
    return [_thread_to_dict(c, query=query, max_chars=max_chars) for c in contexts[:top_k]]


def get_thread(
    thread_id: str,
    collection: Optional[str] = None,
    mode: str = "hybrid",
    *,
    searcher=None,
) -> Dict[str, Any]:
    """Fetch the **full** text of a single thread by ``thread_id`` (opt-in).

    The full-body companion to the bounded ``search_email`` (issue #84): given a
    ``thread_id`` returned by ``search_email``, return that thread's complete
    attributed text plus metadata.

    Resolution is an exact **key lookup** against the ``thread_id`` payload
    field, not a search. It used to re-run retrieval with the thread_id as the
    query and scan the hits for a matching id, which meant a successful lookup
    depended on an opaque message-id embedding near its own thread — it
    resolved only ~25% of the ids ``search_email`` had just returned (#109).

    ``mode`` is accepted for backward compatibility but no longer influences the
    result: a key fetch has no ranking to vary. It still selects which cached
    searcher is used, and all of them share one Qdrant client and collection.

    ``searcher`` is injectable for tests. Raises ``ValueError`` on a blank id, an
    unknown thread, or an unconfigured corpus.
    """
    # Normalise BEFORE validating, so the guards see the canonical form. Stored
    # thread_ids carry no angle brackets, but message_id does (`<abc@host>` vs
    # `abc@host`) and callers hand us ids from both fields, so normalise rather
    # than fail an otherwise-correct lookup on punctuation. Order matters: strip
    # after the emptiness check and `<>` survives it as a "real" id, only to come
    # back as `unknown thread ''` — sending the caller hunting for a thread that
    # was never named.
    thread_id = (thread_id or "").strip().strip("<>").strip()
    if not thread_id:
        raise ValueError("thread_id must be a non-empty string")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    searcher = searcher or get_searcher(collection, mode=mode)
    context = searcher.thread_by_id(thread_id)
    if context is None:
        raise ValueError(f"unknown thread {thread_id!r}")
    return _thread_to_full_dict(context)


def grep_email(
    pattern: str,
    collection: Optional[str] = None,
    max_matches: int = 50,
    regex: bool = False,
    max_files: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Literal / regex search over the raw email corpus — no embeddings (issue #82).

    Thin wrapper over :func:`src.mcp_server.grep.grep_email`: walks the raw ``.eml``
    corpus, decodes each body (QP/base64 + HTML→text), and returns matching lines
    plus message metadata (subject/from/to/date/message-id/attachment names). The
    escape hatch for needle hunts (numbers, IDs, emails, error strings) where dense
    retrieval is blind. Attachment *contents* are not searched (issue #80).

    The scan is bounded by matches, files **and** wall clock, and the result
    reports ``scanned``/``corpus_files``/``complete`` so an empty result can be
    read correctly — see the underlying function for the cost model.
    ``max_seconds=None`` here means "use the module default" (60s), not "no
    deadline"; pass the deadline explicitly to widen it.

    Raises ``ValueError`` on a blank pattern, an invalid regex, or a missing corpus.
    """
    kwargs: Dict[str, Any] = {"collection": collection, "max_matches": max_matches, "regex": regex}
    if max_files is not None:
        kwargs["max_files"] = max_files
    if max_seconds is not None:
        kwargs["max_seconds"] = max_seconds
    return _grep_email(pattern, **kwargs)


def answer_question(
    query: str,
    collection: Optional[str] = None,
    k: int = 3,
    mode: str = "hybrid",
    *,
    searcher=None,
    healthcheck: bool = True,
) -> Dict[str, Any]:
    """Answer ``query`` with a grounded RAG answer over retrieved email threads.

    Retrieves threads then grounds a single-LLM-call answer over the top-``k``
    (``answer_from_threads``). Returns ``{"answer": str, "sources": [...]}`` where
    each source is the ``thread_id``/``subject`` that fed the answer.

    Retrieval goes through the **same** ``get_searcher(collection, mode=...)`` /
    ``search_threads(query)`` path as ``search_email`` — same ``mode`` default and
    the result is materialised to a list *before* it is both fed to the LLM and
    sliced for ``sources`` — so for the same query+collection the sources here are
    always a subset of what ``search_email`` returns. (Previously a lazy/consumed
    context sequence or a mode mismatch could yield an empty ``sources`` while
    ``search_email`` returned hits.)

    Before calling the LLM it runs a one-shot :func:`src.llm.client.healthcheck`
    (unless ``healthcheck=False``) so a mis-configured endpoint / key fails with a
    clear, actionable message naming ``RAG_LLM_API_KEY`` — rather than a raw 401
    (issue #83). The healthcheck is scoped to this LLM path only; ``search_email``
    / ``grep_email`` stay usable even when the LLM is down.

    ``collection`` selects the corpus (defaults to the resolved collection). Raises
    ``ValueError`` on invalid input or an unconfigured corpus.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be >= 1")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if healthcheck:
        from src.llm.client import healthcheck as _llm_healthcheck

        _llm_healthcheck()
    searcher = searcher or get_searcher(collection, mode=mode)
    # Materialise once: the SAME list feeds the answer and the sources, so the
    # two can never diverge (e.g. a generator consumed by answer_from_threads
    # before sources is computed).
    contexts = list(searcher.search_threads(query))
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


def _require_populated_store(store) -> None:
    """Raise when the store has never been ingested, instead of returning nothing.

    An empty store answers every lookup with an empty list, which is
    indistinguishable from "this thread genuinely has no attachments" — so a
    caller reasonably concludes the mail has no attachments when in truth none
    have ever been ingested. That is the same failure the grep scan report
    exists to prevent: absence and not-looked-at must not share a
    representation. Indexing and sync do NOT populate this store (they extract
    attachment text for retrieval down a separate path), so a fully indexed
    corpus with an empty store is the expected state until `attachments build`
    is run once.
    """
    if store.count() == 0:
        raise ValueError(
            f"attachment store at {store.root!r} is empty — no attachments have been "
            "ingested, so every lookup returns nothing regardless of the thread. "
            "Populate it with `mailrag attachments build --profile <corpus.profile.json>`. "
            "(Indexing and sync extract attachment TEXT for search down a separate path "
            "and never write this store, so attachment content can be searchable while "
            "this store is still empty.)"
        )


def list_attachments(
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
    collection: Optional[str] = None,
    include_boilerplate: bool = False,
    *,
    store=None,
) -> List[Dict[str, Any]]:
    """List the attachments belonging to a thread or message.

    Parity with the CLI ``attachments list``: at least one of ``thread_id`` /
    ``message_id`` must be given. Recurring small inline images (signature logos,
    spacer pixels) are excluded unless ``include_boilerplate=True`` — on a real
    corpus they are 73% of all rows and would bury the actual documents.
    Returns a row per attachment
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
        metas = store.list_for(
            thread_id=thread_id, message_id=message_id, include_boilerplate=include_boilerplate
        )
        if not metas:
            _require_populated_store(store)
            if not include_boilerplate:
                # An empty result after filtering is a THIRD kind of nothing, and
                # it must not read like the other two. "No attachments here" and
                # "everything here was decoration" are different answers, and a
                # caller told the first will stop looking.
                raw = store.list_for(
                    thread_id=thread_id, message_id=message_id, include_boilerplate=True
                )
                if raw:
                    raise ValueError(
                        f"all {len(raw)} attachment(s) here are decoration (signature "
                        "images, disclaimer graphics, spacer pixels) — there is no document "
                        "attached. Pass include_boilerplate=true to see them anyway."
                    )
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
            # `from None` on the store-empty path: the KeyError is noise once we
            # know the real cause is an un-ingested store, and chaining it buries
            # the actionable message under "another exception occurred".
            try:
                _require_populated_store(store)
            except ValueError as empty:
                raise empty from None
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
    """Construct the ``MCPServer`` with the mailrag tools registered.

    Imported lazily so importing this module (e.g. for unit tests of the pure
    query/store functions) does not require the ``mcp`` SDK at import time.

    ``MCPServer`` is the SDK v2 name for what v1 called ``FastMCP``; v2 removed
    the old name outright rather than aliasing it, so this import is the whole
    of the v1 -> v2 migration for us. The decorator API below (``@server.tool``)
    and ``run(transport="stdio")`` are unchanged between the two.

    Imported from ``mcp.server`` rather than the deeper ``mcp.server.mcpserver``
    the migration guide shows: the shallower path is a deliberate re-export
    (``MCPServer`` is listed in ``mcp.server.__all__``), so it is the supported
    surface, while the deeper module is an internal layout detail that a v2.x
    patch could reorganise without a deprecation cycle.
    """
    from mcp.server import MCPServer

    server = MCPServer(SERVER_NAME)

    @server.tool(name="list_collections")
    @usage.instrument("list_collections")
    def _tool_list_collections() -> List[Dict[str, Any]]:
        """Discover the indexed email corpora available on the Qdrant instance.

        Returns one row per collection: ``{name, points_count, is_default}``. Use
        ``name`` as the ``collection`` argument of the other tools.
        """
        return list_collections()

    @server.tool(name="search_email")
    @usage.instrument("search_email")
    def _tool_search_email(
        query: str,
        collection: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        max_chars: int = DEFAULT_SEARCH_MAX_CHARS,
        full: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search the indexed email corpus; return relevant threads (bounded).

        Each hit is a compact ``snippet`` window around the match plus metadata
        (subject, date, from, to, message-ids, attachment names) — NOT the full
        thread body. Use ``get_thread`` or ``full=True`` for the whole text.

        Args:
            query: Natural-language search query.
            collection: Corpus to search (default: server-resolved collection).
            top_k: Maximum number of threads to return (default 5).
            mode: Retrieval leg — ``hybrid`` (default), ``dense`` or ``sparse``.
            max_chars: Snippet window size per hit (default 500, hard cap 4000).
            full: Return the full thread text instead of a snippet (default False).
        """
        return search_email(
            query,
            collection=collection,
            top_k=top_k,
            mode=mode,
            max_chars=max_chars,
            full=full,
        )

    @server.tool(name="get_thread")
    @usage.instrument("get_thread")
    def _tool_get_thread(
        thread_id: str,
        collection: Optional[str] = None,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """Fetch the FULL text of one thread by id (full-body companion to search).

        Args:
            thread_id: Thread id from a ``search_email`` hit.
            collection: Corpus to read (default: server-resolved collection).
            mode: Retrieval leg used to resolve the thread (default ``hybrid``).
        """
        return get_thread(thread_id, collection=collection, mode=mode)

    @server.tool(name="grep_email")
    @usage.instrument("grep_email")
    def _tool_grep_email(
        pattern: str,
        collection: Optional[str] = None,
        max_matches: int = 50,
        regex: bool = False,
        max_files: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Exact literal/regex search over the RAW email corpus — no embeddings.

        The escape hatch for needle hunts that retrieval is blind to: account
        numbers, order ids, email addresses, error strings — anything where the
        exact characters matter. Searches subject + decoded body (quoted-printable,
        base64, HTML→text). Attachment CONTENTS are not searched; use
        ``list_attachments`` / ``get_attachment`` for those.

        COST — read this before choosing arguments. There is no index: each call
        decodes raw ``.eml`` files one at a time (~2ms each, over a corpus that is
        routinely tens of thousands of messages). The scan stops the moment
        ``max_matches`` messages have matched, so a pattern that hits often returns
        in seconds — while a pattern that matches rarely, or not at all, walks the
        ENTIRE corpus. "Does X appear anywhere?" is the most expensive question you
        can ask here, and ``regex=True`` costs no more per message than a literal:
        what costs is how rarely the pattern hits.

        So: prefer a literal pattern; use ``max_matches=1`` for an existence check
        (it returns on the first hit); and bound the work with ``max_seconds`` /
        ``max_files`` rather than waiting — a partial answer you can describe beats
        a call that hangs until the client times out.

        Returns ``{matches, scanned, corpus_files, complete, stop_reason,
        elapsed_s, root}``. An empty ``matches`` means "not in this corpus" ONLY
        when ``complete`` is true. When it is false the needle is merely absent
        from the first ``scanned`` of ``corpus_files`` messages — report it that
        way, or re-run with a bigger budget. Do not turn a truncated scan into a
        confident "not found".

        Args:
            pattern: Literal string, or a regex when ``regex=True``.
            collection: Accepted for symmetry (grep is corpus-directory based).
            max_matches: Max matching messages (default 50, hard cap 500).
            regex: Treat ``pattern`` as a Python regex (default False = literal).
            max_files: Stop after scanning this many messages (default: no limit).
            max_seconds: Wall-clock budget in seconds (default 60, hard cap 900).
        """
        return grep_email(
            pattern,
            collection=collection,
            max_matches=max_matches,
            regex=regex,
            max_files=max_files,
            max_seconds=max_seconds,
        )

    @server.tool(name="answer_question")
    @usage.instrument("answer_question")
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
    @usage.instrument("list_attachments")
    def _tool_list_attachments(
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        collection: Optional[str] = None,
        include_boilerplate: bool = False,
    ) -> List[Dict[str, Any]]:
        """List the files attached to a thread — the way in to their contents.

        Attachment contents are INVISIBLE to ``search_email``, ``answer_question``
        and ``grep_email``: those index message bodies only. So when the real
        answer lives in a document somebody emailed — an invoice PDF, a
        spreadsheet of figures, a signed contract, a scanned letter, a ticket or
        boarding pass — body search finds the covering email and then looks as
        though the numbers simply are not there. They are; they are in the file.

        Workflow: ``search_email`` / ``grep_email`` to find the thread → this tool
        to list its attachments → ``get_attachment(sha256)`` to read one. A search
        hit's ``attachment_names`` tells you a document exists; this tool gets you
        the ``sha256`` you need to open it.

        Returns a row per attachment: ``{sha256, filename, mime, size, thread_id,
        message_id, inline}``.

        Signature logos, spacer pixels and footer badges are excluded by default:
        on a real corpus they are 73% of every row and would bury the documents
        you are looking for. They are identified by RECURRENCE (a small inline
        image reused across several messages), not by name or type — because
        ``image002.png`` is a 259-byte spacer in one message and a 12 MB pasted
        screenshot in another. One-off inline images are therefore kept: a pasted
        screenshot is content. Pass ``include_boilerplate=True`` for the raw list.

        Args:
            thread_id: Thread whose attachments to list (one of thread_id/message_id).
            message_id: Message whose attachments to list (one of thread_id/message_id).
            collection: Accepted for symmetry; the attachment store is corpus-wide.
            include_boilerplate: Include recurring inline decoration (default False).
        """
        return list_attachments(
            thread_id=thread_id,
            message_id=message_id,
            collection=collection,
            include_boilerplate=include_boilerplate,
        )

    @server.tool(name="get_attachment")
    @usage.instrument("get_attachment")
    def _tool_get_attachment(sha256: str, ocr: Optional[str] = None) -> Dict[str, Any]:
        """Read the extracted TEXT of one attachment — PDF, spreadsheet, doc, scan.

        The only way to see inside an emailed document. Extracts (or serves the
        cached) text for the attachment identified by ``sha256``, running OCR when
        the file is a scan or an image. Raw bytes are never returned over MCP. Get
        the ``sha256`` from ``list_attachments``.

        Returns ``{sha256, filename, mime, size, text, text_status}``. Always check
        ``text_status``: it reports when extraction failed or OCR was unavailable,
        so an empty ``text`` is not evidence that the document is blank — and
        should not be reported as such.

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
