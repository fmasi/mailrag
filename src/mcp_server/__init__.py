"""MCP (Model Context Protocol) server exposing mailrag's email-RAG query/read path.

A single, multi-collection server wrapping the *existing* stack — the hybrid
searcher (``src.query.hybrid.build_hybrid_searcher``), the grounded-answer path
(``src.llm.answer.answer_from_threads``) and the attachment store
(``src.attachments.store.AttachmentStore``) — as MCP tools so an MCP client (e.g.
Claude Desktop / Claude Code) can discover, search, question and read attachments
from any indexed email corpus. No retrieval/extraction logic is reimplemented.

Launch with ``python -m src.mcp_server`` or ``mailrag mcp``.
"""

from src.mcp_server.server import (
    answer_question,
    build_server,
    get_attachment,
    list_attachments,
    list_collections,
    resolve_attach_store,
    resolve_collection,
    resolve_qdrant_url,
    search_email,
    serve,
)

__all__ = [
    "answer_question",
    "build_server",
    "get_attachment",
    "list_attachments",
    "list_collections",
    "resolve_attach_store",
    "resolve_collection",
    "resolve_qdrant_url",
    "search_email",
    "serve",
]
