"""MCP (Model Context Protocol) server exposing mailrag's email-RAG query path.

Wraps the existing hybrid searcher (``src.query.hybrid.build_hybrid_searcher``)
and grounded-answer path (``src.llm.answer.answer_from_threads``) as MCP tools so
an MCP client (e.g. Claude Desktop / Claude Code) can search and question the
indexed email corpus. No retrieval logic is reimplemented here.

Launch with ``python -m src.mcp_server`` or ``mailrag mcp``.
"""

from src.mcp_server.server import (
    answer_question,
    build_server,
    resolve_collection,
    resolve_qdrant_url,
    search_email,
    serve,
)

__all__ = [
    "answer_question",
    "build_server",
    "resolve_collection",
    "resolve_qdrant_url",
    "search_email",
    "serve",
]
