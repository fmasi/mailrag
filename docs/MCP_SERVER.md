# MCP server — query your mail from any agent

`mailrag` ships a small [Model Context Protocol](https://modelcontextprotocol.io)
server (`src/mcp_server/`) that exposes the **existing** email-RAG query pipeline
over stdio, so an MCP client (Claude Desktop, Claude Code, or any MCP-capable
agent) can search and question your indexed mail without touching the internals.

It is a thin wrapper: retrieval is the same hybrid (bge-m3 dense + sparse, RRF)
searcher used by `mailrag ask`, and answers use the same grounded-answer path.
No ranking, fusion, or answering logic is reimplemented.

## Tools

| Tool | Signature | What it does |
|------|-----------|--------------|
| `search_email` | `search_email(query: str, top_k: int = 5)` | Hybrid retrieval expanded into whole, attributed email **threads**. Returns up to `top_k` rows `{thread_id, subject, num_emails, text}`. **No LLM call.** |
| `answer_question` | `answer_question(query: str, k: int = 3)` | The full RAG answer path: retrieve threads, then ground a single-LLM-call answer over the top-`k`. Returns `{answer, sources: [{thread_id, subject}]}`. |

Use `search_email` when the agent wants raw material to reason over itself
(cheap, no model call); use `answer_question` when you want mailrag to produce a
grounded natural-language answer.

## Running it

The server needs an already-indexed Qdrant collection — build one first with
`mailrag onboard` (see [`QUICKSTART.md`](QUICKSTART.md)). Then launch the server
either way:

```bash
# via the CLI verb (recommended)
mailrag mcp
mailrag mcp --collection work-rag --qdrant-url http://localhost:6333

# or as a module
python -m src.mcp_server
```

The process speaks MCP over stdio and blocks until the client disconnects.

## Configuration

Config mirrors `mailrag ask` and is resolved from flags/environment:

| Setting | Resolution (highest precedence first) |
|---------|---------------------------------------|
| Collection | `--collection` → `$MAILRAG_COLLECTION` → most recent onboarding manifest |
| Qdrant URL | `--qdrant-url` → `$MAILRAG_QDRANT_URL` → `$QDRANT_URL` → `http://localhost:6333` |

`MAILRAG_QDRANT_URL` exists as a dedicated override so a host-side server can
target `localhost` without inheriting a container-oriented `QDRANT_URL` from
`.env` (the [issue #29](https://github.com/fmasi/mailrag/issues/29) gotcha). The
answering LLM used by `answer_question` is the unified `Settings.llm` stack
configured through the usual `RAG_*` env vars (see [`BACKENDS.md`](BACKENDS.md)).

If no collection can be resolved, the tools return a clear error
(`no email collection configured: set MAILRAG_COLLECTION or run 'mailrag onboard'
first`) rather than crashing.

## Registering with an MCP client

Example Claude Desktop / Claude Code entry (`claude_desktop_config.json` or the
project MCP config):

```json
{
  "mcpServers": {
    "mailrag": {
      "command": "mailrag",
      "args": ["mcp"],
      "env": {
        "MAILRAG_COLLECTION": "work-rag",
        "MAILRAG_QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

(Point `command` at your environment's `mailrag` shim, or use
`"command": "python", "args": ["-m", "src.mcp_server"]` with an appropriate
working directory / `PYTHONPATH`.)
