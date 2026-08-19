"""Tests for the MCP usage log — the record of which tools actually get used.

The log exists to answer "is this tool unused because it is unwanted, or because
it is badly described?", so the tests here care about three things: that a call
produces a faithful record, that the decorator does not disturb the function the
MCP SDK introspects to build a tool schema, and that a broken log can never
break a tool call.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from src.mcp_server import usage


def _env(**kwargs):
    """Patch the usage env vars, clearing any inherited ones."""
    base = {k: v for k, v in os.environ.items() if not k.startswith("MAILRAG_MCP_USAGE")}
    base.update(kwargs)
    return mock.patch.dict(os.environ, base, clear=True)


class _Log:
    """Context manager yielding a temp log path plus a reader for its entries."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="mcp_usage_")
        self.path = os.path.join(self.dir, "nested", "usage.jsonl")
        return self

    def entries(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def __exit__(self, *a):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


class TestResolveUsageLog(unittest.TestCase):
    def test_defaults_to_mailrag_home(self):
        with _env():
            self.assertEqual(usage.resolve_usage_log(), os.path.expanduser(usage.DEFAULT_USAGE_LOG))

    def test_env_override_wins(self):
        with _env(MAILRAG_MCP_USAGE_LOG="/tmp/somewhere/else.jsonl"):
            self.assertEqual(usage.resolve_usage_log(), "/tmp/somewhere/else.jsonl")

    def test_disabling_values_turn_logging_off(self):
        for value in ("", "0", "off", "none", "OFF", "False", "no"):
            with self.subTest(value=value):
                with _env(MAILRAG_MCP_USAGE_LOG=value):
                    self.assertIsNone(usage.resolve_usage_log())


class TestSummarizeArgs(unittest.TestCase):
    def test_omitted_args_are_dropped(self):
        # Only what the caller actually chose to pass is signal about the schema.
        with _env():
            self.assertEqual(
                usage.summarize_args({"query": "invoices", "collection": None, "top_k": 5}),
                {"query": "invoices", "top_k": 5},
            )

    def test_long_values_truncated(self):
        with _env():
            out = usage.summarize_args({"query": "x" * 500})
        self.assertLess(len(out["query"]), 500)
        self.assertTrue(out["query"].endswith("…"))

    def test_names_mode_redacts_values(self):
        with _env(MAILRAG_MCP_USAGE_ARGS="names"):
            out = usage.summarize_args({"query": "salary negotiation", "top_k": 5})
        self.assertEqual(out, {"query": "<str>", "top_k": "<int>"})
        self.assertNotIn("salary", json.dumps(out))

    def test_non_scalar_values_reduced_to_type(self):
        with _env():
            self.assertEqual(usage.summarize_args({"rows": [1, 2, 3]}), {"rows": "<list>"})


class TestRecord(unittest.TestCase):
    def test_writes_one_json_line_per_call(self):
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG=log.path):
            usage.record("search_email", {"query": "acme"}, duration_s=0.25, ok=True, result=[1, 2])
            usage.record("get_thread", {"thread_id": "t1"}, duration_s=0.1, ok=True, result={})
            entries = log.entries()
        self.assertEqual([e["tool"] for e in entries], ["search_email", "get_thread"])
        self.assertEqual(entries[0]["result_count"], 2)
        self.assertEqual(entries[0]["duration_ms"], 250.0)
        self.assertTrue(entries[0]["ok"])
        self.assertIn("ts", entries[0])

    def test_counts_rows_inside_a_wrapping_dict(self):
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG=log.path):
            usage.record("grep_email", {}, duration_s=1.0, ok=True, result={"matches": [1, 2, 3]})
            usage.record("answer_question", {}, duration_s=1.0, ok=True, result={"sources": [1]})
            entries = log.entries()
        self.assertEqual([e["result_count"] for e in entries], [3, 1])

    def test_records_the_grep_scan_report(self):
        # A truncated scan is the thing worth spotting in the log later.
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG=log.path):
            usage.record(
                "grep_email",
                {"pattern": "x"},
                duration_s=60.0,
                ok=True,
                result={
                    "matches": [],
                    "complete": False,
                    "scanned": 4138,
                    "stop_reason": "deadline",
                },
            )
            entry = log.entries()[0]
        self.assertFalse(entry["complete"])
        self.assertEqual(entry["scanned"], 4138)
        self.assertEqual(entry["stop_reason"], "deadline")

    def test_records_failures_with_the_error(self):
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG=log.path):
            usage.record("get_thread", {}, duration_s=0.0, ok=False, error=ValueError("unknown"))
            entry = log.entries()[0]
        self.assertFalse(entry["ok"])
        self.assertIn("ValueError", entry["error"])

    def test_writes_nothing_when_disabled(self):
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG="off"):
            usage.record("search_email", {}, duration_s=0.1, ok=True, result=[])
        self.assertEqual(log.entries(), [])

    def test_unwritable_path_is_swallowed(self):
        # A broken log is worth less than a working search: never raise.
        with _env(MAILRAG_MCP_USAGE_LOG="/proc/definitely/not/writable.jsonl"):
            usage.record("search_email", {}, duration_s=0.1, ok=True, result=[])


class TestInstrument(unittest.TestCase):
    def test_logs_a_successful_call_and_returns_its_result(self):
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG=log.path):

            @usage.instrument("search_email")
            def tool(query: str, top_k: int = 5):
                return [{"thread_id": "t1"}]

            self.assertEqual(tool("acme"), [{"thread_id": "t1"}])
            entry = log.entries()[0]
        self.assertEqual(entry["tool"], "search_email")
        # Defaults are bound too, so the log shows the values the tool ran with.
        self.assertEqual(entry["args"], {"query": "acme", "top_k": 5})
        self.assertEqual(entry["result_count"], 1)

    def test_logs_a_failure_and_re_raises_unchanged(self):
        with _Log() as log, _env(MAILRAG_MCP_USAGE_LOG=log.path):

            @usage.instrument("get_thread")
            def tool(thread_id: str):
                raise ValueError("unknown thread 't9'")

            with self.assertRaises(ValueError) as ctx:
                tool("t9")
            entry = log.entries()[0]
        self.assertIn("unknown thread", str(ctx.exception))
        self.assertFalse(entry["ok"])
        self.assertIn("ValueError", entry["error"])

    def test_preserves_the_signature_the_mcp_sdk_introspects(self):
        # The SDK builds each tool's JSON schema from the signature, annotations
        # and docstring of the registered callable. If the decorator replaced any
        # of those with (*args, **kwargs), every tool would silently lose its
        # parameters — so assert them directly rather than trusting functools.
        import inspect

        @usage.instrument("grep_email")
        def tool(pattern: str, regex: bool = False) -> list:
            """Docstring the agent reads."""
            return []

        self.assertEqual(list(inspect.signature(tool).parameters), ["pattern", "regex"])
        self.assertEqual(tool.__doc__, """Docstring the agent reads.""")
        self.assertEqual(tool.__name__, "tool")
        self.assertEqual(inspect.signature(tool).parameters["regex"].default, False)

    def test_a_broken_log_does_not_break_the_call(self):
        with _env(MAILRAG_MCP_USAGE_LOG="/proc/definitely/not/writable.jsonl"):

            @usage.instrument("search_email")
            def tool():
                return ["ok"]

            self.assertEqual(tool(), ["ok"])


if __name__ == "__main__":
    unittest.main()
