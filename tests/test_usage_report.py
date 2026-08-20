"""Tests for reading the MCP usage log back.

The log is only worth writing if it gets read, and it gets read only if the
reading is one command rather than a fresh script each time. These pin the
aggregations that changed decisions in practice: which tools go unused, which
arguments callers pass, and which scans ran out of budget rather than finding
nothing.
"""

import json
import os
import tempfile
import unittest

from src.mcp_server.usage_report import load, render, summarise


def _log(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


class TestLoad(unittest.TestCase):
    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(load("/nonexistent/usage.jsonl"), [])

    def test_a_corrupt_line_does_not_lose_the_rest(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            fh.write('{"tool": "a", "ok": true}\n')
            fh.write("not json at all\n")
            fh.write('{"tool": "b", "ok": true}\n')
        try:
            self.assertEqual([r["tool"] for r in load(path)], ["a", "b"])
        finally:
            os.unlink(path)


class TestSummarise(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "tool": "search_email",
                "ok": True,
                "duration_ms": 100,
                "args": {"query": "x", "top_k": 5},
                "result_count": 5,
            },
            {
                "tool": "search_email",
                "ok": True,
                "duration_ms": 120,
                "args": {"query": "y"},
                "result_count": 0,
            },
            {
                "tool": "grep_email",
                "ok": True,
                "duration_ms": 60000,
                "args": {"pattern": "p"},
                "complete": False,
                "scanned": 900,
                "stop_reason": "deadline",
                "result_count": 0,
            },
            {
                "tool": "get_thread",
                "ok": False,
                "duration_ms": 10,
                "args": {"thread_id": "t"},
                "error": "ValueError: nope",
            },
        ]

    def test_counts_calls_and_failures_per_tool(self):
        s = summarise(self.rows)
        self.assertEqual(s["total"], 4)
        self.assertEqual(s["tools"]["search_email"], 2)
        self.assertEqual(s["failures"]["get_thread"], 1)

    def test_flags_slow_calls(self):
        self.assertEqual([r["tool"] for r in summarise(self.rows)["slow"]], ["grep_email"])

    def test_separates_truncated_scans_from_empty_results(self):
        # A scan that ran out of budget returned nothing YET; an empty search
        # returned nothing FULL STOP. Conflating them is the mistake the whole
        # scan-report design exists to prevent.
        s = summarise(self.rows)
        self.assertEqual(len(s["truncated"]), 1)
        self.assertEqual(len(s["empty"]), 2)

    def test_reports_which_arguments_callers_pass(self):
        args = summarise(self.rows)["args"]["search_email"]
        self.assertEqual(args["query"], 2)
        self.assertEqual(args["top_k"], 1)


class TestRender(unittest.TestCase):
    def test_empty_log_says_so_instead_of_printing_zeroes(self):
        self.assertIn("no MCP usage logged", render(summarise([])))

    def test_report_names_tools_shares_and_budget_exhaustion(self):
        rows = [
            {"tool": "search_email", "ok": True, "duration_ms": 5, "args": {}, "result_count": 1},
            {
                "tool": "grep_email",
                "ok": True,
                "duration_ms": 60000,
                "args": {"pattern": "p"},
                "complete": False,
                "scanned": 10,
                "stop_reason": "deadline",
            },
        ]
        text = render(summarise(rows))
        self.assertIn("search_email", text)
        self.assertIn("50%", text)
        self.assertIn("ran out of budget", text)

    def test_long_arguments_are_truncated(self):
        rows = [
            {"tool": "grep_email", "ok": True, "duration_ms": 60000, "args": {"pattern": "x" * 400}}
        ]
        line = [ln for ln in render(summarise(rows)).splitlines() if "60.0s" in ln][0]
        self.assertLess(len(line), 200)


if __name__ == "__main__":
    unittest.main()
