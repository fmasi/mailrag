# tests/eval/test_flatten.py
import unittest
from dataclasses import dataclass
from typing import List

from src.eval.flatten import flatten_nodes, flatten_threads


class _Inner:
    def __init__(self, metadata, text):
        self.metadata = metadata
        self._text = text

    def get_content(self, *a, **k):
        return self._text


class _Node:  # mimics NodeWithScore
    def __init__(self, metadata, text):
        self.node = _Inner(metadata, text)
        self.metadata = metadata  # some callers read node.metadata directly


@dataclass
class _TEmail:
    message_id: str
    subject: str
    body: str
    summary: str = ""


@dataclass
class _TCtx:
    emails: List[_TEmail]


class FlattenNodesTest(unittest.TestCase):
    def test_dedup_by_message_id_keeps_first_rank(self):
        nodes = [
            _Node({"message_id": "a", "subject": "S-A", "summary": "sumA"}, "body-a"),
            _Node({"message_id": "b", "subject": "S-B"}, "body-b"),
            _Node({"message_id": "a", "subject": "S-A"}, "body-a-chunk2"),
        ]
        hits = flatten_nodes(nodes)
        self.assertEqual([h.message_id for h in hits], ["a", "b"])
        self.assertEqual(hits[0].subject, "S-A")
        self.assertEqual(hits[0].body, "body-a")
        self.assertEqual(hits[0].summary, "sumA")

    def test_skips_missing_message_id(self):
        nodes = [
            _Node({"subject": "no-id"}, "x"),
            _Node({"message_id": "z", "subject": "Z"}, "z-body"),
        ]
        hits = flatten_nodes(nodes)
        self.assertEqual([h.message_id for h in hits], ["z"])


class FlattenThreadsTest(unittest.TestCase):
    def test_flattens_thread_emails_in_order_deduped(self):
        ctxs = [
            _TCtx(emails=[_TEmail("a", "S", "terse"), _TEmail("b", "S", "substantive")]),
            _TCtx(emails=[_TEmail("b", "S", "substantive"), _TEmail("c", "S", "more")]),
        ]
        hits = flatten_threads(ctxs)
        self.assertEqual([h.message_id for h in hits], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
