"""Tests for exact-content dedup (stdlib-only, host-runnable)."""

import unittest

from src.data import dedup


class TestDedupByContent(unittest.TestCase):
    def test_removes_exact_duplicates_preserving_first_order(self):
        self.assertEqual(
            dedup.dedup_by_content(["a", "b", "a", "c", "b"], key=lambda s: s),
            ["a", "b", "c"],
        )

    def test_is_whitespace_insensitive(self):
        # leading/trailing whitespace shouldn't make a chunk look unique
        self.assertEqual(
            dedup.dedup_by_content(["x", "  x  ", "y", "y\n"], key=lambda s: s),
            ["x", "y"],
        )

    def test_keeps_all_when_distinct(self):
        self.assertEqual(
            dedup.dedup_by_content(["a", "b", "c"], key=lambda s: s),
            ["a", "b", "c"],
        )

    def test_empty_input(self):
        self.assertEqual(dedup.dedup_by_content([], key=lambda s: s), [])

    def test_dedupes_objects_via_key(self):
        class Node:
            def __init__(self, t):
                self.text = t

        nodes = [Node("sig"), Node("body1"), Node("sig"), Node("body2")]
        kept = dedup.dedup_by_content(nodes, key=lambda n: n.text)
        self.assertEqual([n.text for n in kept], ["sig", "body1", "body2"])


if __name__ == "__main__":
    unittest.main()
