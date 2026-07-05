"""Integration: dedup_by_content collapses TextNodes by chunk text only.

Confirms the key used in the ingest path (get_content(MetadataMode.NONE))
treats identical body text from different senders as one. Imports llama_index,
so runs in the devcontainer.
"""

import unittest

from llama_index.core.schema import MetadataMode, TextNode

from src.data.dedup import dedup_by_content


def _text_key(node):
    return node.get_content(metadata_mode=MetadataMode.NONE)


class TestDedupTextNodes(unittest.TestCase):
    def test_same_text_different_metadata_collapses(self):
        nodes = [
            TextNode(text="Confidential disclaimer", metadata={"sender": "a@x"}),
            TextNode(text="unique body", metadata={"sender": "a@x"}),
            TextNode(text="Confidential disclaimer", metadata={"sender": "b@y"}),
        ]
        kept = dedup_by_content(nodes, key=_text_key)
        self.assertEqual(
            [_text_key(n) for n in kept],
            ["Confidential disclaimer", "unique body"],
        )


if __name__ == "__main__":
    unittest.main()
