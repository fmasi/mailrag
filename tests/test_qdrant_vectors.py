import unittest

import numpy as np

from src.ingest.qdrant_vectors import read_thread_vectors


class _Rec:
    def __init__(self, vector, payload):
        self.vector = vector
        self.payload = payload


class _FakeClient:
    """Mimics QdrantClient.scroll pagination over a fixed list of records."""
    def __init__(self, records, page=2):
        self._records = records
        self._page = page

    def scroll(self, collection_name, with_vectors=None, with_payload=None,
               limit=None, offset=None):
        start = offset or 0
        chunk = self._records[start:start + self._page]
        nxt = start + self._page
        return chunk, (nxt if nxt < len(self._records) else None)


class TestReadThreadVectors(unittest.TestCase):
    def test_mean_pools_by_thread_id(self):
        recs = [
            _Rec({"dense": [0.0, 0.0]}, {"thread_id": "t1"}),
            _Rec({"dense": [2.0, 0.0]}, {"thread_id": "t1"}),
            _Rec({"dense": [0.0, 4.0]}, {"thread_id": "t2"}),
        ]
        out = read_thread_vectors(_FakeClient(recs), "c")
        self.assertEqual(set(out), {"t1", "t2"})
        np.testing.assert_allclose(out["t1"], [1.0, 0.0])  # mean of the two
        np.testing.assert_allclose(out["t2"], [0.0, 4.0])

    def test_handles_bare_list_vector_and_skips_missing_thread_id(self):
        recs = [
            _Rec([1.0, 1.0], {"thread_id": "t1"}),
            _Rec({"dense": [3.0, 3.0]}, {}),           # no thread_id -> skipped
        ]
        out = read_thread_vectors(_FakeClient(recs), "c")
        self.assertEqual(set(out), {"t1"})


if __name__ == "__main__":
    unittest.main()
