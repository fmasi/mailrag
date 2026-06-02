import unittest
from src.ingest.profile import percentiles, suggest_chunk_size


class TestProfile(unittest.TestCase):
    def test_percentiles_nearest_rank(self):
        vals = list(range(1, 101))  # 1..100
        p = percentiles(vals)
        self.assertEqual(p[50], 51)
        self.assertEqual(p[100], 100)

    def test_suggest_rounds_p90_up_to_64(self):
        # p90 of 1..100 is 91 -> ceil(91/64)*64 = 128 -> clamped to floor 256
        self.assertEqual(suggest_chunk_size(list(range(1, 101))), 256)

    def test_suggest_clamps_low(self):
        self.assertEqual(suggest_chunk_size([10, 20, 30]), 256)

    def test_suggest_clamps_high(self):
        self.assertEqual(suggest_chunk_size([5000] * 10), 1024)

    def test_suggest_empty_returns_default(self):
        self.assertEqual(suggest_chunk_size([]), 512)
        self.assertEqual(suggest_chunk_size([0, 0]), 512)


if __name__ == "__main__":
    unittest.main()
