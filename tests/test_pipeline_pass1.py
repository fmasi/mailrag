import unittest

from src.data.models import NormalizedEmail
from src.data.noise_filter import NoiseFilter, _CategoryRule
from src.pipeline import pass1


def _email(sender, is_bulk=False):
    return NormalizedEmail(
        sender=sender, subject="s", date=None, body="b", source="t", source_id="t0", is_bulk=is_bulk
    )


class TestPass1(unittest.TestCase):
    def setUp(self):
        self.nf = NoiseFilter(
            [_CategoryRule(name="junk", description="", sender_domains=["junk.example"])]
        )

    def test_tags_matches_and_drops_nothing(self):
        emails = [_email("a@junk.example"), _email("b@real.example"), _email("c@junk.example")]
        kept, stats = pass1.run(emails, self.nf)
        self.assertEqual(len(kept), 3)
        self.assertEqual(stats.dropped, 0)
        self.assertEqual(stats.tagged, 2)
        self.assertTrue(kept[0].noise_candidate)
        self.assertFalse(kept[1].noise_candidate)
        self.assertTrue(kept[2].noise_candidate)

    def test_empty_filter_tags_nothing(self):
        kept, stats = pass1.run([_email("a@junk.example")], NoiseFilter([]))
        self.assertEqual(stats.tagged, 0)
        self.assertFalse(kept[0].noise_candidate)


if __name__ == "__main__":
    unittest.main()
