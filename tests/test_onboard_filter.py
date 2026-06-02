import unittest
from src.onboard import filter_kept


class _Email:
    def __init__(self, mid):
        self.message_id = mid
        self.summary = None


def _rec(is_noise, conf, summary=""):
    return {"is_noise": is_noise, "confidence": conf, "summary": summary, "reason": ""}


class TestFilterKept(unittest.TestCase):
    def test_drops_confident_noise_only(self):
        a, b, c = _Email("a"), _Email("b"), _Email("c")
        judgments = {
            "a": _rec(True, 0.9),            # dropped
            "b": _rec(True, 0.5),            # kept (below threshold)
            "c": _rec(False, 0.99, "ham"),   # kept, summary set
        }
        kept, dropped = filter_kept([a, b, c], judgments, min_confidence=0.7)
        self.assertEqual(dropped, 1)
        self.assertEqual([e.message_id for e in kept], ["b", "c"])
        self.assertEqual(c.summary, "ham")

    def test_email_without_judgment_is_kept(self):
        a = _Email("a")
        kept, dropped = filter_kept([a], {}, min_confidence=0.7)
        self.assertEqual(dropped, 0)
        self.assertEqual(kept, [a])


if __name__ == "__main__":
    unittest.main()
