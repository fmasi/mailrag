import unittest

from src.llm import calibration


def _rec(is_noise, subject="", summary="", reason="", sender=""):
    return {
        "sender": sender,
        "subject": subject,
        "is_noise": is_noise,
        "confidence": 1.0,
        "summary": summary,
        "reason": reason,
    }


class TestCalibration(unittest.TestCase):
    def test_noise_rate(self):
        recs = [_rec(True), _rec(True), _rec(False), _rec(False)]
        self.assertAlmostEqual(calibration.noise_rate(recs), 0.5)

    def test_noise_rate_empty_is_zero(self):
        self.assertEqual(calibration.noise_rate([]), 0.0)

    def test_false_noise_flags_recordish_marked_noise(self):
        recs = [
            _rec(True, subject="Your invoice #123", reason="looks like a digest"),
            _rec(True, subject="Big sale today", reason="marketing"),
        ]
        out = calibration.false_noise(recs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["subject"], "Your invoice #123")

    def test_false_keep_flags_promoish_kept(self):
        recs = [
            _rec(False, subject="Weekly newsletter digest", summary="a content digest"),
            _rec(False, subject="Your receipt", summary="receipt from shop"),
        ]
        out = calibration.false_keep(recs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["subject"], "Weekly newsletter digest")

    def test_false_keep_excludes_recordish_even_if_promo_words(self):
        # A "sale" receipt with an invoice is a record, not a false-keep.
        recs = [_rec(False, subject="Sale receipt invoice", summary="invoice for sale items")]
        self.assertEqual(calibration.false_keep(recs), [])

    def test_format_report_contains_rate_and_buckets(self):
        report = calibration.CalibrationReport(
            rubric="personal",
            sample=2,
            noise_rate=0.5,
            false_noise=[_rec(True, subject="invoice", reason="digest")],
            false_keep=[_rec(False, subject="newsletter", summary="digest")],
        )
        text = calibration.format_report(report)
        self.assertIn("personal", text)
        self.assertIn("50%", text)
        self.assertIn("FALSE-NOISE", text)
        self.assertIn("FALSE-KEEP", text)

    def test_format_report_empty_buckets_ok(self):
        report = calibration.CalibrationReport(rubric="work", sample=0, noise_rate=0.0)
        text = calibration.format_report(report)
        self.assertIn("work", text)
        self.assertIn("FALSE-NOISE suspects", text)
        self.assertIn("] 0", text)  # zero-count buckets render without error

    def test_false_keep_kept_when_only_reason_is_recordish(self):
        # PROMO fires from the blob (which includes reason); the REC-exclusion check
        # looks only at subject+summary, so a record-ish *reason* must NOT suppress it.
        recs = [
            _rec(
                False,
                subject="newsletter digest",
                summary="weekly digest",
                reason="contains an invoice mention",
            )
        ]
        self.assertEqual(len(calibration.false_keep(recs)), 1)

    def test_blob_tolerates_none_fields(self):
        recs = [
            {
                "sender": None,
                "subject": None,
                "is_noise": True,
                "confidence": 1.0,
                "summary": None,
                "reason": "your invoice is ready",
            }
        ]
        # reason is record-ish -> false_noise should catch it without a TypeError.
        self.assertEqual(len(calibration.false_noise(recs)), 1)


if __name__ == "__main__":
    unittest.main()
