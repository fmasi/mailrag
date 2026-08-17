"""Tests for src/data/noise_filter.py — NoiseFilter rule-based email classifier."""

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.models import NormalizedEmail
from src.data.noise_filter import _DEFAULT_RULES_PATH, NoiseFilter

# The real config/noise_rules.yaml is gitignored ("sensitive noise rules"), so it
# is absent in CI / fresh checkouts. The committed template carries the public seed.
_TEMPLATE_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "noise_rules.template.yaml"
)


def _make_email(sender: str = "", subject: str = "", is_bulk: bool = False) -> NormalizedEmail:
    return NormalizedEmail(
        sender=sender,
        subject=subject,
        date=None,
        body="body text",
        source="test",
        source_id="test_0",
        is_bulk=is_bulk,
    )


def _filter_from_yaml(content: str) -> NoiseFilter:
    """Write YAML to a temp file and return a NoiseFilter loaded from it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(textwrap.dedent(content))
        path = f.name
    try:
        return NoiseFilter.from_file(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestNoiseFilterLoading(unittest.TestCase):
    def test_loads_sender_domains(self):
        nf = _filter_from_yaml("""
            categories:
              linkedin:
                description: LinkedIn
                sender_domains:
                  - linkedin.com
        """)
        self.assertEqual(nf.category_names(), ["linkedin"])
        self.assertFalse(nf.is_empty())

    def test_loads_multiple_categories(self):
        nf = _filter_from_yaml("""
            categories:
              cat_a:
                description: A
                sender_domains: [a.com]
              cat_b:
                description: B
                sender_domains: [b.com]
        """)
        self.assertEqual(nf.category_names(), ["cat_a", "cat_b"])

    def test_empty_categories_block_gives_empty_filter(self):
        nf = _filter_from_yaml("categories: {}\n")
        self.assertTrue(nf.is_empty())

    def test_missing_categories_key_gives_empty_filter(self):
        nf = _filter_from_yaml("# no categories here\n")
        self.assertTrue(nf.is_empty())

    def test_from_project_rules_returns_empty_when_file_missing(self):
        non_existent = Path("/tmp/does_not_exist_noise_rules.yaml")
        with patch("src.data.noise_filter._DEFAULT_RULES_PATH", non_existent):
            nf = NoiseFilter.from_project_rules()
        self.assertTrue(nf.is_empty())

    @unittest.skipUnless(
        _DEFAULT_RULES_PATH.exists(),
        "real config/noise_rules.yaml is gitignored / absent (e.g. in CI)",
    )
    def test_from_project_rules_loads_real_file(self):
        """The actual project noise_rules.yaml should load without error."""
        nf = NoiseFilter.from_project_rules()
        # linkedin_notifications is the seed rule we committed
        self.assertIn("linkedin_notifications", nf.category_names())

    def test_template_rules_load(self):
        """The committed noise_rules.template.yaml carries the public seed rule.

        This runs everywhere (the template is tracked), giving CI real coverage of
        rule loading even though the real noise_rules.yaml is gitignored.
        """
        self.assertTrue(_TEMPLATE_RULES_PATH.exists(), _TEMPLATE_RULES_PATH)
        nf = NoiseFilter.from_file(_TEMPLATE_RULES_PATH)
        self.assertIn("linkedin_notifications", nf.category_names())


# ---------------------------------------------------------------------------
# sender_domains matching
# ---------------------------------------------------------------------------


class TestSenderDomainMatching(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml("""
            categories:
              linkedin:
                description: LinkedIn
                sender_domains:
                  - linkedin.com
                  - e.linkedin.com
        """)

    def test_exact_domain_match(self):
        email = _make_email(sender="LinkedIn <jobs@linkedin.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_subdomain_match(self):
        email = _make_email(sender="LinkedIn <updates@e.linkedin.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_domain_match_is_case_insensitive(self):
        email = _make_email(sender="LinkedIn <jobs@LINKEDIN.COM>")
        self.assertTrue(self.nf.is_noise(email))

    def test_display_name_containing_domain_does_not_match(self):
        # "linkedin.com" only in display name, actual address is different
        email = _make_email(sender="linkedin.com info <noreply@example.com>")
        self.assertFalse(self.nf.is_noise(email))

    def test_partial_domain_does_not_match(self):
        # "notlinkedin.com" should NOT match "linkedin.com"
        email = _make_email(sender="test@notlinkedin.com")
        self.assertFalse(self.nf.is_noise(email))

    def test_clean_work_email_not_matched(self):
        email = _make_email(sender="Kevin Kim <kevin@example.com>")
        self.assertFalse(self.nf.is_noise(email))

    def test_empty_sender_not_matched(self):
        email = _make_email(sender="")
        self.assertFalse(self.nf.is_noise(email))


# ---------------------------------------------------------------------------
# sender_patterns matching
# ---------------------------------------------------------------------------


class TestSenderPatternMatching(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml("""
            categories:
              noreply:
                description: Generic no-reply addresses
                sender_patterns:
                  - "no.?reply@"
        """)

    def test_noreply_matches(self):
        email = _make_email(sender="System <noreply@example.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_no_dash_reply_matches(self):
        email = _make_email(sender="System <no-reply@example.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_pattern_is_case_insensitive(self):
        email = _make_email(sender="System <NoReply@example.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_real_sender_not_matched(self):
        email = _make_email(sender="Alice <alice@example.com>")
        self.assertFalse(self.nf.is_noise(email))


# ---------------------------------------------------------------------------
# subject_patterns matching
# ---------------------------------------------------------------------------


class TestSubjectPatternMatching(unittest.TestCase):
    def setUp(self):
        # Use YAML single-quoted strings for regexes: no escape processing,
        # so \b and \[ reach the regex engine as literal backslash sequences.
        self.nf = _filter_from_yaml(
            "categories:\n"
            "  unsubscribe_spam:\n"
            "    description: Marketing / unsubscribe footers\n"
            "    subject_patterns:\n"
            "      - '\\bunsubscribe\\b'\n"
            "      - '^\\[EXTERNAL\\]'\n"
        )

    def test_unsubscribe_in_subject(self):
        email = _make_email(subject="Click here to unsubscribe from our list")
        self.assertTrue(self.nf.is_noise(email))

    def test_external_prefix_matched(self):
        email = _make_email(subject="[EXTERNAL] Free offer inside")
        self.assertTrue(self.nf.is_noise(email))

    def test_subject_pattern_case_insensitive(self):
        email = _make_email(subject="UNSUBSCRIBE")
        self.assertTrue(self.nf.is_noise(email))

    def test_clean_subject_not_matched(self):
        email = _make_email(subject="Weekly sync agenda")
        self.assertFalse(self.nf.is_noise(email))

    def test_empty_subject_not_matched(self):
        email = _make_email(subject="")
        self.assertFalse(self.nf.is_noise(email))


# ---------------------------------------------------------------------------
# matched_category
# ---------------------------------------------------------------------------


class TestMatchedCategory(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml("""
            categories:
              linkedin:
                description: LinkedIn
                sender_domains: [linkedin.com]
              marketing:
                description: Marketing
                subject_patterns: ["special offer"]
        """)

    def test_returns_correct_category_for_sender_match(self):
        email = _make_email(sender="LinkedIn <jobs@linkedin.com>")
        self.assertEqual(self.nf.matched_category(email), "linkedin")

    def test_returns_correct_category_for_subject_match(self):
        email = _make_email(subject="Special Offer just for you")
        self.assertEqual(self.nf.matched_category(email), "marketing")

    def test_returns_none_for_clean_email(self):
        email = _make_email(sender="alice@example.com", subject="Project update")
        self.assertIsNone(self.nf.matched_category(email))


# ---------------------------------------------------------------------------
# match_payload (used for post-index Qdrant scanning)
# ---------------------------------------------------------------------------


class TestMatchPayload(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml("""
            categories:
              linkedin:
                description: LinkedIn
                sender_domains: [linkedin.com]
        """)

    def test_payload_dict_matched(self):
        matched, category = self.nf.match_payload(
            {"sender": "LinkedIn <jobs@linkedin.com>", "subject": "New jobs"}
        )
        self.assertTrue(matched)
        self.assertEqual(category, "linkedin")

    def test_payload_dict_not_matched(self):
        matched, category = self.nf.match_payload(
            {"sender": "kevin@example.com", "subject": "Sync tomorrow"}
        )
        self.assertFalse(matched)
        self.assertIsNone(category)

    def test_missing_keys_in_payload_do_not_raise(self):
        matched, category = self.nf.match_payload({})
        self.assertFalse(matched)
        self.assertIsNone(category)

    def test_partial_payload_sender_only(self):
        matched, _ = self.nf.match_payload({"sender": "x@linkedin.com"})
        self.assertTrue(matched)

    def test_partial_payload_subject_only(self):
        # No sender → can't match sender_domains
        matched, _ = self.nf.match_payload({"subject": "Hello"})
        self.assertFalse(matched)


# ---------------------------------------------------------------------------
# Multiple rules on one category + multiple categories
# ---------------------------------------------------------------------------


class TestMultipleRulesAndCategories(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml("""
            categories:
              social:
                description: Social media
                sender_domains: [linkedin.com, twitter.com]
                subject_patterns: ["follow us"]
              internal_digest:
                description: Internal digests
                sender_patterns: ["digest@"]
        """)

    def test_first_matching_domain_triggers_category(self):
        email = _make_email(sender="Twitter <notify@twitter.com>")
        self.assertTrue(self.nf.is_noise(email))
        self.assertEqual(self.nf.matched_category(email), "social")

    def test_subject_pattern_triggers_correct_category(self):
        email = _make_email(subject="Follow us on social media")
        self.assertTrue(self.nf.is_noise(email))
        self.assertEqual(self.nf.matched_category(email), "social")

    def test_second_category_matched(self):
        email = _make_email(sender="noreply <digest@company.com>")
        self.assertTrue(self.nf.is_noise(email))
        self.assertEqual(self.nf.matched_category(email), "internal_digest")

    def test_email_matching_neither_category_is_clean(self):
        email = _make_email(sender="alice@company.com", subject="Budget review")
        self.assertFalse(self.nf.is_noise(email))

    def test_is_empty_false_with_rules(self):
        self.assertFalse(self.nf.is_empty())


# ---------------------------------------------------------------------------
# Empty / edge-case filters
# ---------------------------------------------------------------------------


class TestEmptyFilter(unittest.TestCase):
    def test_empty_filter_never_matches_email(self):
        nf = NoiseFilter([])
        email = _make_email(sender="spam@linkedin.com", subject="You won!")
        self.assertFalse(nf.is_noise(email))

    def test_empty_filter_never_matches_payload(self):
        nf = NoiseFilter([])
        matched, cat = nf.match_payload({"sender": "spam@linkedin.com"})
        self.assertFalse(matched)
        self.assertIsNone(cat)

    def test_empty_filter_is_empty(self):
        nf = NoiseFilter([])
        self.assertTrue(nf.is_empty())

    def test_empty_filter_category_names_is_empty_list(self):
        nf = NoiseFilter([])
        self.assertEqual(nf.category_names(), [])


# ---------------------------------------------------------------------------
# bulk_filter — header-driven (List-Unsubscribe / Precedence:bulk) filtering
#
# Design: Pass-1 is conservative. A bulk-mail header marks a message as a
# *candidate* for dropping, but three guards keep it if it is likely wanted:
#   - sender is a freemail/personal domain  (human mailing-list traffic)
#   - sender domain is explicitly whitelisted (e.g. courses we want to keep)
#   - subject looks transactional            (receipts / shipping / bookings)
# ---------------------------------------------------------------------------


class TestBulkHeaderFiltering(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml(
            "categories:\n"
            "  known_spam:\n"
            "    description: explicit spam domain\n"
            "    sender_domains: [spammer.com]\n"
            "bulk_filter:\n"
            "  description: Drop bulk mail unless a guard keeps it\n"
            "  keep_freemail_domains: [gmail.com, googlemail.com, yahoo.com]\n"
            "  keep_domains: [edx.org]\n"
            "  keep_sender_patterns:\n"
            "    - 'inmail-hit-reply@linkedin\\.com'\n"
            "  keep_subject_patterns:\n"
            "    - '\\b(order|booking|reservation|payment|delivery|shipment|parcel|package)\\b.{0,40}\\b(confirm\\w*|cancel\\w*|ship\\w*|dispatch\\w*|deliver\\w*|receiv\\w*|refund\\w*)\\b'\n"
            "    - '\\b(confirm\\w*|cancel\\w*|ship\\w*|dispatch\\w*|deliver\\w*)\\b.{0,40}\\b(order|booking|reservation|payment|delivery|shipment|parcel|package)\\b'\n"
            "    - '\\b(receipt|invoice|e-?ticket|boarding pass|itinerary)\\b'\n"
            "    - '\\b(fatura|factura|facture|rechnung|recibo|re[\\u00e7c]u|nota fiscal)\\b'\n"
        )

    def test_bulk_marketing_email_is_noise(self):
        email = _make_email(sender="YouTube <noreply@youtube.com>", is_bulk=True)
        self.assertTrue(self.nf.is_noise(email))
        self.assertEqual(self.nf.matched_category(email), "bulk_unsubscribe")

    def test_same_sender_without_bulk_header_is_kept(self):
        email = _make_email(sender="YouTube <noreply@youtube.com>", is_bulk=False)
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_from_freemail_domain_is_kept(self):
        # Human mailing-list traffic (Google Groups injects List-Unsubscribe).
        email = _make_email(
            sender="Dana Reyes <d.reyes@gmail.com>",
            subject="Re: [Ubuntu-PE] Gutsy Release",
            is_bulk=True,
        )
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_from_whitelisted_domain_is_kept(self):
        email = _make_email(sender="edX <noreply@edx.org>", is_bulk=True)
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_from_exempt_sender_pattern_is_kept(self):
        # LinkedIn InMail carries List-Unsubscribe but is real human outreach.
        email = _make_email(sender="Recruiter <inmail-hit-reply@linkedin.com>", is_bulk=True)
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_with_transactional_subject_is_kept(self):
        email = _make_email(
            sender="Shop <noreply@shop.example>",
            subject="Your order 48587276 has been shipped out",
            is_bulk=True,
        )
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_with_receipt_subject_is_kept(self):
        email = _make_email(
            sender="Store <noreply@store.example>",
            subject="Your receipt from Acme",
            is_bulk=True,
        )
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_with_multilingual_invoice_subject_is_kept(self):
        # A utility bill in Portuguese: "Fatura" must be recognised too.
        email = _make_email(
            sender="Energia <clientes@energia.example>",
            subject="Energia: Fatura Eletrónica",
            is_bulk=True,
        )
        self.assertFalse(self.nf.is_noise(email))

    def test_bulk_email_with_marketing_booking_subject_is_still_noise(self):
        # "Finish booking" is an abandoned-cart nudge, not a real booking notice.
        email = _make_email(
            sender="Travelsite <updates@mp1.travelsite.example>",
            subject="Finish booking: Seaview Hotel",
            is_bulk=True,
        )
        self.assertTrue(self.nf.is_noise(email))

    def test_explicit_category_still_wins_for_bulk_email(self):
        email = _make_email(sender="x@spammer.com", is_bulk=True)
        self.assertTrue(self.nf.is_noise(email))
        self.assertEqual(self.nf.matched_category(email), "known_spam")

    def test_match_payload_honours_is_bulk_true(self):
        matched, cat = self.nf.match_payload(
            {"sender": "noreply@youtube.com", "subject": "new video", "is_bulk": True}
        )
        self.assertTrue(matched)
        self.assertEqual(cat, "bulk_unsubscribe")

    def test_match_payload_without_is_bulk_does_not_bulk_filter(self):
        matched, _ = self.nf.match_payload(
            {"sender": "noreply@youtube.com", "subject": "new video"}
        )
        self.assertFalse(matched)


class TestNoBulkFilterSection(unittest.TestCase):
    def test_bulk_email_kept_when_no_bulk_filter_configured(self):
        nf = _filter_from_yaml("""
            categories:
              linkedin:
                description: LinkedIn
                sender_domains: [linkedin.com]
        """)
        email = _make_email(sender="news@some-newsletter.example", is_bulk=True)
        self.assertFalse(nf.is_noise(email))


# ---------------------------------------------------------------------------
# LinkedIn rule narrowed: drop automated notifications, keep genuine InMail
# ---------------------------------------------------------------------------


class TestLinkedInNarrowing(unittest.TestCase):
    def setUp(self):
        self.nf = _filter_from_yaml(
            "categories:\n"
            "  linkedin_notifications:\n"
            "    description: LinkedIn automated notifications (not InMail)\n"
            "    sender_patterns:\n"
            "      - '-noreply@linkedin\\.com'\n"
            "      - '@e\\.linkedin\\.com'\n"
        )

    def test_job_alert_noreply_is_noise(self):
        email = _make_email(sender="LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_newsletter_noreply_is_noise(self):
        email = _make_email(sender="Notion via LinkedIn <newsletters-noreply@linkedin.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_marketing_subdomain_is_noise(self):
        email = _make_email(sender="LinkedIn <updates@e.linkedin.com>")
        self.assertTrue(self.nf.is_noise(email))

    def test_inmail_is_kept(self):
        # Real human outreach relayed via LinkedIn InMail — must NOT be filtered.
        email = _make_email(sender="Vanshika Bajaj <inmail-hit-reply@linkedin.com>")
        self.assertFalse(self.nf.is_noise(email))


if __name__ == "__main__":
    unittest.main()
