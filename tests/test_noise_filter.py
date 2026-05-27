"""Tests for src/data/noise_filter.py — NoiseFilter rule-based email classifier."""

import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import os

from src.data.noise_filter import NoiseFilter, _DEFAULT_RULES_PATH
from src.data.models import NormalizedEmail

# The real config/noise_rules.yaml is gitignored ("sensitive noise rules"), so it
# is absent in CI / fresh checkouts. The committed template carries the public seed.
_TEMPLATE_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "noise_rules.template.yaml"
)


def _make_email(sender: str = "", subject: str = "") -> NormalizedEmail:
    return NormalizedEmail(
        sender=sender,
        subject=subject,
        date=None,
        body="body text",
        source="test",
        source_id="test_0",
    )


def _filter_from_yaml(content: str) -> NoiseFilter:
    """Write YAML to a temp file and return a NoiseFilter loaded from it."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
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


if __name__ == "__main__":
    unittest.main()
