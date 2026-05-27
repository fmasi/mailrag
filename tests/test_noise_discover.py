"""Tests for discover-related helpers in scripts/noise.py."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# Patch _WHITELIST_PATH and _RULES_PATH before importing so tests use temp files
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLoadWhitelist(unittest.TestCase):

    def _write_whitelist(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return Path(f.name)

    def tearDown(self):
        # Remove any temp files created by _save_whitelist_domain via patch
        pass

    def test_file_not_found_returns_empty(self):
        from scripts.noise import _load_whitelist
        with patch("scripts.noise._WHITELIST_PATH", Path("/tmp/__no_such_whitelist__.yaml")):
            result = _load_whitelist()
        self.assertEqual(result, frozenset())

    def test_empty_domains_list(self):
        p = self._write_whitelist("domains: []\n")
        try:
            from scripts.noise import _load_whitelist
            with patch("scripts.noise._WHITELIST_PATH", p):
                result = _load_whitelist()
            self.assertEqual(result, frozenset())
        finally:
            os.unlink(p)

    def test_domains_loaded_and_lowercased(self):
        p = self._write_whitelist("domains:\n  - Globex.com\n  - Partner.IO\n")
        try:
            from scripts.noise import _load_whitelist
            with patch("scripts.noise._WHITELIST_PATH", p):
                result = _load_whitelist()
            self.assertIn("globex.com", result)
            self.assertIn("partner.io", result)
        finally:
            os.unlink(p)

    def test_malformed_yaml_returns_empty(self):
        p = self._write_whitelist("domains: [unclosed\n")
        try:
            from scripts.noise import _load_whitelist
            with patch("scripts.noise._WHITELIST_PATH", p):
                result = _load_whitelist()
            self.assertEqual(result, frozenset())
        finally:
            os.unlink(p)

    def test_missing_domains_key_returns_empty(self):
        p = self._write_whitelist("# comment only\n")
        try:
            from scripts.noise import _load_whitelist
            with patch("scripts.noise._WHITELIST_PATH", p):
                result = _load_whitelist()
            self.assertEqual(result, frozenset())
        finally:
            os.unlink(p)


class TestSaveWhitelistDomain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        self.tmp.close()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        if self.path.exists():
            os.unlink(self.path)

    def test_creates_file_with_domain(self):
        from scripts.noise import _save_whitelist_domain, _load_whitelist
        with patch("scripts.noise._WHITELIST_PATH", self.path):
            _save_whitelist_domain("globex.com")
            result = _load_whitelist()
        self.assertIn("globex.com", result)

    def test_idempotent_duplicate_not_added(self):
        import yaml
        from scripts.noise import _save_whitelist_domain
        with patch("scripts.noise._WHITELIST_PATH", self.path):
            _save_whitelist_domain("globex.com")
            _save_whitelist_domain("globex.com")
            with open(self.path) as f:
                data = yaml.safe_load(f) or {}
        self.assertEqual(data.get("domains", []).count("globex.com"), 1)

    def test_appends_to_existing(self):
        from scripts.noise import _save_whitelist_domain, _load_whitelist
        with patch("scripts.noise._WHITELIST_PATH", self.path):
            _save_whitelist_domain("first.com")
            _save_whitelist_domain("second.com")
            result = _load_whitelist()
        self.assertIn("first.com", result)
        self.assertIn("second.com", result)

    def test_domain_stored_lowercase(self):
        from scripts.noise import _save_whitelist_domain, _load_whitelist
        with patch("scripts.noise._WHITELIST_PATH", self.path):
            _save_whitelist_domain("Globex.COM")
            result = _load_whitelist()
        self.assertIn("globex.com", result)


class TestResultToRule(unittest.TestCase):

    def test_dedicated_noise_produces_sender_domains_rule(self):
        from scripts.noise import _result_to_rule
        rule = _result_to_rule("acme.com", {"is_noise": True, "description": "Acme spam"}, is_general=False)
        self.assertIsNotNone(rule)
        self.assertEqual(rule["domain"], "acme.com")
        self.assertIn("acme.com", rule["sender_domains"])
        self.assertEqual(rule["description"], "Acme spam")

    def test_dedicated_clean_still_produces_rule_when_user_overrides(self):
        # If user chose [y] despite LLM saying clean, we still build a rule
        from scripts.noise import _result_to_rule
        rule = _result_to_rule("acme.com", {"is_noise": False, "description": ""}, is_general=False)
        self.assertIsNotNone(rule)
        self.assertIn("acme.com", rule["sender_domains"])

    def test_dedicated_none_result_uses_fallback_description(self):
        from scripts.noise import _result_to_rule
        rule = _result_to_rule("acme.com", None, is_general=False)
        self.assertIsNotNone(rule)
        self.assertIn("acme.com", rule["sender_domains"])
        self.assertIn("acme.com", rule["description"])

    def test_general_with_patterns_produces_pattern_rule(self):
        from scripts.noise import _result_to_rule
        result = {
            "is_noise": True,
            "description": "Gmail newsletters",
            "sender_patterns": ["noreply@gmail"],
            "subject_patterns": ["unsubscribe"],
        }
        rule = _result_to_rule("gmail.com", result, is_general=True)
        self.assertIsNotNone(rule)
        self.assertEqual(rule["sender_patterns"], ["noreply@gmail"])
        self.assertEqual(rule["subject_patterns"], ["unsubscribe"])
        self.assertNotIn("sender_domains", rule)

    def test_general_without_patterns_returns_none(self):
        from scripts.noise import _result_to_rule
        result = {"is_noise": False, "sender_patterns": [], "subject_patterns": [], "description": ""}
        rule = _result_to_rule("gmail.com", result, is_general=True)
        self.assertIsNone(rule)

    def test_general_none_result_returns_none(self):
        from scripts.noise import _result_to_rule
        rule = _result_to_rule("gmail.com", None, is_general=True)
        self.assertIsNone(rule)


class TestInteractiveDomainPrompt(unittest.TestCase):
    """Tests for _interactive_domain_prompt — mocks input() to simulate user choices."""

    def _make_entry(self, subjects=None):
        return {
            "unique_emails": 42,
            "sample_senders": ["noreply@acme.com"],
            "sample_subjects": subjects or ["Newsletter Q1", "Promo: 50% off"],
        }

    def _make_result(self, is_noise=True, desc="Acme spam"):
        return {"is_noise": is_noise, "description": desc}

    def _prompt(self, choices, domain="acme.com", is_general=False, **kwargs):
        from scripts.noise import _interactive_domain_prompt
        entry = self._make_entry()
        result = self._make_result()
        qdrant = MagicMock()
        llm = MagicMock()
        with patch("builtins.input", side_effect=choices):
            return _interactive_domain_prompt(
                domain, entry, result, is_general,
                qdrant, "test-collection", llm, "gpt-4o-mini", 10,
            )

    def test_y_returns_rule(self):
        decision = self._prompt(["y"])
        self.assertEqual(decision, "rule")

    def test_n_returns_skip(self):
        decision = self._prompt(["n"])
        self.assertEqual(decision, "skip")

    def test_w_calls_save_and_returns_whitelist(self):
        from scripts.noise import _interactive_domain_prompt
        with patch("scripts.noise._save_whitelist_domain") as mock_save:
            with patch("builtins.input", side_effect=["w"]):
                decision = _interactive_domain_prompt(
                    "acme.com", self._make_entry(), self._make_result(), False,
                    MagicMock(), "col", MagicMock(), "gpt-4o-mini", 10,
                )
        self.assertEqual(decision, "whitelist")
        mock_save.assert_called_once_with("acme.com")

    def test_invalid_then_y_returns_rule(self):
        decision = self._prompt(["z", "?", "y"])
        self.assertEqual(decision, "rule")

    def test_keyboard_interrupt_returns_skip(self):
        from scripts.noise import _interactive_domain_prompt
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            decision = _interactive_domain_prompt(
                "acme.com", self._make_entry(), self._make_result(), False,
                MagicMock(), "col", MagicMock(), "gpt-4o-mini", 10,
            )
        self.assertEqual(decision, "skip")

    def test_eof_returns_skip(self):
        from scripts.noise import _interactive_domain_prompt
        with patch("builtins.input", side_effect=EOFError):
            decision = _interactive_domain_prompt(
                "acme.com", self._make_entry(), self._make_result(), False,
                MagicMock(), "col", MagicMock(), "gpt-4o-mini", 10,
            )
        self.assertEqual(decision, "skip")

    def test_deep_inspect_then_y_returns_rule(self):
        """[2] deep inspect then [y] should return 'rule'."""
        from scripts.noise import _interactive_domain_prompt

        mock_qdrant = MagicMock()
        mock_llm = MagicMock()

        sample = [
            {"sender": "noreply@acme.com", "subject": "Newsletter", "body": "spam body"},
            {"sender": "noreply@acme.com", "subject": "Promo",      "body": "buy now"},
        ]

        with patch("scripts.noise._fetch_domain_email_sample", return_value=sample), \
             patch("scripts.noise._llm_classify_email_batch",  return_value=[True, True]), \
             patch("builtins.input", side_effect=["2", "y"]):
            decision = _interactive_domain_prompt(
                "acme.com", self._make_entry(), self._make_result(), False,
                mock_qdrant, "col", mock_llm, "gpt-4o-mini", 10,
            )
        self.assertEqual(decision, "rule")

    def test_read_email_then_n_returns_skip(self):
        """[3] read email (decline to see more), then [n] = skip."""
        from scripts.noise import _interactive_domain_prompt

        sample = [{"sender": "a@b.com", "subject": "Hi", "body": "Hello world"}]

        with patch("scripts.noise._fetch_domain_email_sample", return_value=sample), \
             patch("builtins.input", side_effect=["3", "n"]):
            decision = _interactive_domain_prompt(
                "acme.com", self._make_entry(), self._make_result(), False,
                MagicMock(), "col", MagicMock(), "gpt-4o-mini", 10,
            )
        self.assertEqual(decision, "skip")


class TestExtractBody(unittest.TestCase):
    """Tests for _extract_body — LlamaIndex stores body in _node_content JSON."""

    def test_extracts_text_from_node_content(self):
        from scripts.noise import _extract_body
        payload = {"_node_content": json.dumps({"text": "Hello world", "metadata": {}})}
        self.assertEqual(_extract_body(payload), "Hello world")

    def test_extracts_content_key_from_node_content(self):
        from scripts.noise import _extract_body
        payload = {"_node_content": json.dumps({"content": "Alt body", "metadata": {}})}
        self.assertEqual(_extract_body(payload), "Alt body")

    def test_falls_back_to_body_key(self):
        from scripts.noise import _extract_body
        payload = {"body": "Direct body"}
        self.assertEqual(_extract_body(payload), "Direct body")

    def test_falls_back_to_text_key(self):
        from scripts.noise import _extract_body
        payload = {"text": "Direct text"}
        self.assertEqual(_extract_body(payload), "Direct text")

    def test_empty_payload_returns_empty_string(self):
        from scripts.noise import _extract_body
        self.assertEqual(_extract_body({}), "")

    def test_malformed_node_content_falls_back_to_direct_keys(self):
        from scripts.noise import _extract_body
        payload = {"_node_content": "not valid json", "body": "fallback"}
        self.assertEqual(_extract_body(payload), "fallback")

    def test_node_content_with_empty_text_falls_back_to_direct_keys(self):
        from scripts.noise import _extract_body
        payload = {"_node_content": json.dumps({"text": ""}), "body": "fallback"}
        self.assertEqual(_extract_body(payload), "fallback")


class TestLlmBodyCharsUsed(unittest.TestCase):
    """Verify _llm_classify_email_batch truncates body to _LLM_BODY_CHARS, not a hard-coded 400."""

    def test_body_truncated_to_llm_body_chars(self):
        from scripts.noise import _llm_classify_email_batch, _LLM_BODY_CHARS

        long_body = "x" * 2000  # longer than any reasonable limit
        batch = [{"sender": "a@b.com", "subject": "Test", "body": long_body}]

        captured_prompt = []

        def fake_call_llm(client, model, prompt):
            captured_prompt.append(prompt)
            return [False]

        with patch("scripts.noise._call_llm", side_effect=fake_call_llm):
            _llm_classify_email_batch(None, "gpt-4o-mini", "b.com", batch)

        self.assertEqual(len(captured_prompt), 1)
        prompt = captured_prompt[0]
        # The body in the prompt must be at most _LLM_BODY_CHARS 'x' chars
        body_section = "x" * _LLM_BODY_CHARS
        self.assertIn(body_section, prompt)
        # And the character after the limit must NOT appear
        self.assertNotIn("x" * (_LLM_BODY_CHARS + 1), prompt)


if __name__ == "__main__":
    unittest.main()
