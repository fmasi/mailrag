"""Shared setup for the private eval scripts (`scripts/eval/_paths.py`).

These scripts are run rarely — months apart, to re-verify published numbers — so
their failure modes matter more than their happy path. A stale absolute path or a
missing key must say what to do, not raise from three frames deeper.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

from scripts.eval._paths import REPO_ROOT, bootstrap, data_path, jreq, require_key


class TestRepoRoot(unittest.TestCase):
    def test_points_at_the_actual_checkout(self):
        """Derived from __file__, so the scripts run against the tree they live
        in — the old hardcoded worktree path measured a stale checkout."""
        self.assertTrue((REPO_ROOT / "pyproject.toml").exists())
        self.assertTrue((REPO_ROOT / "src").is_dir())

    def test_is_derived_from_this_file_not_hardcoded(self):
        """It may legitimately sit under a home directory — the invariant is that
        it is computed from the module's own location, so the scripts follow the
        checkout wherever it is cloned."""
        import scripts.eval._paths as mod

        self.assertEqual(REPO_ROOT, pathlib.Path(mod.__file__).resolve().parents[2])


class TestBootstrap(unittest.TestCase):
    def test_puts_the_repo_on_sys_path(self):
        bootstrap()
        self.assertIn(str(REPO_ROOT), sys.path)

    def test_does_not_change_the_working_directory(self):
        """The old scripts chdir'd into a worktree, which is how they ended up
        measuring stale code. Regression guard."""
        before = os.getcwd()
        bootstrap()
        self.assertEqual(os.getcwd(), before)

    def test_sets_qdrant_and_offline_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            bootstrap()
            self.assertEqual(os.environ["QDRANT_URL"], "http://localhost:6333")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_does_not_override_an_explicit_qdrant_url(self):
        with mock.patch.dict(os.environ, {"QDRANT_URL": "http://elsewhere:9999"}):
            bootstrap()
            self.assertEqual(os.environ["QDRANT_URL"], "http://elsewhere:9999")

    def test_offline_false_leaves_the_hub_alone(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            bootstrap(offline=False)
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_is_idempotent(self):
        bootstrap()
        n = sys.path.count(str(REPO_ROOT))
        bootstrap()
        self.assertEqual(sys.path.count(str(REPO_ROOT)), n)


class TestDataPath(unittest.TestCase):
    def test_returns_an_existing_default(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(data_path("X_UNSET", d, what="thing"), pathlib.Path(d))

    def test_the_env_var_wins_over_the_default(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"MAILRAG_EVAL_X": d}):
                self.assertEqual(
                    data_path("MAILRAG_EVAL_X", "/nonexistent", what="thing"), pathlib.Path(d)
                )

    def test_expands_a_tilde(self):
        with mock.patch.dict(os.environ, {"MAILRAG_EVAL_X": "~"}):
            self.assertEqual(
                data_path("MAILRAG_EVAL_X", "/nope", what="thing"), pathlib.Path.home()
            )

    def test_missing_data_names_the_variable_to_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                data_path("MAILRAG_EVAL_TREC", "/definitely/not/here", what="the TREC corpus")
        msg = str(cm.exception)
        self.assertIn("MAILRAG_EVAL_TREC", msg)
        self.assertIn("the TREC corpus", msg)
        self.assertIn("/definitely/not/here", msg)

    def test_missing_data_points_at_the_public_alternative(self):
        """Someone hitting this may not realise these are private scripts."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                data_path("MAILRAG_EVAL_TREC", "/nope", what="x")
        self.assertIn("make bench", str(cm.exception))


class TestRequireKey(unittest.TestCase):
    def test_returns_the_key_when_set(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "sk-test"}):
            self.assertEqual(require_key(), "sk-test")

    def test_whitespace_only_counts_as_missing(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "   "}):
            with self.assertRaises(SystemExit):
                require_key()

    def test_the_message_warns_that_the_endpoint_is_paid(self):
        """A benchmark that silently starts billing is a bad surprise."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                require_key(what="the rerank arm")
        msg = str(cm.exception)
        self.assertIn("NVIDIA_API_KEY", msg)
        self.assertIn("PAID", msg)
        self.assertIn("the rerank arm", msg)

    def test_it_does_not_raise_a_bare_assertion(self):
        """The scripts used to `assert KEY`, which says nothing about which key
        or why."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                require_key()

    def test_a_keychain_reference_is_dereferenced(self):
        """The token should never need to sit in a dotfile or a shell history —
        same rule the sync passwords and RAG_LLM_API_KEY already follow."""
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "keychain:some.service"}):
            with mock.patch("src.config.secrets.resolve_secret", return_value="nvapi-xyz") as rs:
                self.assertEqual(require_key(), "nvapi-xyz")
            rs.assert_called_once_with("keychain:some.service")

    def test_an_unresolvable_reference_says_so(self):
        from src.config.secrets import SecretError

        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "keychain:missing.service"}):
            with mock.patch("src.config.secrets.resolve_secret", side_effect=SecretError("nope")):
                with self.assertRaises(SystemExit) as cm:
                    require_key()
        self.assertIn("keychain", str(cm.exception))

    def test_a_literal_key_still_works(self):
        """Exporting a key for one manual run is normal; this variable is not
        read from a config file, so a literal is not the hazard it is there."""
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-literal"}):
            self.assertEqual(require_key(), "nvapi-literal")

    def test_the_message_offers_the_keychain_form(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                require_key()
        self.assertIn("keychain:mailrag.nvidia.token", str(cm.exception))


class TestJreqRetries(unittest.TestCase):
    """Two live runs of the R6 benchmark died on transient connection errors —
    a TLS handshake timeout and `[Errno 54] Connection reset by peer` — because
    the old per-script helpers retried on HTTP 429 and nothing else. Hundreds of
    calls and the GPU time behind them, discarded by one blip.
    """

    def _urlopen(self, *effects):
        """A urlopen double that yields *effects* in order; a dict means success."""
        calls = {"n": 0}

        def fake(req, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            eff = effects[min(i, len(effects) - 1)]
            if isinstance(eff, Exception):
                raise eff
            cm = mock.MagicMock()
            cm.__enter__.return_value = io.BytesIO(json.dumps(eff).encode())
            return cm

        return fake, calls

    def test_returns_the_decoded_body_on_success(self):
        fake, _ = self._urlopen({"ok": True})
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            self.assertEqual(jreq("https://x/y", {"a": 1}), {"ok": True})

    def test_retries_a_connection_reset_then_succeeds(self):
        fake, calls = self._urlopen(ConnectionResetError(54, "Connection reset by peer"), {"ok": 1})
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            self.assertEqual(jreq("https://x/y", {}), {"ok": 1})
        self.assertEqual(calls["n"], 2)

    def test_retries_a_handshake_timeout_then_succeeds(self):
        fake, calls = self._urlopen(
            urllib.error.URLError("_ssl.c:1015: The handshake operation timed out"), {"ok": 1}
        )
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            self.assertEqual(jreq("https://x/y", {}), {"ok": 1})
        self.assertEqual(calls["n"], 2)

    def test_still_retries_rate_limiting(self):
        err = urllib.error.HTTPError("https://x/y", 429, "Too Many Requests", {}, None)
        fake, calls = self._urlopen(err, {"ok": 1})
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            self.assertEqual(jreq("https://x/y", {}), {"ok": 1})
        self.assertEqual(calls["n"], 2)

    def test_retries_a_server_error(self):
        err = urllib.error.HTTPError("https://x/y", 503, "Service Unavailable", {}, None)
        fake, calls = self._urlopen(err, {"ok": 1})
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            self.assertEqual(jreq("https://x/y", {}), {"ok": 1})
        self.assertEqual(calls["n"], 2)

    def test_does_not_retry_a_client_error(self):
        """401/400 will fail identically however many times it is sent — retrying
        just turns a clear auth failure into a slow one."""
        err = urllib.error.HTTPError("https://x/y", 401, "Unauthorized", {}, None)
        fake, calls = self._urlopen(err)
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                jreq("https://x/y", {})
        self.assertEqual(calls["n"], 1)

    def test_gives_up_eventually_and_names_the_last_error(self):
        fake, calls = self._urlopen(ConnectionResetError(54, "reset"))
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            with self.assertRaises(RuntimeError) as cm:
                jreq("https://x/y", {}, attempts=3)
        self.assertEqual(calls["n"], 3)
        self.assertIn("reset", str(cm.exception))

    def test_backoff_is_exponential_and_capped(self):
        fake, _ = self._urlopen(ConnectionResetError(54, "reset"))
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sl:
            with self.assertRaises(RuntimeError):
                jreq("https://x/y", {}, attempts=8)
        waits = [c.args[0] for c in sl.call_args_list]
        self.assertEqual(waits[:5], [1, 2, 4, 8, 16])
        self.assertTrue(all(w <= 30 for w in waits))


class TestNoScriptCarriesAHardcodedHome(unittest.TestCase):
    """The whole point of this module: a machine-specific absolute path in an
    eval script means the number it produces cannot be reproduced elsewhere."""

    def test_no_eval_script_hardcodes_an_absolute_home_path(self):
        offenders = []
        for path in sorted((REPO_ROOT / "scripts" / "eval").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if "/Users/" in line and not line.lstrip().startswith(("#", '"', "'")):
                    # _paths.py quotes the old pattern in its module docstring.
                    if path.name == "_paths.py":
                        continue
                    offenders.append(f"{path.name}:{i}: {line.strip()[:70]}")
        self.assertEqual(offenders, [], "hardcoded home paths found:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
