"""Model/quant/source provenance for LLM-produced judgments.

A model id alone does not identify a judge: the same model at a different
quantisation, or served remotely rather than locally, produces different output.
A cache that mixes them makes every noise-rate comparison over the corpus
unattributable — prompt, rubric, or a model file swapped months ago?
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.llm.cache import Pass2Cache
from src.llm.provenance import (
    Provenance,
    classify_source,
    describe_backend,
    model_fingerprint,
)


class TestClassifySource(unittest.TestCase):
    def test_loopback_endpoints_are_local(self):
        for url in (
            "http://localhost:1234/v1",
            "http://127.0.0.1:1234/v1",
            "http://host.docker.internal:1234/v1",
        ):
            self.assertEqual(classify_source(url), "local", url)

    def test_a_hosted_endpoint_is_remote(self):
        self.assertEqual(classify_source("https://api.openai.com/v1"), "remote")
        self.assertEqual(classify_source("https://integrate.api.nvidia.com/v1"), "remote")

    def test_an_unknown_host_defaults_to_remote_not_local(self):
        """The safe direction: calling a hosted endpoint 'local' would understate
        both cost and data exposure."""
        self.assertEqual(classify_source("http://some-box.lan:8000/v1"), "remote")

    def test_an_unparseable_endpoint_is_unknown(self):
        self.assertEqual(classify_source(""), "unknown")


class TestFingerprint(unittest.TestCase):
    def test_quant_is_part_of_the_identity(self):
        a = Provenance(model="gemma-4-26b", quant="Q4_K_M")
        b = Provenance(model="gemma-4-26b", quant="Q8_0")
        self.assertNotEqual(model_fingerprint(a), model_fingerprint(b))

    def test_a_bare_model_id_is_used_when_the_quant_is_unknown(self):
        """So rows written before quant was recorded still compare equal to
        themselves."""
        self.assertEqual(model_fingerprint(Provenance(model="gemma-4-26b")), "gemma-4-26b")

    def test_none_is_tolerated(self):
        self.assertEqual(model_fingerprint(None), "")

    def test_the_label_names_model_quant_and_where_it_ran(self):
        p = Provenance(
            model="gemma-4-26b", quant="Q4_K_M", endpoint="http://localhost:1234/v1", source="local"
        )
        label = p.label()
        for part in ("gemma-4-26b", "Q4_K_M", "local", "localhost"):
            self.assertIn(part, label)


class TestDescribeBackend(unittest.TestCase):
    def test_it_reads_the_quantisation_from_lm_studios_native_api(self):
        payload = {"data": [{"id": "gemma-4-26b", "quantization": "Q4_K_M", "arch": "gemma"}]}
        with mock.patch("src.llm.provenance._lmstudio_model_info", return_value=payload["data"][0]):
            p = describe_backend(model="gemma-4-26b", api_base="http://localhost:1234/v1")
        self.assertEqual(p.quant, "Q4_K_M")
        self.assertEqual(p.arch, "gemma")
        self.assertEqual(p.source, "local")

    def test_an_endpoint_without_that_api_degrades_to_empty_not_to_a_guess(self):
        with mock.patch("src.llm.provenance._lmstudio_model_info", return_value={}):
            p = describe_backend(model="gpt-4o", api_base="https://api.openai.com/v1")
        self.assertEqual(p.quant, "")
        self.assertEqual(p.model, "gpt-4o")
        self.assertEqual(p.source, "remote")

    def test_metadata_capture_never_raises(self):
        """It must never be the reason a sweep fails to start."""
        # unreachable endpoint
        p = describe_backend(model="m", api_base="http://127.0.0.1:1/v1")
        self.assertEqual(p.model, "m")
        self.assertEqual(p.quant, "")

    def test_a_non_string_endpoint_degrades_instead_of_crashing(self):
        """Callers pass a client attribute straight through; in tests that is a
        Mock, and in future it might be a URL object."""
        p = describe_backend(model="m", api_base=mock.Mock())
        self.assertEqual(p.model, "m")
        self.assertIn(p.source, ("unknown", "remote"))

    def test_it_falls_back_to_the_environment(self):
        env = {"RAG_LLM_MODEL": "envmodel", "RAG_LLM_API_BASE": "http://localhost:1234/v1"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("src.llm.provenance._lmstudio_model_info", return_value={}),
        ):
            self.assertEqual(describe_backend().model, "envmodel")


class TestCacheRecordsProvenance(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.cache = Pass2Cache(os.path.join(self.d, "p2.db"))
        self.addCleanup(self.cache.close)

    def _put(self, sha, **kw):
        self.cache.put(sha, {"summary": "s", "is_noise": 0, "confidence": 0.9}, **kw)

    def test_quant_endpoint_and_source_round_trip(self):
        self._put(
            "a",
            model="gemma@Q4_K_M",
            quant="Q4_K_M",
            endpoint="http://localhost:1234/v1",
            source="local",
        )
        row = self.cache.get("a")
        self.assertEqual(row["quant"], "Q4_K_M")
        self.assertEqual(row["source"], "local")
        self.assertEqual(row["endpoint"], "http://localhost:1234/v1")

    def test_judges_reports_a_corpus_judged_by_more_than_one_model(self):
        """The question the cache must be able to answer before a comparison is
        trusted."""
        self._put("a", model="gemma@Q4_K_M", quant="Q4_K_M", source="local")
        self._put("b", model="gemma@Q4_K_M", quant="Q4_K_M", source="local")
        self._put("c", model="magistral@Q8_0", quant="Q8_0", source="local")
        judges = self.cache.judges()
        self.assertEqual(len(judges), 2)
        self.assertEqual(judges["gemma@Q4_K_M [local]"], 2)
        self.assertEqual(judges["magistral@Q8_0 [local]"], 1)

    def test_rows_written_before_provenance_existed_are_still_reported(self):
        self._put("a", model="gemma-4-26b")
        self.assertIn("gemma-4-26b", "".join(self.cache.judges()))

    def test_an_older_cache_file_is_migrated_in_place(self):
        import sqlite3

        path = os.path.join(self.d, "legacy.db")
        con = sqlite3.connect(path)
        con.execute(
            "CREATE TABLE pass2 (sha256 TEXT PRIMARY KEY, summary TEXT NOT NULL DEFAULT '', "
            "is_noise INTEGER NOT NULL, confidence REAL NOT NULL, reason TEXT NOT NULL DEFAULT '', "
            "model TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)"
        )
        con.execute(
            "INSERT INTO pass2 (sha256, is_noise, confidence, model, created_at) "
            "VALUES ('old', 1, 0.9, 'gemma-4-26b', 't')"
        )
        con.commit()
        con.close()
        c = Pass2Cache(path)
        self.addCleanup(c.close)
        self.assertEqual(c.get("old")["model"], "gemma-4-26b")
        self.assertIsNone(c.get("old")["quant"])


if __name__ == "__main__":
    unittest.main()


class TestKeyReferenceResolution(unittest.TestCase):
    """RAG_LLM_API_KEY may be a reference (keychain:/env:/file:). Provenance must
    resolve it the same way the client does — sending the raw reference as a
    bearer token yields a 401 and degrades the record to "unknown quant" without
    saying so, which defeats the point of recording provenance at all."""

    def test_a_reference_is_resolved_before_the_metadata_call(self):
        seen = {}

        def spy(api_base, model, api_key=""):
            seen["api_key"] = api_key
            return {"quantization": "8bit"}

        env = {
            "RAG_LLM_API_KEY": "env:MY_REAL_TOKEN",
            "MY_REAL_TOKEN": "the-actual-secret",
            "RAG_LLM_API_BASE": "http://localhost:1234/v1",
        }
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("src.llm.provenance._lmstudio_model_info", side_effect=spy),
        ):
            p = describe_backend(model="m", api_base="http://localhost:1234/v1")
        self.assertEqual(seen["api_key"], "the-actual-secret")
        self.assertNotIn("env:", seen["api_key"])
        self.assertEqual(p.quant, "8bit")

    def test_an_unresolvable_reference_does_not_block_the_run(self):
        env = {"RAG_LLM_API_KEY": "env:DOES_NOT_EXIST"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("src.llm.provenance._lmstudio_model_info", return_value={}),
        ):
            self.assertEqual(describe_backend(model="m", api_base="http://x/v1").model, "m")


class TestProvenanceOnEveryJudgePath(unittest.TestCase):
    """Provenance must be recorded on the SYNC path, not only the interactive
    `summarize` verb. Scheduled sync is where it matters most: nobody is
    watching, so a model or quant change there would otherwise be undetectable
    after the fact. Found in review — 237 sync-judged rows had NULL quant."""

    def test_the_sync_runner_passes_provenance_to_run_pass(self):
        import inspect

        from src.sync import runner

        src = inspect.getsource(runner._default_run_pass)
        self.assertIn("describe_backend", src)
        self.assertIn("provenance=", src)

    def test_the_judge_verb_passes_provenance_to_run_pass(self):
        import inspect

        from src.pipeline import judge

        src = inspect.getsource(judge)
        self.assertIn("describe_backend", src)
        self.assertIn("provenance=", src)

    def test_run_pass_records_the_quant_it_is_given(self):
        """End-to-end through the real run_pass, so a caller that forgets the
        kwarg cannot pass this by accident."""
        import shutil
        import tempfile

        from src.llm.cache import Pass2Cache
        from src.llm.pass2 import run_pass
        from src.llm.provenance import Provenance

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "m.eml")
        with open(path, "w") as fh:
            fh.write("Subject: x\n\nbody")
        cache = Pass2Cache(os.path.join(d, "p2.db"))
        self.addCleanup(cache.close)
        run_pass(
            [path],
            cache,
            lambda p: {"sender": "a", "subject": "s", "date": "", "body": "b", "message_id": ""},
            lambda e: {"summary": "s", "is_noise": 0, "confidence": 0.9},
            "gemma",
            provenance=Provenance(
                model="gemma", quant="8bit", endpoint="http://localhost:1234/v1", source="local"
            ),
        )
        self.assertEqual(cache.judges(), {"gemma@8bit [local]": 1})
