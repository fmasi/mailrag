# tests/test_llm_pass2.py
"""Tests for the Pass-2 orchestration core (stdlib-only; fake LLM)."""

import os
import tempfile
import unittest

from src.llm import pass2
from src.llm.cache import Pass2Cache


def _write(path, data=b"body"):
    with open(path, "wb") as fh:
        fh.write(data)


class TestRunPass(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = Pass2Cache(os.path.join(self.dir.name, "c.db"))
        self.f1 = os.path.join(self.dir.name, "a.eml")
        _write(self.f1, b"one")
        self.f2 = os.path.join(self.dir.name, "b.eml")
        _write(self.f2, b"two")

    def tearDown(self):
        self.cache.close()
        self.dir.cleanup()

    def _summarize(self, email):
        return {"is_noise": False, "confidence": 0.9, "summary": "s", "reason": "r"}

    def _load(self, path):
        return {"sender": "x", "subject": "y", "date": "z", "body": "b"}

    def test_processes_all_then_skips_cached(self):
        c1 = pass2.run_pass([self.f1, self.f2], self.cache, self._load, self._summarize, model="m")
        self.assertEqual(c1, {"cached": 0, "done": 2, "error": 0})
        c2 = pass2.run_pass([self.f1, self.f2], self.cache, self._load, self._summarize, model="m")
        self.assertEqual(c2, {"cached": 2, "done": 0, "error": 0})

    def test_limit_stops_early(self):
        c = pass2.run_pass(
            [self.f1, self.f2], self.cache, self._load, self._summarize, model="m", limit=1
        )
        self.assertEqual(c["done"], 1)
        self.assertEqual(self.cache.stats()["total"], 1)

    def test_error_leaves_uncached(self):
        def boom(email):
            raise RuntimeError("llm down")

        c = pass2.run_pass([self.f1], self.cache, self._load, boom, model="m")
        self.assertEqual(c["error"], 1)
        self.assertEqual(self.cache.stats()["total"], 0)

    def test_progress_true_returns_same_counts(self):
        c = pass2.run_pass(
            [self.f1, self.f2], self.cache, self._load, self._summarize, model="m", progress=True
        )
        self.assertEqual(c, {"cached": 0, "done": 2, "error": 0})
        self.assertEqual(self.cache.stats()["total"], 2)

    def test_workers_process_all_then_skip_cached(self):
        c1 = pass2.run_pass(
            [self.f1, self.f2], self.cache, self._load, self._summarize, model="m", workers=4
        )
        self.assertEqual(c1, {"cached": 0, "done": 2, "error": 0})
        self.assertEqual(self.cache.stats()["total"], 2)
        c2 = pass2.run_pass(
            [self.f1, self.f2], self.cache, self._load, self._summarize, model="m", workers=4
        )
        self.assertEqual(c2, {"cached": 2, "done": 0, "error": 0})

    def test_workers_error_leaves_uncached(self):
        def boom(email):
            raise RuntimeError("llm down")

        c = pass2.run_pass([self.f1, self.f2], self.cache, self._load, boom, model="m", workers=4)
        self.assertEqual(c["error"], 2)
        self.assertEqual(self.cache.stats()["total"], 0)

    def test_workers_respect_limit(self):
        c = pass2.run_pass(
            [self.f1, self.f2],
            self.cache,
            self._load,
            self._summarize,
            model="m",
            workers=4,
            limit=1,
        )
        self.assertEqual(c["done"], 1)
        self.assertEqual(self.cache.stats()["total"], 1)

    def test_process_file_stores_resilient_identity(self):
        from src.data.blacklist import file_sha256

        def load(path):
            return {"sender": "a", "subject": "s", "date": "z", "body": "b", "message_id": "<M@x>"}

        pass2.run_pass([self.f1], self.cache, load, self._summarize, model="m")
        row = self.cache.get(file_sha256(self.f1))
        self.assertEqual(row["message_id"], "M@x")
        self.assertTrue(row["content_sha256"])


class _FakeEmail:
    def __init__(self, source_id, message_id):
        self.source_id = source_id
        self.message_id = message_id
        self.sender, self.subject, self.date, self.body = "a", "s", None, "b"
        self.summary = None


class TestInjectSummariesResilient(unittest.TestCase):
    def test_injects_via_message_id_when_file_hash_differs(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Pass2Cache(os.path.join(d, "c.db"))
            f = os.path.join(d, "e.eml")
            _write(f, b"re-exported bytes")
            # KEPT row stored under a different (stale) file hash, but the
            # email's normalized Message-ID matches -> must still inject.
            cache.put(
                "STALEFILEHASH",
                {"is_noise": False, "confidence": 1.0, "summary": "hi", "reason": ""},
                message_id="M@x",
            )
            email = _FakeEmail(f, "<M@x>")
            n = pass2.inject_summaries([email], cache)
            self.assertEqual(n, 1)
            self.assertEqual(email.summary, "hi")
            cache.close()


class TestApplyPass2(unittest.TestCase):
    def test_drops_confident_noise_injects_summaries_keeps_unjudged(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Pass2Cache(os.path.join(d, "c.db"))
            fk = os.path.join(d, "keep.eml")
            _write(fk, b"keep bytes")
            fn = os.path.join(d, "noise.eml")
            _write(fn, b"noise bytes")
            fu = os.path.join(d, "unjudged.eml")
            _write(fu, b"unjudged bytes")
            cache.put(
                "HK",
                {"is_noise": False, "confidence": 0.9, "summary": "kept summary", "reason": ""},
                message_id="K@x",
            )
            cache.put(
                "HN",
                {"is_noise": True, "confidence": 0.9, "summary": "", "reason": "spam"},
                message_id="N@x",
            )
            keep = _FakeEmail(fk, "<K@x>")
            noise = _FakeEmail(fn, "<N@x>")
            unjudged = _FakeEmail(fu, "<U@x>")  # no cache row
            kept, dropped = pass2.apply_pass2([keep, noise, unjudged], cache, min_confidence=0.7)
            self.assertEqual(dropped, 1)
            self.assertEqual([e.message_id for e in kept], ["<K@x>", "<U@x>"])
            self.assertEqual(keep.summary, "kept summary")
            self.assertIsNone(unjudged.summary)
            cache.close()

    def test_low_confidence_noise_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Pass2Cache(os.path.join(d, "c.db"))
            f = os.path.join(d, "x.eml")
            _write(f, b"x")
            cache.put(
                "H",
                {"is_noise": True, "confidence": 0.4, "summary": "", "reason": "maybe"},
                message_id="M@x",
            )
            kept, dropped = pass2.apply_pass2([_FakeEmail(f, "<M@x>")], cache, min_confidence=0.7)
            self.assertEqual(dropped, 0)
            self.assertEqual(len(kept), 1)  # below threshold -> kept (zero-loss)
            cache.close()


class TestNoiseHashes(unittest.TestCase):
    def test_returns_shas_above_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Pass2Cache(os.path.join(d, "c.db"))
            cache.put("hi", {"is_noise": True, "confidence": 0.95, "summary": "", "reason": "r"})
            cache.put("lo", {"is_noise": True, "confidence": 0.40, "summary": "", "reason": "r"})
            cache.put("ok", {"is_noise": False, "confidence": 0.99, "summary": "s", "reason": "r"})
            self.assertEqual(pass2.noise_hashes(cache, 0.7), ["hi"])
            cache.close()


class TestAgreementRate(unittest.TestCase):
    def test_fraction_matching_is_noise(self):
        local = {"a": True, "b": False, "c": True}
        ref = {"a": True, "b": True, "c": True}
        self.assertAlmostEqual(pass2.agreement_rate(local, ref), 2 / 3)

    def test_empty_is_zero(self):
        self.assertEqual(pass2.agreement_rate({}, {}), 0.0)


class TestSampleFiles(unittest.TestCase):
    def test_returns_all_when_n_none_or_large(self):
        paths = ["a", "b", "c"]
        self.assertEqual(pass2.sample_files(paths, None), paths)
        self.assertEqual(set(pass2.sample_files(paths, 10)), set(paths))

    def test_samples_n_deterministically_by_seed(self):
        paths = [str(i) for i in range(100)]
        s1 = pass2.sample_files(paths, 10, seed=42)
        s2 = pass2.sample_files(paths, 10, seed=42)
        self.assertEqual(len(s1), 10)
        self.assertEqual(s1, s2)  # same seed -> same sample
        self.assertEqual(len(set(s1)), 10)  # no duplicates
        self.assertTrue(set(s1).issubset(set(paths)))

    def test_different_seed_differs(self):
        paths = [str(i) for i in range(100)]
        self.assertNotEqual(
            pass2.sample_files(paths, 10, seed=1), pass2.sample_files(paths, 10, seed=2)
        )


if __name__ == "__main__":
    unittest.main()
