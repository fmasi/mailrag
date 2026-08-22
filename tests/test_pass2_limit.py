"""The pass-2 limit must bound WORK, not the file list.

Slicing `paths[:limit]` caps how many files are considered. On a corpus that is
mostly cached — the normal case, since the sweep is resumable — that means a
"limit 1500" run examines 1,500 already-done files and performs nothing. The
whole reason to bound this sweep is that the remaining emails cost money or
hours, so the limit has to count the ones that actually incur it.

Observed for real: a bounded paid run reported `cached=1500, done=0` and spent
nothing, verifying nothing.
"""

import unittest
from unittest import mock

from src.llm.pass2 import run_pass


class _Cache:
    def __init__(self, cached):
        self._cached = set(cached)
        self.puts = []

    def has(self, sha):
        return sha in self._cached

    def put(self, sha, record, **kw):
        self.puts.append(sha)


class TestLimitBoundsWork(unittest.TestCase):
    def setUp(self):
        # sha == path, so "cached" is easy to express.
        self.calls = []

    def _run(self, paths, cached, limit):
        cache = _Cache(cached)

        def summarize(email):
            self.calls.append(email["source_id"])
            return {"summary": "s", "is_noise": False, "confidence": 0.1, "reason": "r"}

        with mock.patch("src.llm.pass2.file_sha256", side_effect=lambda p: p):
            return run_pass(
                paths,
                cache,
                load_email=lambda p: {
                    "source_id": p,
                    "body": "b",
                    "subject": "s",
                    "sender": "x",
                    "message_id": p,
                    "date": "",
                    "to": "",
                },
                summarize=summarize,
                model="test-model",
                progress=False,
                limit=limit,
            )

    def test_cached_files_do_not_consume_the_limit(self):
        paths = [f"p{i}" for i in range(10)]
        counts = self._run(paths, cached=paths[:6], limit=2)
        self.assertEqual(counts["done"], 2, "two uncached emails should be processed")
        self.assertEqual(len(self.calls), 2)

    def test_a_fully_cached_prefix_does_not_exhaust_the_run(self):
        # The observed failure: limit consumed entirely by already-done files.
        paths = [f"p{i}" for i in range(2000)]
        counts = self._run(paths, cached=paths[:1500], limit=5)
        self.assertEqual(counts["done"], 5)

    def test_no_limit_processes_everything_uncached(self):
        paths = [f"p{i}" for i in range(6)]
        counts = self._run(paths, cached=["p0"], limit=None)
        self.assertEqual(counts["done"], 5)

    def test_limit_larger_than_the_work_is_harmless(self):
        paths = [f"p{i}" for i in range(4)]
        counts = self._run(paths, cached=[], limit=99)
        self.assertEqual(counts["done"], 4)


class TestLimitHashesEachFileOnce(unittest.TestCase):
    """GH #184: the bounding loop reads+hashes every candidate to decide cache
    coverage, then the sweep that follows used to read+hash each of those same
    files AGAIN before doing any work — doubling disk I/O and CPU on every
    --limit-bounded run, exactly the run --limit exists to make cheap.

    file_sha256 must be called exactly once per bounded file, regardless of
    worker count.
    """

    def _summarize(self, email):
        return {"summary": "s", "is_noise": False, "confidence": 0.1, "reason": "r"}

    def _load(self, path):
        return {
            "source_id": path,
            "body": "b",
            "subject": "s",
            "sender": "x",
            "message_id": path,
            "date": "",
            "to": "",
        }

    def test_serial_sweep_hashes_each_bounded_file_once(self):
        from src.llm.pass2 import run_pass

        paths = [f"p{i}" for i in range(10)]
        cache = _Cache(cached=[])
        with mock.patch("src.llm.pass2.file_sha256", side_effect=lambda p: p) as spy:
            counts = run_pass(
                paths,
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                limit=5,
                workers=1,
            )
        self.assertEqual(counts["done"], 5)
        # One hash per bounded file: the bounding loop's hash must be reused by
        # the sweep, not recomputed.
        self.assertEqual(spy.call_count, 5)

    def test_worker_sweep_hashes_each_bounded_file_once(self):
        from src.llm.pass2 import run_pass

        paths = [f"p{i}" for i in range(10)]
        cache = _Cache(cached=[])
        with mock.patch("src.llm.pass2.file_sha256", side_effect=lambda p: p) as spy:
            counts = run_pass(
                paths,
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                limit=5,
                workers=4,
            )
        self.assertEqual(counts["done"], 5)
        self.assertEqual(spy.call_count, 5)
