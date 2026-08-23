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

from src.llm.pass2 import process_file, run_pass


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

    ``file_sha256`` is called at most once per bounded file that hashes
    successfully. A file that raises ``OSError`` in the bounding loop is
    absent from ``sha_of``, so the sweep falls back to a fresh
    ``file_sha256`` call (which also fails); see the OSError tests below. In
    both cases only one slot is consumed from ``remaining`` -- "one attempt"
    is about the bounding loop's own accounting, not the hash call count.
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

    def test_worker_sweep_hashes_bounded_files_including_cached_ones(self):
        """Workers-path mirror of the serial mixed-cache test below: the
        ``if cache.has(cur_sha): _record(path, "cached")`` branch, where
        cur_sha came from sha_of, has no other coverage for workers > 1.
        """
        paths = [f"p{i}" for i in range(10)]
        cache = _Cache(cached=["p0", "p1"])
        with mock.patch("src.llm.pass2.file_sha256", side_effect=lambda p: p) as spy:
            counts = run_pass(
                paths,
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                limit=3,
                workers=4,
            )
        self.assertEqual(counts["cached"], 2)
        self.assertEqual(counts["done"], 3)
        self.assertEqual(spy.call_count, 5)

    def test_serial_sweep_hashes_bounded_files_including_cached_ones(self):
        """The hash-once invariant covers a cached-but-bounded file too: the
        bounding loop must still hash it (to call cache.has), store it in
        sha_of, and the sweep must reuse that hash rather than re-hashing a
        file it's about to report as "cached" anyway.
        """
        paths = [f"p{i}" for i in range(10)]
        cache = _Cache(cached=["p0", "p1"])
        with mock.patch("src.llm.pass2.file_sha256", side_effect=lambda p: p) as spy:
            counts = run_pass(
                paths,
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                limit=3,
                workers=1,
            )
        # Bounding loop hashes p0, p1 (cached, don't consume the limit), p2,
        # p3, p4 (three uncached attempts fill limit=3) -- five files total.
        self.assertEqual(counts["cached"], 2)
        self.assertEqual(counts["done"], 3)
        self.assertEqual(spy.call_count, 5)

    def test_bounding_loop_oserror_is_recorded_as_error(self):
        """An unreadable file under --limit still consumes one attempt (the
        original behavior) and is absent from sha_of, so the sweep falls
        through to file_sha256 again, fails again, and records "error" --
        the only path where the new sha_of lookup changes control flow
        relative to before this fix.
        """
        paths = ["bad", "p1", "p2"]
        cache = _Cache(cached=[])

        def sha_side_effect(p):
            if p == "bad":
                raise OSError("unreadable")
            return p

        with mock.patch("src.llm.pass2.file_sha256", side_effect=sha_side_effect) as spy:
            counts = run_pass(
                paths,
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                limit=3,
                workers=1,
            )
        self.assertEqual(counts["error"], 1)
        self.assertEqual(counts["done"], 2)
        # "bad" is hashed twice: bounding loop (fails) + sweep fallthrough
        # (fails again, since sha_of has no entry for it). "p1" and "p2" are
        # each hashed once, in the bounding loop, and reused by the sweep.
        self.assertEqual(spy.call_count, 4)

    def test_bounding_loop_oserror_is_recorded_as_error_workers(self):
        """The workers path handles a missing sha_of entry inline (before the
        todo-split) rather than delegating to process_file, so it's a
        distinct branch from the serial case above and needs its own pin.
        """
        paths = ["bad", "p1", "p2"]
        cache = _Cache(cached=[])

        def sha_side_effect(p):
            if p == "bad":
                raise OSError("unreadable")
            return p

        with mock.patch("src.llm.pass2.file_sha256", side_effect=sha_side_effect) as spy:
            counts = run_pass(
                paths,
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                limit=3,
                workers=4,
            )
        self.assertEqual(counts["error"], 1)
        self.assertEqual(counts["done"], 2)
        # Same shape as the serial case: "bad" hashed twice (bounding loop +
        # workers path's own inline retry), "p1"/"p2" hashed once each.
        self.assertEqual(spy.call_count, 4)

    def test_process_file_skips_hash_when_sha_supplied(self):
        """process_file's new sha parameter is only exercised indirectly
        through run_pass elsewhere in this file; pin it directly too so a
        future refactor of run_pass can't silently drop the coverage.
        """
        cache = _Cache(cached=[])
        with mock.patch("src.llm.pass2.file_sha256") as spy:
            outcome = process_file(
                "p0",
                cache,
                load_email=self._load,
                summarize=self._summarize,
                model="test-model",
                sha="precomputed-hash",
            )
        spy.assert_not_called()
        self.assertEqual(outcome, "done")

    def test_process_file_sha_supplied_load_email_oserror(self):
        """The other half of the docstring's TOCTOU note: a file that
        disappears between the bounding loop's hash and this call must still
        resolve to "error" through classify_failure, without ever re-hashing
        (there's nothing left to hash).
        """
        cache = _Cache(cached=[])

        def bad_load(path):
            raise FileNotFoundError(f"gone: {path}")

        with mock.patch("src.llm.pass2.file_sha256") as spy:
            outcome = process_file(
                "p0",
                cache,
                load_email=bad_load,
                summarize=self._summarize,
                model="test-model",
                sha="precomputed-hash",
            )
        spy.assert_not_called()
        self.assertEqual(outcome, "error")
