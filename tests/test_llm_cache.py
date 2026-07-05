# tests/test_llm_cache.py
"""Tests for the content-addressed SQLite Pass-2 cache (stdlib-only)."""

import os
import tempfile
import unittest

from src.llm.cache import Pass2Cache


def _rec(is_noise, conf, summary="s", reason="r"):
    return {"is_noise": is_noise, "confidence": conf, "summary": summary, "reason": reason}


class TestPass2Cache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = Pass2Cache(os.path.join(self.dir.name, "c.db"))

    def tearDown(self):
        self.cache.close()
        self.dir.cleanup()

    def test_put_get_roundtrip(self):
        self.cache.put("sha1", _rec(False, 0.9, "hello"), model="gemma")
        row = self.cache.get("sha1")
        self.assertEqual(row["summary"], "hello")
        self.assertEqual(row["is_noise"], 0)
        self.assertAlmostEqual(row["confidence"], 0.9)
        self.assertEqual(row["model"], "gemma")
        self.assertTrue(row["created_at"])

    def test_has_and_resumability(self):
        self.assertFalse(self.cache.has("x"))
        self.cache.put("x", _rec(True, 1.0))
        self.assertTrue(self.cache.has("x"))

    def test_put_replaces(self):
        self.cache.put("k", _rec(False, 0.1, "old"))
        self.cache.put("k", _rec(False, 0.2, "new"))
        self.assertEqual(self.cache.get("k")["summary"], "new")

    def test_iter_noise_threshold_and_order(self):
        self.cache.put("a", _rec(True, 0.95))
        self.cache.put("b", _rec(True, 0.60))
        self.cache.put("c", _rec(False, 0.99))
        shas = [r["sha256"] for r in self.cache.iter_noise(min_confidence=0.7)]
        self.assertEqual(shas, ["a"])  # b below threshold, c not noise

    def test_stats(self):
        self.cache.put("a", _rec(True, 0.9))
        self.cache.put("b", _rec(False, 0.9))
        self.assertEqual(self.cache.stats(), {"total": 2, "noise": 1, "kept": 1})

    def test_missing_get_returns_none(self):
        self.assertIsNone(self.cache.get("nope"))


class TestPass2CacheResilientIdentity(unittest.TestCase):
    """Stable fallback identifiers so a re-export reuses the cache."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "c.db")
        self.cache = Pass2Cache(self.path)

    def tearDown(self):
        self.cache.close()
        self.dir.cleanup()

    def test_put_stores_identity_columns(self):
        self.cache.put(
            "filehash",
            _rec(False, 1.0, "s"),
            model="g",
            message_id="<mid@x>",
            content_sha256="chash",
        )
        row = self.cache.get("filehash")
        self.assertEqual(row["message_id"], "<mid@x>")
        self.assertEqual(row["content_sha256"], "chash")

    def test_get_resilient_falls_back_to_message_id(self):
        self.cache.put(
            "filehash", _rec(False, 1.0, "s"), message_id="<mid@x>", content_sha256="chash"
        )
        # file hash changed (re-export) but Message-ID is stable
        row = self.cache.get_resilient("DIFFERENT", message_id="<mid@x>")
        self.assertIsNotNone(row)
        self.assertEqual(row["summary"], "s")

    def test_get_resilient_falls_back_to_content_sha256(self):
        self.cache.put(
            "filehash", _rec(False, 1.0, "s"), message_id="<mid@x>", content_sha256="chash"
        )
        # both file hash and Message-ID differ; content hash still matches
        row = self.cache.get_resilient("DIFFERENT", message_id="<other>", content_sha256="chash")
        self.assertIsNotNone(row)
        self.assertEqual(row["summary"], "s")

    def test_get_resilient_prefers_exact_file_hash(self):
        self.cache.put("fileA", _rec(False, 1.0, "A"), message_id="<a>")
        self.cache.put("fileB", _rec(False, 1.0, "B"), message_id="<b>")
        row = self.cache.get_resilient("fileA", message_id="<b>")
        self.assertEqual(row["summary"], "A")  # file hash wins over message_id

    def test_get_resilient_returns_none_when_nothing_matches(self):
        self.cache.put("filehash", _rec(False, 1.0, "s"), message_id="<mid@x>")
        self.assertIsNone(self.cache.get_resilient("nope", message_id="<no>", content_sha256="no"))

    def test_set_identity_backfills_existing_row(self):
        self.cache.put("filehash", _rec(False, 1.0, "s"))  # no identity yet
        self.assertIsNone(self.cache.get("filehash")["message_id"])
        self.cache.set_identity("filehash", "<mid@x>", "chash")
        row = self.cache.get_resilient("DIFFERENT", message_id="<mid@x>")
        self.assertEqual(row["summary"], "s")


class TestPass2CacheMigration(unittest.TestCase):
    """Opening a legacy (pre-identity) DB must add the new columns in place."""

    def _make_old_db(self, path):
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE pass2 (
                   sha256 TEXT PRIMARY KEY, summary TEXT NOT NULL DEFAULT '',
                   is_noise INTEGER NOT NULL, confidence REAL NOT NULL,
                   reason TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
                   created_at TEXT NOT NULL)"""
        )
        conn.execute("INSERT INTO pass2 VALUES ('old1','sum',0,1.0,'','g','2026-01-01')")
        conn.commit()
        conn.close()

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "old.db")
        self._make_old_db(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_migration_adds_identity_columns(self):
        cache = Pass2Cache(self.path)
        cols = {r[1] for r in cache._conn.execute("PRAGMA table_info(pass2)")}
        self.assertIn("message_id", cols)
        self.assertIn("content_sha256", cols)
        cache.close()

    def test_migration_preserves_existing_rows(self):
        cache = Pass2Cache(self.path)
        self.assertEqual(cache.get("old1")["summary"], "sum")
        self.assertIsNone(cache.get("old1")["message_id"])  # backfillable
        cache.close()

    def test_migration_is_idempotent(self):
        Pass2Cache(self.path).close()
        cache = Pass2Cache(self.path)  # second open must not error
        self.assertEqual(cache.stats()["total"], 1)
        cache.close()


if __name__ == "__main__":
    unittest.main()
