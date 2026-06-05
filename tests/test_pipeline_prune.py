import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.pipeline import prune
from src.data.blacklist import load_blacklist


def _row(sha, conf, reason):
    return {"sha256": sha, "confidence": conf, "reason": reason}


class TestCollect(unittest.TestCase):
    def test_from_judge_cache(self):
        prof = SimpleNamespace(pass2_cache="/tmp/c.db")
        cache = mock.Mock()
        cache.iter_noise.return_value = [_row("h1", 0.9, "newsletter"),
                                         _row("h2", 0.8, "automated alert")]
        with mock.patch("src.pipeline.prune.Pass2Cache", return_value=cache):
            hashes, preview = prune.collect(prof, source="judge", min_confidence=0.7)
        self.assertEqual(hashes, ["h1", "h2"])
        self.assertIn("0.90", preview[0])
        self.assertIn("newsletter", preview[0])
        cache.close.assert_called_once()

    def test_from_tag(self):
        prof = SimpleNamespace(resolved_root=lambda: "/r", selection_rules=[])
        e1 = SimpleNamespace(source_id="/r/a.eml", noise_candidate=True,
                             is_bulk=False, sender="x@bulk.com", subject="Sale")
        e2 = SimpleNamespace(source_id="/r/b.eml", noise_candidate=False,
                             is_bulk=False, sender="y@work.com", subject="Re: plan")
        with mock.patch("src.pipeline.prune.resolve_index_files",
                        return_value=(["/r/a.eml", "/r/b.eml"], [])), \
             mock.patch("src.pipeline.prune.MailArchiveXLoader") as loader, \
             mock.patch("src.pipeline.prune.pass1"), \
             mock.patch("src.pipeline.prune.NoiseFilter"), \
             mock.patch("src.pipeline.prune.file_sha256", return_value="hA"):
            loader.return_value.load.return_value = [e1, e2]
            hashes, preview = prune.collect(prof, source="tag")
        self.assertEqual(hashes, ["hA"])               # only the tagged one
        self.assertIn("x@bulk.com", preview[0])

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            prune.collect(SimpleNamespace(), source="bogus")


class TestRun(unittest.TestCase):
    def _prof(self, bl):
        return SimpleNamespace(pass2_cache="/tmp/c.db", blacklist=bl)

    def test_confirm_false_writes_nothing(self):
        prof = self._prof("/tmp/bl.txt")
        with mock.patch("src.pipeline.prune.collect",
                        return_value=(["h1"], ["preview"])), \
             mock.patch("src.pipeline.prune.append_to_blacklist") as app:
            n = prune.run(prof, source="judge", confirm=lambda preview: False)
        self.assertEqual(n, 0)
        app.assert_not_called()

    def test_confirm_true_appends(self):
        with tempfile.TemporaryDirectory() as d:
            bl = os.path.join(d, "bl.txt")
            prof = self._prof(bl)
            with mock.patch("src.pipeline.prune.collect",
                            return_value=(["h1", "h2"], ["p"])):
                n = prune.run(prof, source="judge", confirm=lambda preview: True)
            self.assertEqual(n, 2)
            self.assertEqual(load_blacklist(bl), {"h1", "h2"})

    def test_empty_drop_set_no_ops(self):
        prof = self._prof("/tmp/bl.txt")
        with mock.patch("src.pipeline.prune.collect", return_value=([], [])), \
             mock.patch("src.pipeline.prune.append_to_blacklist") as app:
            self.assertEqual(prune.run(prof, source="judge"), 0)
        app.assert_not_called()

    def test_no_blacklist_path_raises(self):
        with self.assertRaises(ValueError):
            prune.run(self._prof(None), source="judge")


if __name__ == "__main__":
    unittest.main()
