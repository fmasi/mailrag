import json, os, tempfile, unittest
from unittest import mock
from src.profile import CorpusProfile


class TestCliDispatch(unittest.TestCase):
    def _profile_file(self, d):
        fp = os.path.join(d, "p.json")
        CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}],
                      collection="c", chunk_size=512).save(fp)
        return fp

    def test_pass1_verb_previews_partition(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            with mock.patch("src.cli.resolve_index_files", return_value=(["/r/a/x.eml"], [])), \
                 mock.patch("src.cli.MailArchiveXLoader") as Loader, \
                 mock.patch("src.cli.NoiseFilter") as NF:
                from src.data.models import NormalizedEmail
                Loader.return_value.load.return_value = [
                    NormalizedEmail(sender="a@junk.example", subject="s", date=None,
                                    body="b", source="t", source_id="t0")]
                NF.from_project_rules.return_value.matched_category.return_value = "junk"
                rc = cli.main(["pass1", "--profile", fp])
        self.assertEqual(rc, 0)

    def test_build_verb_saves_profile(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            with mock.patch("src.cli.build_stage") as bs, \
                 mock.patch("src.cli.BgeM3Embedder"):
                bs.run.return_value = mock.Mock(chunks=10, collection="c")
                rc = cli.main(["build", "--profile", fp, "--limit", "1"])
        self.assertEqual(rc, 0)
        self.assertTrue(bs.run.called)


if __name__ == "__main__":
    unittest.main()
