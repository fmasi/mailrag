import unittest
from unittest import mock

from src.data.models import NormalizedEmail
from src.profile import CorpusProfile


def _email(sender):
    return NormalizedEmail(
        sender=sender, subject="s", date=None, body="b", source="t", source_id="t0"
    )


class TestBuildStage(unittest.TestCase):
    def test_runs_pass1_and_disables_internal_filter(self):
        from src.pipeline import build

        prof = CorpusProfile(
            root="/r",
            selection_rules=[{"type": "prefix", "value": "a/"}],
            chunk_size=1024,
            collection="c",
            qdrant_url="http://x",
        )
        emails = [_email("a@junk.example"), _email("b@real.example")]
        with (
            mock.patch("src.pipeline.build.resolve_index_files", return_value=(["/r/a/x.eml"], [])),
            mock.patch("src.pipeline.build.MailArchiveXLoader") as Loader,
            mock.patch("src.pipeline.build.build_contextual_index") as bci,
            mock.patch("src.pipeline.build.NoiseFilter") as NF,
        ):
            Loader.return_value.load.return_value = emails
            NF.from_project_rules.return_value.matched_category.side_effect = lambda e: (
                "junk" if "junk" in e.sender else None
            )
            build.run(prof, embedder=mock.Mock(), recreate=True)
        _, kwargs = bci.call_args
        self.assertIs(kwargs["apply_noise_filter"], False)
        self.assertEqual(kwargs["collection"], "c")
        self.assertEqual(kwargs["chunk_size"], 1024)
        self.assertTrue(emails[0].noise_candidate)
        self.assertFalse(emails[1].noise_candidate)

    def test_embed_summary_applies_pass2_cache_and_drops_noise(self):
        from src.pipeline import build

        prof = CorpusProfile(
            root="/r",
            selection_rules=[{"type": "prefix", "value": "a/"}],
            chunk_size=1024,
            collection="c",
            qdrant_url="http://x",
            pass2_cache="/tmp/p.db",
        )
        emails = [_email("a@junk.example"), _email("b@real.example")]
        kept = [emails[1]]  # apply_pass2 drops the noise one
        with (
            mock.patch("src.pipeline.build.resolve_index_files", return_value=(["/r/a/x.eml"], [])),
            mock.patch("src.pipeline.build.MailArchiveXLoader") as Loader,
            mock.patch("src.pipeline.build.build_contextual_index") as bci,
            mock.patch("src.pipeline.build.NoiseFilter") as NF,
            mock.patch("src.pipeline.build.Pass2Cache") as Cache,
            mock.patch("src.pipeline.build.apply_pass2", return_value=(kept, 1)) as ap,
        ):
            Loader.return_value.load.return_value = emails
            NF.from_project_rules.return_value.matched_category.return_value = None
            build.run(
                prof,
                embedder=mock.Mock(),
                recreate=True,
                embed_summary=True,
                noise_min_confidence=0.7,
            )
        # cache opened on the profile's path; apply_pass2 invoked with the threshold
        Cache.assert_called_once_with("/tmp/p.db")
        self.assertEqual(ap.call_args.kwargs["min_confidence"], 0.7)
        # only the kept (non-noise) emails are indexed, with summaries embedded
        args, kwargs = bci.call_args
        self.assertEqual(args[0], kept)
        self.assertIs(kwargs["embed_summary"], True)

    def test_attachment_docs_are_built_and_passed_as_extra_docs(self):
        """build.run extracts attachment docs from the indexed emails' paths and
        hands them to build_contextual_index as extra_docs (issue #80)."""
        from src.pipeline import build

        prof = CorpusProfile(
            root="/r",
            selection_rules=[{"type": "prefix", "value": "a/"}],
            collection="c",
        )
        e = _email("b@real.example")
        e.source_id = "/r/a/x.eml"  # loader sets source_id to the .eml path
        sentinel_docs = [object()]
        with (
            mock.patch("src.pipeline.build.resolve_index_files", return_value=(["/r/a/x.eml"], [])),
            mock.patch("src.pipeline.build.MailArchiveXLoader") as Loader,
            mock.patch("src.pipeline.build.build_contextual_index") as bci,
            mock.patch("src.pipeline.build.NoiseFilter") as NF,
            mock.patch(
                "src.pipeline.build.build_attachment_documents", return_value=sentinel_docs
            ) as bad,
        ):
            Loader.return_value.load.return_value = [e]
            NF.from_project_rules.return_value.matched_category.return_value = None
            build.run(prof, embedder=mock.Mock(), recreate=True)
        # attachments extracted from the .eml path of the indexed email
        self.assertEqual(bad.call_args[0][0], ["/r/a/x.eml"])
        # and forwarded to the indexer as extra_docs
        self.assertIs(bci.call_args.kwargs["extra_docs"], sentinel_docs)

    def test_index_attachments_false_skips_extraction(self):
        from src.pipeline import build

        prof = CorpusProfile(root="/r", selection_rules=[], collection="c")
        with (
            mock.patch("src.pipeline.build.resolve_index_files", return_value=(["/r/a/x.eml"], [])),
            mock.patch("src.pipeline.build.MailArchiveXLoader") as Loader,
            mock.patch("src.pipeline.build.build_contextual_index") as bci,
            mock.patch("src.pipeline.build.NoiseFilter") as NF,
            mock.patch("src.pipeline.build.build_attachment_documents") as bad,
        ):
            Loader.return_value.load.return_value = [_email("b@real.example")]
            NF.from_project_rules.return_value.matched_category.return_value = None
            build.run(prof, embedder=mock.Mock(), recreate=True, index_attachments=False)
        bad.assert_not_called()
        self.assertIsNone(bci.call_args.kwargs["extra_docs"])

    def test_no_embed_summary_skips_pass2_cache(self):
        from src.pipeline import build

        prof = CorpusProfile(
            root="/r",
            selection_rules=[{"type": "prefix", "value": "a/"}],
            collection="c",
            pass2_cache="/tmp/p.db",
        )
        with (
            mock.patch("src.pipeline.build.resolve_index_files", return_value=(["/r/a/x.eml"], [])),
            mock.patch("src.pipeline.build.MailArchiveXLoader") as Loader,
            mock.patch("src.pipeline.build.build_contextual_index") as bci,
            mock.patch("src.pipeline.build.NoiseFilter") as NF,
            mock.patch("src.pipeline.build.Pass2Cache") as Cache,
        ):
            Loader.return_value.load.return_value = [_email("a@real.example")]
            NF.from_project_rules.return_value.matched_category.return_value = None
            build.run(prof, embedder=mock.Mock(), recreate=True)  # embed_summary defaults False
        Cache.assert_not_called()
        self.assertIs(bci.call_args.kwargs["embed_summary"], False)


if __name__ == "__main__":
    unittest.main()
