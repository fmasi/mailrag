import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.cluster.noise_pockets import ClusterReport
from src.pipeline.explore import aggregate_threads, run


def _email(message_id, subject, sender, tagged, in_reply_to="", references=""):
    return SimpleNamespace(
        message_id=message_id,
        subject=subject,
        sender=sender,
        in_reply_to=in_reply_to,
        references=references,
        body="b",
        source="mail_archive_x",
        source_id=f"{message_id}.eml",
        noise_candidate=tagged,
        is_bulk=False,
    )


def _emails_three():
    return [
        _email("<a@x>", "Project plan", "alice@work.com", False),
        _email("<b@x>", "Re: Project plan", "bob@work.com", False, in_reply_to="<a@x>"),
        _email("<n@x>", "Weekly digest", "no-reply@bulk.com", True),
    ]


class TestAggregateThreads(unittest.TestCase):
    def test_groups_by_thread_and_computes_metrics(self):
        emails = _emails_three()
        metas, tid_to_idx = aggregate_threads(emails)
        self.assertEqual(len(metas), 2)  # 2 distinct threads
        by_sub = {m.sample_subjects[0]: m for m in metas}
        digest = by_sub["Weekly digest"]
        self.assertEqual(digest.n_emails, 1)
        self.assertEqual(digest.tag_fraction, 1.0)
        self.assertEqual(digest.dominant_sender, "no-reply@bulk.com")
        self.assertEqual(digest.top_sender_share, 1.0)
        convo = by_sub["Project plan"]
        self.assertEqual(convo.n_emails, 2)
        self.assertEqual(convo.tag_fraction, 0.0)
        self.assertEqual(convo.n_senders, 2)
        self.assertEqual(convo.top_sender_share, 0.5)
        # members carry the .eml file paths (source_id), not the loader type
        self.assertEqual(set(convo.paths), {"<a@x>.eml", "<b@x>.eml"})
        # the index map points back to the right rows
        self.assertEqual(sorted(len(v) for v in tid_to_idx.values()), [1, 2])


class TestExploreRun(unittest.TestCase):
    def _profile(self):
        prof = SimpleNamespace(
            collection="c", qdrant_url="http://localhost:6333", selection_rules=[], chunk_size=512
        )
        prof.resolved_root = lambda: "/root"
        return prof

    def test_uses_qdrant_when_collection_exists(self):
        from src.pipeline.explore import _thread_id

        prof = self._profile()
        client = mock.Mock()
        client.collection_exists.return_value = True
        emails = _emails_three()
        tids = {_thread_id(e) for e in emails}
        tvecs = {tid: np.array([1.0, 0.0]) for tid in tids}
        with (
            mock.patch(
                "src.pipeline.explore.resolve_index_files", return_value=(["/root/a.eml"], [])
            ),
            mock.patch("src.pipeline.explore._load_emails", return_value=emails),
            mock.patch("src.pipeline.explore.get_client", return_value=client),
            mock.patch("src.pipeline.explore.read_thread_vectors", return_value=tvecs) as rtv,
            mock.patch("src.pipeline.explore.BgeM3Embedder") as emb,
            tempfile.TemporaryDirectory() as d,
        ):
            out = os.path.join(d, "e.json")
            report = run(prof, json_path=out, seed=11)
            with open(out) as fh:
                written = json.load(fh)
        rtv.assert_called_once()
        emb.assert_not_called()  # reused vectors, no embed
        self.assertIsInstance(report, ClusterReport)
        self.assertEqual(report.vector_source, "qdrant")
        self.assertEqual(written["vector_source"], "qdrant")

    def test_falls_back_to_fresh_when_no_collection(self):
        prof = self._profile()
        client = mock.Mock()
        client.collection_exists.return_value = False
        emails = _emails_three()
        embedder = mock.Mock()
        embedder.encode.return_value = (np.array([[1.0, 0.0]] * 3), [{}, {}, {}])
        with (
            mock.patch(
                "src.pipeline.explore.resolve_index_files", return_value=(["/root/a.eml"], [])
            ),
            mock.patch("src.pipeline.explore._load_emails", return_value=emails),
            mock.patch("src.pipeline.explore.get_client", return_value=client),
            mock.patch("src.pipeline.explore.read_thread_vectors") as rtv,
            mock.patch("src.pipeline.explore.BgeM3Embedder", return_value=embedder),
            tempfile.TemporaryDirectory() as d,
        ):
            report = run(prof, json_path=os.path.join(d, "e.json"), seed=11)
        rtv.assert_not_called()
        embedder.encode.assert_called_once()
        self.assertEqual(report.vector_source, "fresh")

    def test_force_fresh_skips_qdrant_even_if_collection_exists(self):
        prof = self._profile()
        client = mock.Mock()
        client.collection_exists.return_value = True
        emails = _emails_three()
        embedder = mock.Mock()
        embedder.encode.return_value = (np.array([[1.0, 0.0]] * 3), [{}, {}, {}])
        with (
            mock.patch(
                "src.pipeline.explore.resolve_index_files", return_value=(["/root/a.eml"], [])
            ),
            mock.patch("src.pipeline.explore._load_emails", return_value=emails),
            mock.patch("src.pipeline.explore.get_client", return_value=client),
            mock.patch("src.pipeline.explore.read_thread_vectors") as rtv,
            mock.patch("src.pipeline.explore.BgeM3Embedder", return_value=embedder),
            tempfile.TemporaryDirectory() as d,
        ):
            report = run(prof, json_path=os.path.join(d, "e.json"), seed=11, force_fresh=True)
        rtv.assert_not_called()
        embedder.encode.assert_called_once()
        self.assertEqual(report.vector_source, "fresh")

    def test_empty_selection_raises(self):
        prof = self._profile()
        with (
            mock.patch("src.pipeline.explore.resolve_index_files", return_value=([], [])),
            tempfile.TemporaryDirectory() as d,
        ):
            with self.assertRaises(ValueError):
                run(prof, json_path=os.path.join(d, "e.json"))


if __name__ == "__main__":
    unittest.main()
