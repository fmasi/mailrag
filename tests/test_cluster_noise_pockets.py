import json
import unittest

import numpy as np

from src.cluster.noise_pockets import (
    ThreadMeta,
    cluster_threads,
    default_k,
)


def _meta(tid, n_emails, dominant, top_share, n_senders, tag_fraction, subjects, paths):
    return ThreadMeta(
        thread_id=tid,
        n_emails=n_emails,
        dominant_sender=dominant,
        top_sender_share=top_share,
        n_senders=n_senders,
        tag_fraction=tag_fraction,
        sample_subjects=subjects,
        paths=paths,
    )


def _noisy_and_clean(dim=8):
    """20 tight single-sender fully-tagged 'noise' threads near +e0;
    20 diffuse multi-sender untagged 'genuine' threads spread elsewhere."""
    rng = np.random.RandomState(0)
    vecs, metas = [], []
    base = np.zeros(dim)
    base[0] = 1.0
    for i in range(20):
        vecs.append(base + rng.normal(scale=0.01, size=dim))
        metas.append(
            _meta(
                f"noise{i}",
                1,
                "no-reply@bulk.com",
                1.0,
                1,
                1.0,
                ["Your weekly digest"],
                [f"noise{i}.eml"],
            )
        )
    for i in range(20):
        v = rng.normal(scale=1.0, size=dim)
        v[0] = -1.0
        vecs.append(v)
        metas.append(
            _meta(
                f"real{i}",
                3,
                f"alice{i}@work.com",
                0.4,
                3,
                0.0,
                ["Re: project plan"],
                [f"real{i}.eml"],
            )
        )
    return np.array(vecs, dtype=float), metas


class TestNoisePockets(unittest.TestCase):
    def test_default_k_heuristic_and_clamp(self):
        self.assertEqual(default_k(2), 2)  # floor
        self.assertEqual(default_k(200), 10)  # round(sqrt(100))
        self.assertEqual(default_k(100000), 50)  # cap
        self.assertEqual(default_k(3, requested=99), 3)  # clamp to n_threads

    def test_noise_cluster_ranks_top(self):
        vecs, metas = _noisy_and_clean()
        report = cluster_threads(vecs, metas, k=2, seed=11)
        top = report.clusters[0]
        # The top-scored cluster should be the tagged, single-sender, tight one.
        self.assertGreater(top.tag_rate, 0.9)
        self.assertGreater(top.sender_concentration, 0.9)
        self.assertGreater(top.tightness, 0.9)
        self.assertGreater(top.score, report.clusters[-1].score)

    def test_determinism(self):
        vecs, metas = _noisy_and_clean()
        a = cluster_threads(vecs, metas, k=2, seed=11)
        b = cluster_threads(vecs, metas, k=2, seed=11)
        self.assertEqual([c.score for c in a.clusters], [c.score for c in b.clusters])
        self.assertEqual([c.size for c in a.clusters], [c.size for c in b.clusters])

    def test_baseline_and_json_shape(self):
        vecs, metas = _noisy_and_clean()
        report = cluster_threads(vecs, metas, k=2, seed=11)
        self.assertEqual(report.n_threads, 40)
        self.assertAlmostEqual(report.corpus_baseline_tag_rate, 0.5, places=6)
        d = report.to_json_dict(profile="p.json", collection="c", vector_source="fresh")
        # round-trips and carries the documented keys
        d = json.loads(json.dumps(d))
        self.assertEqual(d["vector_source"], "fresh")
        self.assertEqual(d["k"], 2)
        self.assertIn("corpus_baseline_tag_rate", d)
        cl0 = d["clusters"][0]
        for key in (
            "id",
            "size",
            "n_emails",
            "score",
            "components",
            "top_senders",
            "n_senders",
            "sample_subjects",
            "members",
        ):
            self.assertIn(key, cl0)
        self.assertIn("thread_id", cl0["members"][0])
        self.assertIn("paths", cl0["members"][0])

    def test_format_report_returns_text_with_clusters(self):
        from src.cluster.noise_pockets import format_report

        vecs, metas = _noisy_and_clean()
        report = cluster_threads(vecs, metas, k=2, seed=11)
        text = format_report(report, top=5)
        self.assertIn("baseline", text.lower())
        self.assertIn("no-reply@bulk.com", text)  # the noisy cluster's top sender
        self.assertIn("score", text.lower())

    def test_obvious_noise_fraction_and_recommendation_high(self):
        from src.cluster.noise_pockets import recommend_persona

        vecs, metas = _noisy_and_clean()
        report = cluster_threads(vecs, metas, k=2, seed=11)
        # ~half the threads are in the obvious (tagged, tight, single-sender) pocket
        self.assertGreater(report.obvious_noise_fraction, 0.3)
        persona, reason = recommend_persona(report)
        self.assertEqual(persona, "llm-verify")
        self.assertIn("%", reason)

    def test_recommendation_low_noise_picks_llm_all(self):
        from src.cluster.noise_pockets import recommend_persona

        rng = np.random.RandomState(1)
        vecs, metas = [], []
        for i in range(20):
            v = rng.normal(scale=1.0, size=8)
            vecs.append(v)
            metas.append(
                _meta(f"real{i}", 2, f"p{i}@x.com", 0.5, 2, 0.0, ["Re: chat"], [f"r{i}.eml"])
            )
        report = cluster_threads(np.array(vecs, dtype=float), metas, k=2, seed=11)
        self.assertLess(report.obvious_noise_fraction, 0.25)
        persona, _ = recommend_persona(report)
        self.assertEqual(persona, "llm-all")

    def test_recommendation_in_json(self):
        vecs, metas = _noisy_and_clean()
        report = cluster_threads(vecs, metas, k=2, seed=11)
        d = report.to_json_dict(profile="p", collection="c", vector_source="fresh")
        self.assertIn("obvious_noise_fraction", d)
        self.assertIn("recommended_persona", d)


if __name__ == "__main__":
    unittest.main()
