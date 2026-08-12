"""The public Enron-QA benchmark's scoring logic (issue #97).

Only the pure parts are unit-tested here: fixture loading, row->email parsing,
rank/recall arithmetic. Building a collection and searching it needs Qdrant and
bge-m3 weights, which is what `make bench` is for.

These tests exist because a benchmark that silently miscounts is worse than no
benchmark — it produces a confident, wrong, *public* number.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.eval.bench_public import (
    gold_rank,
    load_fixtures,
    mcnemar_exact,
    recall_at_k,
    to_email,
    wilson_interval,
)
from scripts.eval.gen_public_benchset import SIZES, select


class _Node:
    """Stand-in for a LlamaIndex NodeWithScore — only `.metadata` is read."""

    def __init__(self, message_id):
        self.metadata = {"message_id": message_id}


class TestGoldRank(unittest.TestCase):
    def test_returns_the_position_of_the_gold_document(self):
        hits = [_Node("a"), _Node("b"), _Node("gold")]
        self.assertEqual(gold_rank(hits, "gold"), 2)

    def test_absent_gold_is_none_not_zero(self):
        self.assertIsNone(gold_rank([_Node("a"), _Node("b")], "gold"))

    def test_no_hits_at_all_is_none(self):
        self.assertIsNone(gold_rank([], "gold"))

    def test_chunks_of_one_email_count_as_a_single_document(self):
        """The metric is document recall. Three chunks of email 'a' ahead of the
        gold must leave the gold at rank 1 — counting them separately would
        inflate recall@k for every arm and make the public number wrong."""
        hits = [_Node("a"), _Node("a"), _Node("a"), _Node("gold")]
        self.assertEqual(gold_rank(hits, "gold"), 1)

    def test_the_first_occurrence_of_the_gold_wins(self):
        hits = [_Node("gold"), _Node("x"), _Node("gold")]
        self.assertEqual(gold_rank(hits, "gold"), 0)

    def test_rank_zero_is_a_hit_at_k_equals_one(self):
        """Guards the off-by-one between 0-indexed ranks and 1-indexed @k."""
        self.assertEqual(recall_at_k([0], 1, 1), 100.0)


class TestRecallAtK(unittest.TestCase):
    def test_counts_only_ranks_strictly_inside_k(self):
        # ranks 0..4; @5 admits 0-4, @1 admits only 0
        ranks = [0, 1, 2, 3, 4]
        self.assertEqual(recall_at_k(ranks, 5, 5), 100.0)
        self.assertEqual(recall_at_k(ranks, 1, 5), 20.0)

    def test_a_rank_equal_to_k_is_outside_the_cut(self):
        self.assertEqual(recall_at_k([5], 5, 1), 0.0)

    def test_misses_are_counted_in_the_denominator(self):
        """A None (gold never retrieved) must lower recall, not be skipped."""
        self.assertEqual(recall_at_k([0, None], 5, 2), 50.0)

    def test_no_queries_is_zero_not_a_zero_division(self):
        self.assertEqual(recall_at_k([], 5, 0), 0.0)


class TestWilsonInterval(unittest.TestCase):
    def test_zero_n_is_a_degenerate_interval(self):
        self.assertEqual(wilson_interval(50.0, 0), (0.0, 0.0))

    def test_a_larger_sample_gives_a_tighter_interval(self):
        wide = wilson_interval(50.0, 100)
        tight = wilson_interval(50.0, 400)
        self.assertLess(tight[1] - tight[0], wide[1] - wide[0])

    def test_the_point_estimate_lies_inside_its_own_interval(self):
        for p in (0.0, 25.0, 76.1, 97.5, 100.0):
            with self.subTest(p=p):
                lo, hi = wilson_interval(p, 360)
                self.assertLessEqual(lo, p + 1e-9)
                self.assertGreaterEqual(hi, p - 1e-9)

    def test_a_perfect_score_still_carries_uncertainty(self):
        """The reason this is Wilson and not Wald. Wald gives ±0.00 at p=1 and
        would claim certainty from 360 samples; Wilson keeps a real lower bound."""
        lo, hi = wilson_interval(100.0, 360)
        self.assertAlmostEqual(hi, 100.0)
        self.assertLess(lo, 99.5)
        self.assertGreater(lo, 98.0)

    def test_the_interval_is_asymmetric_near_the_ceiling(self):
        """Why bounds are reported instead of a single ±: at the benchmark's own
        recall values the two sides differ by about a percentage point."""
        p = 97.5
        lo, hi = wilson_interval(p, 360)
        self.assertGreater((p - lo) - (hi - p), 0.5)

    def test_it_is_never_wider_than_the_possible_range(self):
        for p in (0.0, 100.0):
            with self.subTest(p=p):
                lo, hi = wilson_interval(p, 360)
                self.assertGreaterEqual(lo, 0.0)
                self.assertLessEqual(hi, 100.0)

    def test_matches_the_published_wilson_value(self):
        # p=0.5, n=100 -> [40.4, 59.6] (standard worked example)
        lo, hi = wilson_interval(50.0, 100)
        self.assertAlmostEqual(lo, 40.4, places=1)
        self.assertAlmostEqual(hi, 59.6, places=1)


class TestMcNemarExact(unittest.TestCase):
    def test_counts_only_the_queries_the_arms_disagree_on(self):
        # q0: both hit (concordant)  q1: only A  q2,q3: only B  q4: both miss
        a = [0, 0, None, None, None]
        b = [0, None, 0, 0, None]
        b_count, c_count, _ = mcnemar_exact(a, b, 5)
        self.assertEqual((b_count, c_count), (1, 2))

    def test_no_disagreement_is_p_one_not_a_division_by_zero(self):
        ranks = [0, 1, None]
        self.assertEqual(mcnemar_exact(ranks, ranks, 5), (0, 0, 1.0))

    def test_a_one_sided_sweep_is_significant(self):
        """10 fixes and 0 breaks: p = 2 * 0.5^10 ≈ 0.002."""
        a = [None] * 10
        b = [0] * 10
        _, _, p = mcnemar_exact(a, b, 5)
        self.assertAlmostEqual(p, 2 * 0.5**10, places=6)

    def test_an_even_split_is_not_significant(self):
        a = [0] * 5 + [None] * 5
        b = [None] * 5 + [0] * 5
        _, _, p = mcnemar_exact(a, b, 5)
        self.assertEqual(p, 1.0)

    def test_the_k_cut_decides_what_counts_as_a_hit(self):
        """A rank of 7 is a hit @10 but a miss @5, so the same ranks give
        different discordant counts at different k."""
        a, b = [7], [0]
        self.assertEqual(mcnemar_exact(a, b, 5)[:2], (0, 1))
        self.assertEqual(mcnemar_exact(a, b, 10)[:2], (0, 0))

    def test_p_never_exceeds_one(self):
        a = [0, None, 0, None]
        b = [None, 0, None, 0]
        self.assertLessEqual(mcnemar_exact(a, b, 5)[2], 1.0)


class TestToEmail(unittest.TestCase):
    def test_pulls_subject_and_sender_out_of_the_header_block(self):
        row = {
            "email": "Subject: Ameren\nSender: a@enron.com\nRecipients: []\n\nbody text here",
            "path": "user/sent/1.",
        }
        em = to_email(row)
        self.assertEqual(em.subject, "Ameren")
        self.assertEqual(em.sender, "a@enron.com")

    def test_message_id_is_the_dataset_path_because_that_is_the_gold_key(self):
        em = to_email({"email": "Subject: x\n\nbody", "path": "user/sent/7."})
        self.assertEqual(em.message_id, "user/sent/7.")

    def test_body_is_taken_after_the_rule_when_present(self):
        row = {"email": "Subject: x\n=====\n" + ("real body " * 10), "path": "p"}
        self.assertNotIn("Subject:", to_email(row).body)

    def test_a_too_short_tail_falls_back_to_the_whole_text(self):
        """A rule with almost nothing after it is a formatting artefact, not a
        body; splitting on it would throw the email away."""
        row = {"email": "Subject: x\nSender: s@e.com\n=====\nhi", "path": "p"}
        self.assertIn("Subject: x", to_email(row).body)

    def test_missing_headers_fall_back_without_raising(self):
        em = to_email({"email": "no headers at all, just text", "path": "p"})
        self.assertEqual(em.subject, "")
        self.assertEqual(em.sender, "someone@enron.com")


class TestLoadFixtures(unittest.TestCase):
    def test_reads_the_manifest_and_queries(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "enron_qa_standard_corpus.txt").write_text("a\nb\n\nc\n")
            (root / "enron_qa_standard_queries.jsonl").write_text(
                json.dumps({"query": "q", "answer_path": "a", "category": "enron-qa"}) + "\n"
            )
            with mock.patch("scripts.eval.bench_public.FIXTURES", root):
                paths, queries = load_fixtures("standard")
        self.assertEqual(paths, ["a", "b", "c"])  # blank line skipped
        self.assertEqual(queries[0]["answer_path"], "a")

    def test_missing_fixtures_exit_with_the_regeneration_command(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("scripts.eval.bench_public.FIXTURES", pathlib.Path(d)):
                with self.assertRaises(SystemExit) as cm:
                    load_fixtures("standard")
        self.assertIn("gen_public_benchset", str(cm.exception))


class TestSelectionIsDeterministic(unittest.TestCase):
    def _rows(self, n=50):
        return [
            {"path": f"u/sent/{i}.", "questions": [f"q{i}"], "email": "x" * 200} for i in range(n)
        ]

    def test_same_rows_give_the_same_selection(self):
        a = select(self._rows(), 20, 5)
        b = select(self._rows(), 20, 5)
        self.assertEqual([r["path"] for r in a[0]], [r["path"] for r in b[0]])
        self.assertEqual(a[1], b[1])

    def test_selection_does_not_depend_on_input_order(self):
        """The corpus is sorted by path before shuffling, so an upstream
        reordering must not change which documents the benchmark scores."""
        rows = self._rows()
        forward = select(rows, 20, 5)
        backward = select(list(reversed(rows)), 20, 5)
        self.assertEqual([r["path"] for r in forward[0]], [r["path"] for r in backward[0]])

    def test_every_gold_path_is_present_in_the_corpus(self):
        """If a query's gold document is not indexed, that query is unanswerable
        and silently drags recall down for every arm."""
        corpus, queries = select(self._rows(), 20, 5)
        self.assertTrue({q["answer_path"] for q in queries} <= {r["path"] for r in corpus})

    def test_rows_without_a_question_or_body_are_excluded(self):
        rows = self._rows(10) + [
            {"path": "u/sent/no-q.", "questions": [], "email": "x" * 200},
            {"path": "u/sent/short.", "questions": ["q"], "email": "tiny"},
        ]
        corpus, _ = select(rows, 50, 5)
        paths = {r["path"] for r in corpus}
        self.assertNotIn("u/sent/no-q.", paths)
        self.assertNotIn("u/sent/short.", paths)


class TestCommittedFixturesMatchTheDeclaredSizes(unittest.TestCase):
    """The fixtures are committed, so drift between them and SIZES is possible
    and would silently change the published number."""

    def test_each_size_has_fixtures_of_the_declared_length(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "eval" / "public"
        for size, (n_corpus, n_queries) in SIZES.items():
            with self.subTest(size=size):
                corpus = (root / f"enron_qa_{size}_corpus.txt").read_text().split()
                queries = [
                    json.loads(ln)
                    for ln in (root / f"enron_qa_{size}_queries.jsonl").read_text().splitlines()
                    if ln.strip()
                ]
                self.assertEqual(len(corpus), n_corpus)
                self.assertEqual(len(queries), n_queries)
                self.assertTrue({q["answer_path"] for q in queries} <= set(corpus))


if __name__ == "__main__":
    unittest.main()
