"""Tests for src/indexing/contextual_index.py — hermetic (mocked embedder + qdrant).

NOTE: `transformers` (AutoTokenizer) is NOT installed in the mailrag conda
env.  We inject a fake ``transformers`` module into ``sys.modules`` at test
setup time so the lazy ``from transformers import AutoTokenizer`` inside
``build_contextual_index`` resolves to our mock without importing the real
package.  This is an explicit, documented workaround for the env gap — NOT
silent removal of a real dependency.

CONCERN: ``transformers`` is absent from the mailrag conda env.  The
production path uses ``AutoTokenizer.from_pretrained("BAAI/bge-m3")`` which
downloads / reads from the HF cache.  This env gap should be fixed by adding
``transformers`` to the mailrag requirements; the workaround here keeps
tests running in the meantime.
"""

import sys
import types
import unittest
from unittest import mock


# ---------------------------------------------------------------------------
# Inject a fake `transformers` stub into sys.modules so the lazy import
# inside build_contextual_index resolves without the real package.
# ---------------------------------------------------------------------------
def _install_fake_transformers():
    """Create a minimal transformers stub and insert it into sys.modules."""
    stub = types.ModuleType("transformers")
    fake_tok = mock.Mock()
    # encode() returns a short list — SentenceSplitter just needs token counts.
    fake_tok.encode.side_effect = lambda text, add_special_tokens=True: list(range(10))
    fake_cls = mock.Mock()
    fake_cls.from_pretrained.return_value = fake_tok
    stub.AutoTokenizer = fake_cls
    sys.modules.setdefault("transformers", stub)
    return stub, fake_tok, fake_cls


_TRANSFORMERS_STUB, _FAKE_TOKENIZER, _FAKE_TOKENIZER_CLS = _install_fake_transformers()

# Now it's safe to import the module under test.
from src.data.models import NormalizedEmail  # noqa: E402
from src.indexing.contextual_index import build_contextual_index  # noqa: E402
from src.indexing.point_ids import content_hash  # noqa: E402


def _email(body, subject="Re: plan", mid="<a@x>"):
    # NormalizedEmail positional fields: sender, subject, date, body, source, source_id
    # Optional: recipients (str not list), cc (str or None), message_id, ...
    return NormalizedEmail(
        sender="s@x.com",
        subject=subject,
        date=None,
        body=body,
        source="enron",
        source_id=mid,
        recipients="r@x.com",
        cc=None,
        message_id=mid,
    )


class TestBuildContextualIndex(unittest.TestCase):
    def test_embeds_summary_and_upserts(self):
        emails = [_email("hello world this is a real business email about the plan")]
        fake_embedder = mock.Mock()
        fake_embedder.encode.return_value = ([[0.1] * 1024], [{"7": 0.9}])
        fake_qdrant = mock.MagicMock()
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client", return_value=fake_qdrant),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection") as ensure,
            mock.patch("src.indexing.contextual_index.hq.upsert") as upsert,
        ):
            res = build_contextual_index(
                emails,
                collection="t",
                embedder=fake_embedder,
                summaries={emails[0].message_id: "SUM about the plan"},
                embed_summary=True,
                recreate=True,
                qdrant_url="http://x",
            )
        ensure.assert_called_once()
        self.assertTrue(upsert.called)
        self.assertGreaterEqual(res.chunks, 1)
        self.assertEqual(res.collection, "t")
        embedded_text = fake_embedder.encode.call_args[0][0][0]
        self.assertIn("SUM about the plan", embedded_text)

    def _noise_filter_flagging(self, domain):
        """A hermetic NoiseFilter that flags one sender domain (independent of
        the gitignored project rules), patched in for deterministic tests."""
        from src.data.noise_filter import NoiseFilter, _CategoryRule

        return NoiseFilter([_CategoryRule(name="x", description="", sender_domains=[domain])])

    def _build_one(self, email, **kwargs):
        fake_embedder = mock.Mock()
        fake_embedder.encode.return_value = ([], [])
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.upsert"),
            mock.patch(
                "src.indexing.contextual_index.NoiseFilter.from_project_rules",
                return_value=self._noise_filter_flagging("spammy.example"),
            ),
        ):
            return build_contextual_index([email], collection="t", embedder=fake_embedder, **kwargs)

    def test_noise_filtered_emails_dropped(self):
        noise = _email("Unsubscribe digest", mid="<n@x>")
        noise.sender = "notifications@spammy.example"
        res = self._build_one(noise)
        self.assertEqual(res.kept_emails, 0)

    def test_apply_noise_filter_false_keeps_noise(self):
        # Callers that pre-filter + tag (e.g. the local build) opt out of the
        # redundant internal filter so tagged bulk is not silently re-dropped.
        noise = _email("Unsubscribe digest", mid="<n@x>")
        noise.sender = "notifications@spammy.example"
        res = self._build_one(noise, apply_noise_filter=False)
        self.assertEqual(res.kept_emails, 1)

    def test_collection_sized_from_embedder_dim(self):
        """Collection vector size must follow embedder.dim, not a hardcoded 1024 —
        so a non-1024 embedder (e.g. a NIM at 2048) gets a correctly-sized collection."""
        email = _email("a real business email about the quarterly plan and budget")
        fake_embedder = mock.Mock()
        fake_embedder.dim = 2048
        fake_embedder.encode.return_value = ([[0.0] * 2048], [{"7": 0.9}])
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection") as ensure,
            mock.patch("src.indexing.contextual_index.hq.upsert"),
        ):
            build_contextual_index(
                [email],
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
            )
        self.assertEqual(ensure.call_args.kwargs.get("dim"), 2048)

    def test_identical_bodies_are_deduplicated(self):
        """Two emails with identical body collapse to a single embedded chunk."""
        body = "the quarterly plan budget meeting notes are attached here"
        e1, e2 = _email(body, mid="<a@x>"), _email(body, mid="<b@x>")
        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.encode.return_value = ([[0.0] * 1024], [{"7": 0.9}])
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.upsert"),
        ):
            res = build_contextual_index(
                [e1, e2],
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
            )
        self.assertEqual(res.chunks, 1)
        self.assertEqual(len(fake_embedder.encode.call_args[0][0]), 1)

    def test_dense_only_embedder_uses_dense_collection(self):
        """A produces_sparse=False embedder builds a dense-only collection and
        dense-only points (no sparse leg)."""
        email = _email("a real business email about the quarterly plan and budget")
        fake = mock.Mock()
        fake.produces_sparse = False
        fake.dim = 1024
        fake.encode.return_value = ([[0.0] * 1024], [{}])
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_dense_collection") as ed,
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection") as eh,
            mock.patch("src.indexing.contextual_index.hq.make_dense_point") as mdp,
            mock.patch("src.indexing.contextual_index.hq.make_point") as mp,
            mock.patch("src.indexing.contextual_index.hq.upsert"),
        ):
            build_contextual_index(
                [email],
                collection="t",
                embedder=fake,
                apply_noise_filter=False,
                qdrant_url="http://x",
            )
        ed.assert_called_once()
        eh.assert_not_called()
        self.assertTrue(mdp.called)
        mp.assert_not_called()

    def test_no_emails_returns_zero_and_skips_collection(self):
        """Empty input returns a zero BuildResult and never touches Qdrant or the embedder."""
        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client") as gc,
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection") as ensure,
            mock.patch("src.indexing.contextual_index.hq.upsert") as upsert,
        ):
            res = build_contextual_index(
                [],
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
            )
        self.assertEqual((res.kept_emails, res.chunks), (0, 0))
        gc.assert_not_called()
        ensure.assert_not_called()
        upsert.assert_not_called()
        fake_embedder.encode.assert_not_called()


class TestIncrementalAppend(unittest.TestCase):
    """Append mode (recreate=False) — the indexing half of continuous sync (#101).

    Re-indexing the same mail must not duplicate chunks, and re-indexing changed
    mail must not leave the old chunks behind.
    """

    def _build(self, emails, **kwargs):
        """Run a build with Qdrant fully mocked; return (upserted_points, mocks)."""
        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.encode.side_effect = lambda texts, **kw: (
            [[0.1] * 1024 for _ in texts],
            [{"7": 0.9} for _ in texts],
        )
        upserted = []
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection") as ensure,
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes") as ensure_idx,
            mock.patch("src.indexing.contextual_index.hq.delete_by_message_keys") as delete,
            mock.patch(
                "src.indexing.contextual_index.hq.upsert",
                side_effect=lambda c, n, pts: upserted.extend(pts),
            ),
        ):
            res = build_contextual_index(
                emails,
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
                **kwargs,
            )
        return upserted, res, ensure, ensure_idx, delete

    def test_same_email_yields_the_same_point_ids_across_runs(self):
        body = "the quarterly plan and the budget for the coming year"
        first, _, _, _, _ = self._build([_email(body)], recreate=False)
        second, _, _, _, _ = self._build([_email(body)], recreate=False)
        self.assertTrue(first)
        self.assertEqual([p.id for p in first], [p.id for p in second])

    def test_point_ids_are_unaffected_by_other_emails_in_the_run(self):
        """The property that makes a 40-email delta safe to index into a
        20,000-email collection."""
        target = _email("the quarterly plan and the budget", mid="<target@x>")
        alone, _, _, _, _ = self._build([target], recreate=False)
        crowded, _, _, _, _ = self._build(
            [
                _email("something else entirely about logistics", mid="<other@x>"),
                _email("the quarterly plan and the budget", mid="<target@x>"),
            ],
            recreate=False,
        )
        crowded_target = [p for p in crowded if p.payload.get("message_key") == "target@x"]
        self.assertEqual([p.id for p in alone], [p.id for p in crowded_target])

    def test_append_deletes_the_emails_existing_points_before_upserting(self):
        _, _, _, _, delete = self._build([_email("a body about the plan")], recreate=False)
        delete.assert_called_once()
        self.assertEqual(list(delete.call_args[0][2]), ["a@x"])

    def test_append_backfills_payload_indexes_on_a_preexisting_collection(self):
        """A collection built before message_key existed has no index for it, so
        the delete filter would fail without this."""
        _, _, _, ensure_idx, _ = self._build([_email("a body about the plan")], recreate=False)
        ensure_idx.assert_called_once()

    def test_recreate_mode_does_not_delete(self):
        """A full rebuild drops the collection outright — a per-email delete would
        be wasted work against an empty collection."""
        _, _, _, ensure_idx, delete = self._build([_email("a body")], recreate=True)
        delete.assert_not_called()
        ensure_idx.assert_not_called()

    def test_every_point_carries_message_key_and_content_hash(self):
        points, _, _, _, _ = self._build([_email("a body about the plan")], recreate=False)
        self.assertTrue(points)
        for p in points:
            self.assertEqual(p.payload["message_key"], "a@x")
            self.assertEqual(p.payload["content_hash"], content_hash(p.payload["text"]))

    def test_point_ids_are_unique_within_a_run(self):
        emails = [
            _email(f"a distinct body number {i} about the plan", mid=f"<m{i}@x>") for i in range(5)
        ]
        points, _, _, _, _ = self._build(emails, recreate=False)
        self.assertEqual(len(points), len({p.id for p in points}))

    def _build_against_legacy(self, **kwargs):
        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.encode.side_effect = lambda texts, **kw: (
            [[0.1] * 1024 for _ in texts],
            [{"7": 0.9} for _ in texts],
        )
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes"),
            mock.patch("src.indexing.contextual_index.hq.has_legacy_points", return_value=True),
            mock.patch("src.indexing.contextual_index.hq.delete_by_message_keys") as delete,
            mock.patch("src.indexing.contextual_index.hq.upsert") as upsert,
        ):
            res = build_contextual_index(
                [_email("a body about the plan")],
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
                **kwargs,
            )
            return res, delete, upsert

    def test_append_into_a_legacy_collection_is_refused(self):
        """Duplicating a 20k-email collection is the worst failure here, so the
        default is to stop and ask for one rebuild."""
        with self.assertRaises(RuntimeError) as ctx:
            self._build_against_legacy(recreate=False)
        self.assertIn("--recreate", str(ctx.exception))

    def test_the_refusal_happens_before_anything_is_written(self):
        with self.assertRaises(RuntimeError):
            self._build_against_legacy(recreate=False)
        # nothing upserted, nothing deleted — the collection is untouched
        with mock.patch("src.indexing.contextual_index.hq.upsert") as upsert:
            self.assertFalse(upsert.called)

    def test_the_guard_can_be_overridden(self):
        res, _, upsert = self._build_against_legacy(recreate=False, allow_legacy_append=True)
        self.assertGreaterEqual(res.chunks, 1)
        self.assertTrue(upsert.called)

    def test_the_guard_does_not_apply_to_a_full_rebuild(self):
        res, _, upsert = self._build_against_legacy(recreate=True)
        self.assertGreaterEqual(res.chunks, 1)
        self.assertTrue(upsert.called)

    def _build_with_policy(self, existing_policy, **kwargs):
        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.encode.side_effect = lambda texts, **kw: (
            [[0.1] * 1024 for _ in texts],
            [{"7": 0.9} for _ in texts],
        )
        upserted = []
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes"),
            mock.patch("src.indexing.contextual_index.hq.has_legacy_points", return_value=False),
            mock.patch(
                "src.indexing.contextual_index.hq.collection_policy",
                return_value=existing_policy,
            ),
            mock.patch("src.indexing.contextual_index.hq.delete_by_message_keys"),
            mock.patch(
                "src.indexing.contextual_index.hq.upsert",
                side_effect=lambda c, n, pts: upserted.extend(pts),
            ),
        ):
            res = build_contextual_index(
                [_email("a body about the quarterly plan")],
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
                **kwargs,
            )
        return res, upserted

    def test_every_point_is_stamped_with_the_index_policy(self):
        _res, points = self._build_with_policy("", recreate=True)
        self.assertTrue(points)
        stamps = {p.payload["policy_fingerprint"] for p in points}
        self.assertEqual(len(stamps), 1)
        self.assertTrue(next(iter(stamps)))

    def test_appending_under_a_different_policy_is_refused(self):
        """Otherwise two incomparable vector populations end up in one collection
        with nothing to signal it (#101)."""
        with self.assertRaises(RuntimeError) as ctx:
            self._build_with_policy("a-different-policy", recreate=False)
        self.assertIn("--recreate", str(ctx.exception))

    def test_appending_under_the_same_policy_is_allowed(self):
        _res, points = self._build_with_policy("", recreate=False)
        matching = points[0].payload["policy_fingerprint"]
        _res2, points2 = self._build_with_policy(matching, recreate=False)
        self.assertTrue(points2)

    def test_a_collection_with_no_recorded_policy_does_not_block_an_append(self):
        """An empty or pre-fingerprint collection reads as 'unknown'; the legacy
        guard, not this one, is what handles the pre-deterministic-id case."""
        _res, points = self._build_with_policy("", recreate=False)
        self.assertTrue(points)

    def test_a_full_rebuild_ignores_the_existing_policy(self):
        _res, points = self._build_with_policy("a-different-policy", recreate=True)
        self.assertTrue(points)


class TestDedupDeleteInteraction(unittest.TestCase):
    """The corpus-wide dedup runs AFTER ids are assigned, so an email can finish
    with zero surviving chunks. The delete set must come from what survives, or
    that email's existing points are erased and nothing replaces them (#101
    review finding)."""

    def _build(self, emails, **kwargs):
        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.encode.side_effect = lambda texts, **kw: (
            [[0.1] * 1024 for _ in texts],
            [{"7": 0.9} for _ in texts],
        )
        upserted = []
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes"),
            mock.patch("src.indexing.contextual_index.hq.has_legacy_points", return_value=False),
            mock.patch("src.indexing.contextual_index.hq.collection_policy", return_value=""),
            mock.patch("src.indexing.contextual_index.hq.delete_by_message_keys") as delete,
            mock.patch(
                "src.indexing.contextual_index.hq.upsert",
                side_effect=lambda c, n, pts: upserted.extend(pts),
            ),
        ):
            res = build_contextual_index(
                emails,
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
                **kwargs,
            )
        return res, upserted, delete

    def test_an_email_absorbed_by_dedup_does_not_have_its_points_deleted(self):
        """Two emails with byte-identical bodies: the second contributes no
        surviving chunk, so deleting its existing points would erase it."""
        body = "Backup completed successfully."
        res, points, delete = self._build(
            [_email(body, mid="<a@x>"), _email(body, mid="<b@x>")], recreate=False
        )
        deleted = set(delete.call_args[0][2])
        written = {p.payload["message_key"] for p in points}
        self.assertEqual(deleted, written)
        self.assertNotIn("b@x", deleted)

    def test_such_an_email_is_reported_as_not_indexed(self):
        """So the sync ledger leaves it pending instead of recording success."""
        body = "Backup completed successfully."
        res, _points, _delete = self._build(
            [_email(body, mid="<a@x>"), _email(body, mid="<b@x>")], recreate=False
        )
        self.assertIn("a@x", res.indexed_message_keys)
        self.assertNotIn("b@x", res.indexed_message_keys)

    def test_normal_emails_are_all_reported_as_indexed(self):
        res, _points, _delete = self._build(
            [
                _email("first distinct body about the plan", mid="<a@x>"),
                _email("second quite different body about budgets", mid="<b@x>"),
            ],
            recreate=False,
        )
        self.assertEqual(set(res.indexed_message_keys), {"a@x", "b@x"})

    def test_the_delete_set_never_exceeds_what_is_written(self):
        res, points, delete = self._build(
            [_email("a body about the quarterly plan", mid="<a@x>")], recreate=False
        )
        self.assertTrue(set(delete.call_args[0][2]) <= {p.payload["message_key"] for p in points})


class TestBatchIsolation(unittest.TestCase):
    """One undindexable document must not take a whole batch with it, and a
    failure mid-upsert must not leave the delta deleted-with-no-replacement
    (third-round audit findings)."""

    def _embedder(self):
        e = mock.Mock()
        e.dim = 1024
        e.encode.side_effect = lambda texts, **kw: (
            [[0.1] * 1024 for _ in texts],
            [{"7": 0.9} for _ in texts],
        )
        return e

    def test_a_document_the_splitter_rejects_does_not_kill_the_batch(self):
        good = [
            _email(f"a distinct body number {i} about the plan", mid=f"<g{i}@x>") for i in range(3)
        ]
        upserted = []
        real_split = None

        def selective(self_splitter, docs, show_progress=False):
            if len(docs) > 1:
                raise ValueError("Metadata length (543) is longer than chunk size (512)")
            if docs[0].metadata.get("message_key") == "g1@x":
                raise ValueError("Metadata length (543) is longer than chunk size (512)")
            return real_split(self_splitter, docs, show_progress=show_progress)

        from llama_index.core.node_parser import SentenceSplitter

        real_split = SentenceSplitter.get_nodes_from_documents
        with (
            mock.patch.object(SentenceSplitter, "get_nodes_from_documents", selective),
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes"),
            mock.patch("src.indexing.contextual_index.hq.has_legacy_points", return_value=False),
            mock.patch("src.indexing.contextual_index.hq.collection_policy", return_value=""),
            mock.patch("src.indexing.contextual_index.hq.delete_by_message_keys"),
            mock.patch(
                "src.indexing.contextual_index.hq.upsert",
                side_effect=lambda c, n, pts: upserted.extend(pts),
            ),
        ):
            res = build_contextual_index(
                good,
                collection="t",
                embedder=self._embedder(),
                apply_noise_filter=False,
                qdrant_url="http://x",
                recreate=False,
            )
        written = {p.payload["message_key"] for p in upserted}
        self.assertEqual(written, {"g0@x", "g2@x"})  # the good mail survived
        self.assertIn("g1@x", res.failed_message_keys)  # and the culprit is named

    def test_a_failure_mid_upsert_only_exposes_one_batch(self):
        """Deleting the whole delta up front meant a mid-loop failure removed
        every not-yet-rewritten email — reproduced as 4 of 6 vanishing."""
        emails = [
            _email(f"body number {i} about the quarterly plan", mid=f"<m{i}@x>") for i in range(6)
        ]
        deleted, upserted = [], []

        def failing_upsert(c, n, pts):
            if len(upserted) >= 2:
                raise ConnectionError("qdrant connection reset mid-batch")
            upserted.extend(pts)

        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes"),
            mock.patch("src.indexing.contextual_index.hq.has_legacy_points", return_value=False),
            mock.patch("src.indexing.contextual_index.hq.collection_policy", return_value=""),
            mock.patch(
                "src.indexing.contextual_index.hq.delete_by_message_keys",
                side_effect=lambda c, n, keys: deleted.extend(keys),
            ),
            mock.patch("src.indexing.contextual_index.hq.upsert", side_effect=failing_upsert),
        ):
            with self.assertRaises(ConnectionError):
                build_contextual_index(
                    emails,
                    collection="t",
                    embedder=self._embedder(),
                    apply_noise_filter=False,
                    qdrant_url="http://x",
                    recreate=False,
                    upsert_batch=2,
                )
        # Only the batches actually attempted may have been deleted — never the
        # whole delta.
        self.assertLessEqual(len(set(deleted)), 4)
        self.assertLess(len(set(deleted)), 6)

    def test_a_full_rebuild_never_deletes(self):
        deleted = []
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch(
                "src.indexing.contextual_index.hq.delete_by_message_keys",
                side_effect=lambda c, n, keys: deleted.extend(keys),
            ),
            mock.patch("src.indexing.contextual_index.hq.upsert"),
        ):
            build_contextual_index(
                [_email("a body about the plan", mid="<a@x>")],
                collection="t",
                embedder=self._embedder(),
                apply_noise_filter=False,
                qdrant_url="http://x",
                recreate=True,
            )
        self.assertEqual(deleted, [])

    def test_every_batch_deletes_exactly_its_own_emails_before_writing_them(self):
        """The per-batch delete must be correctly SCOPED, not merely bounded.

        A mutation audit showed the previous test — which only asserted an upper
        bound on how much was deleted — stayed green when the delete was hoisted
        so that only the first batch's keys were removed. That leaves every later
        batch's old points in place: stale chunks surviving a replacement, which
        is the failure delete-then-upsert exists to prevent.

        So this pins the ordering directly: for each email written, its delete
        must have happened, and must have happened BEFORE its first upsert.
        """
        emails = [
            _email(f"body number {i} about the quarterly plan", mid=f"<m{i}@x>") for i in range(6)
        ]
        events = []  # ("del", key) / ("up", key) in call order

        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.encode.side_effect = lambda texts, **kw: (
            [[0.1] * 1024 for _ in texts],
            [{"7": 0.9} for _ in texts],
        )
        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.ensure_payload_indexes"),
            mock.patch("src.indexing.contextual_index.hq.has_legacy_points", return_value=False),
            mock.patch("src.indexing.contextual_index.hq.collection_policy", return_value=""),
            mock.patch(
                "src.indexing.contextual_index.hq.delete_by_message_keys",
                side_effect=lambda c, n, keys: events.extend(("del", k) for k in keys),
            ),
            mock.patch(
                "src.indexing.contextual_index.hq.upsert",
                side_effect=lambda c, n, pts: events.extend(
                    ("up", p.payload["message_key"]) for p in pts
                ),
            ),
        ):
            build_contextual_index(
                emails,
                collection="t",
                embedder=fake_embedder,
                apply_noise_filter=False,
                qdrant_url="http://x",
                recreate=False,
                upsert_batch=2,
            )

        written = [k for kind, k in events if kind == "up"]
        deleted = [k for kind, k in events if kind == "del"]
        self.assertEqual(len(written), 6)
        # Every written email had its old points removed...
        self.assertEqual(set(deleted), set(written))
        # ...exactly once...
        self.assertEqual(len(deleted), len(set(deleted)))
        # ...and before it was written.
        for key in set(written):
            self.assertLess(
                events.index(("del", key)),
                events.index(("up", key)),
                f"{key} was upserted before its delete",
            )
