"""Tests for src/indexing/contextual_index.py — hermetic (mocked embedder + qdrant).

NOTE: `transformers` (AutoTokenizer) is NOT installed in the mailrag-test conda
env.  We inject a fake ``transformers`` module into ``sys.modules`` at test
setup time so the lazy ``from transformers import AutoTokenizer`` inside
``build_contextual_index`` resolves to our mock without importing the real
package.  This is an explicit, documented workaround for the env gap — NOT
silent removal of a real dependency.

CONCERN: ``transformers`` is absent from the mailrag-test conda env.  The
production path uses ``AutoTokenizer.from_pretrained("BAAI/bge-m3")`` which
downloads / reads from the HF cache.  This env gap should be fixed by adding
``transformers`` to the mailrag-test requirements; the workaround here keeps
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
