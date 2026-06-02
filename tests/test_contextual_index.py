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
from src.indexing.contextual_index import build_contextual_index  # noqa: E402
from src.data.models import NormalizedEmail  # noqa: E402


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
        with mock.patch("src.indexing.contextual_index.hq.get_client", return_value=fake_qdrant), \
             mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection") as ensure, \
             mock.patch("src.indexing.contextual_index.hq.upsert") as upsert:
            res = build_contextual_index(
                emails, collection="t", embedder=fake_embedder,
                summaries={emails[0].message_id: "SUM about the plan"},
                embed_summary=True, recreate=True, qdrant_url="http://x")
        ensure.assert_called_once()
        self.assertTrue(upsert.called)
        self.assertGreaterEqual(res.chunks, 1)
        self.assertEqual(res.collection, "t")
        embedded_text = fake_embedder.encode.call_args[0][0][0]
        self.assertIn("SUM about the plan", embedded_text)

    def test_noise_filtered_emails_dropped(self):
        # Noise filter is domain-based; must use a sender from a known noise domain.
        noise = NormalizedEmail(
            sender="notifications@linkedin.com",
            subject="You have 5 new notifications",
            date=None,
            body="Unsubscribe | LinkedIn notifications digest",
            source="enron",
            source_id="<n@x>",
            message_id="<n@x>",
        )
        fake_embedder = mock.Mock()
        fake_embedder.encode.return_value = ([], [])
        with mock.patch("src.indexing.contextual_index.hq.get_client"), \
             mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"), \
             mock.patch("src.indexing.contextual_index.hq.upsert"):
            res = build_contextual_index([noise], collection="t", embedder=fake_embedder)
        self.assertEqual(res.kept_emails, 0)
