"""Canary regression test for the attachment-indexing root cause (issue #80).

The real-world miss: ``Team Target FY2025 $210,000,000`` lived inside an ``.xlsx``
attachment; the email body was four lines with no number, and attachment text was
never fed into the index — so ``search_email`` returned a confident empty set.

This test drives a tiny fixture ``.eml`` (carrying an ``.xlsx`` whose cells contain
``210,000,000``) through the *whole* ingest -> index path with a fake, in-memory
embedder + Qdrant (same mocking style as tests/test_contextual_index.py) and asserts:

1. the attachment's text (incl. ``210,000,000``) reaches the embedder, and
2. the upserted Qdrant point carries ``attachment_name`` + ``parent_message_id``
   (and ``thread_id``) so a hit traces back to its email.

The body of the fixture email contains NO number, so a pass proves the fact came
from the attachment — not the body.
"""

import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from email.message import EmailMessage
from unittest import mock


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_HAVE_OPENPYXL = _have("openpyxl")


# Inject a fake `transformers` so the lazy AutoTokenizer import in
# build_contextual_index resolves without the (env-absent) real package —
# mirrors tests/test_contextual_index.py.
def _install_fake_transformers():
    stub = types.ModuleType("transformers")
    fake_tok = mock.Mock()
    fake_tok.encode.side_effect = lambda text, add_special_tokens=True: list(range(10))
    fake_cls = mock.Mock()
    fake_cls.from_pretrained.return_value = fake_tok
    stub.AutoTokenizer = fake_cls
    sys.modules.setdefault("transformers", stub)


_install_fake_transformers()

from src.indexing.attachment_docs import build_attachment_documents  # noqa: E402
from src.indexing.contextual_index import build_contextual_index  # noqa: E402


def _xlsx_bytes(cells) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    for row in cells:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _write_mbo_eml(path: str) -> str:
    m = EmailMessage()
    m["From"] = "dana.reyes@northwind.example"
    m["To"] = "sam.okafor@northwind.example"
    m["Subject"] = "Q3 MBO targets partner team.xlsx"
    m["Message-ID"] = "<mb01@northwind.example>"
    # Body carries NO number — the whole informational payload is the attachment.
    m.set_content("Team\nHere are MBO targets for Q3\nLMK if anything is wrong")
    m.add_attachment(
        _xlsx_bytes(
            [
                ["Objective", "Weight", "Team Target FY2025"],
                ["Partner bookings", 0.50, "210,000,000"],
            ]
        ),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="Q3 MBO targets partner team.xlsx",
    )
    with open(path, "wb") as fh:
        fh.write(bytes(m))
    return path


@unittest.skipUnless(_HAVE_OPENPYXL, "openpyxl not installed")
class TestAttachmentIndexingCanary(unittest.TestCase):
    def setUp(self):
        import shutil

        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.eml = _write_mbo_eml(os.path.join(self.d, "mbo.eml"))

    def test_attachment_number_is_indexed_with_lineage(self):
        # Build the attachment documents from the fixture .eml, then run the real
        # index path with a fake embedder + Qdrant so we can inspect what got
        # embedded and what payload landed on the point.
        attach_docs = build_attachment_documents([self.eml], extractor_name="tesseract")
        self.assertEqual(len(attach_docs), 1, "the .xlsx attachment must produce one document")

        captured = {"texts": [], "points": []}

        fake_embedder = mock.Mock()
        fake_embedder.dim = 1024
        fake_embedder.produces_sparse = True

        def _encode(texts, batch_size=32, max_length=512):
            captured["texts"].extend(texts)
            return ([[0.0] * 1024 for _ in texts], [{"7": 0.9} for _ in texts])

        fake_embedder.encode.side_effect = _encode

        def _capture_upsert(client, collection, points):
            captured["points"].extend(points)

        with (
            mock.patch("src.indexing.contextual_index.hq.get_client"),
            mock.patch("src.indexing.contextual_index.hq.ensure_hybrid_collection"),
            mock.patch("src.indexing.contextual_index.hq.upsert", side_effect=_capture_upsert),
        ):
            res = build_contextual_index(
                [],  # no email bodies needed for the canary; attachments only
                collection="canary",
                embedder=fake_embedder,
                apply_noise_filter=False,
                extra_docs=attach_docs,
                qdrant_url="http://x",
            )

        # 1. The attachment number reached the embedder (it IS indexed).
        joined = "\n".join(captured["texts"])
        self.assertIn("210,000,000", joined, "attachment cell value must be embedded")
        # The numeric augmentation also appended the canonical form (issue #82).
        self.assertIn("210000000", joined)

        # 2. The upserted point traces back to its email.
        self.assertGreaterEqual(res.chunks, 1)
        self.assertTrue(captured["points"], "at least one point must be upserted")
        payloads = [p.payload for p in captured["points"]]
        attach_payloads = [p for p in payloads if p.get("content_kind") == "attachment"]
        self.assertTrue(attach_payloads, "an attachment-kind point must exist")
        p = attach_payloads[0]
        self.assertEqual(p["attachment_name"], "Q3 MBO targets partner team.xlsx")
        self.assertEqual(p["parent_message_id"], "<mb01@northwind.example>")
        self.assertTrue(p.get("thread_id"))
        # The stored text payload keeps the human-readable surface form.
        self.assertIn("210,000,000", p["text"])


if __name__ == "__main__":
    unittest.main()
