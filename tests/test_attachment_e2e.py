"""End-to-end integration tests for the attachment pipeline.

Every other ``test_attachment_*`` module exercises a single unit with mocks/fakes
(fake OCR providers, in-memory blobs, stubbed libraries). This module proves the
*whole* path wired together with real objects and real files:

    .eml on disk -> ingest_eml -> AttachmentStore (sqlite + content-addressed blobs)
                 -> store.fetch(sha) -> Extractor dispatch -> handler/OCR -> text

Two tiers:

* :class:`TestAttachmentPipelineE2E` needs no optional libraries or system
  binaries, so it always runs (CI included). It proves preserve + dispatch +
  cache for the plaintext leg.
* :class:`TestAttachmentOcrE2E` needs the real OCR stack (``tesseract`` binary +
  ``pytesseract`` + ``Pillow``). It ``skipUnless`` those are present, so it
  proves the image-OCR leg on a dev box while degrading to a skip — never a
  failure — where the engine is absent. That skip *is* the graceful-degradation
  contract at the test layer.
"""

import importlib.util
import io
import os
import shutil
import tempfile
import unittest
from email.message import EmailMessage


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_TESSERACT = shutil.which("tesseract") is not None and _have("pytesseract") and _have("PIL")
_HAVE_DOCX = _have("docx")


def _png_with_text(text: str) -> bytes:
    """A white PNG with ``text`` drawn on it (black), for a real OCR round-trip."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (720, 200), "white")
    # Draw twice at 2x-ish spacing so the default bitmap font gives tesseract a
    # clear signal even without a TrueType face installed.
    d = ImageDraw.Draw(img)
    d.text((20, 60), text, fill="black")
    d.text((20, 110), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAttachmentPipelineE2E(unittest.TestCase):
    """The real ingest -> store -> extract chain, no optional libs required."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        # Import here so the module imports even if src layout changes underneath.
        from src.attachments.store import AttachmentStore

        self.store = AttachmentStore(os.path.join(self.d, "store"))
        self.addCleanup(self.store.close)

    def _write_eml(self, name: str, message_id: str, attachments) -> str:
        m = EmailMessage()
        m["From"] = "alice@work.com"
        m["To"] = "bob@work.com"
        m["Subject"] = "E2E"
        m["Message-ID"] = message_id
        m.set_content("body")
        for data, maintype, subtype, filename in attachments:
            m.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
        path = os.path.join(self.d, name)
        with open(path, "wb") as fh:
            fh.write(bytes(m))
        return path

    def test_plaintext_attachment_ingested_and_extracted(self):
        """A text/csv attachment survives ingest and comes back as real text via
        ``store.fetch`` — the full preserve -> dispatch -> extract loop with no
        mocks and no optional libraries."""
        from src.attachments.ingest_eml import ingest_eml

        path = self._write_eml(
            "plain.eml",
            "<e2e-plain@work>",
            [(b"name,amount\nInvoice,99\n", "text", "csv", "ledger.csv")],
        )
        counts = ingest_eml([path], self.store)
        self.assertEqual(counts, {"emails": 1, "attachments": 1, "skipped": 0})

        metas = self.store.list_for(message_id="<e2e-plain@work>")
        self.assertEqual(len(metas), 1)
        meta = metas[0]
        self.assertEqual(meta.filename, "ledger.csv")
        # bytes preserved verbatim in the content-addressed blob store
        self.assertEqual(self.store.get_bytes(meta.sha256), b"name,amount\nInvoice,99\n")

        # extractor="tesseract" avoids constructing an LLM client; the plaintext
        # handler wins dispatch before OCR is ever consulted for a text/* part.
        res = self.store.fetch(meta.sha256, extractor="tesseract")
        self.assertEqual(res["text_status"], "extracted")
        self.assertIn("Invoice", res["text"])
        self.assertEqual(res["mime"], "text/csv")

    def test_encoded_word_filename_survives_the_pipeline(self):
        """An RFC2047 encoded-word filename is decoded during ingest and the decoded
        name is what a downstream caller (e.g. the MCP server in #32) reads back."""
        from src.attachments.ingest_eml import ingest_eml

        # =?utf-8?B?SW52b2ljZS5jc3Y=?=  ==  "Invoice.csv"
        path = self._write_eml(
            "enc.eml", "<e2e-enc@work>", [(b"x\n", "text", "csv", "=?utf-8?B?SW52b2ljZS5jc3Y=?=")]
        )
        ingest_eml([path], self.store)
        metas = self.store.list_for(message_id="<e2e-enc@work>")
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0].filename, "Invoice.csv")

    def test_unsupported_binary_type_degrades_not_crashes(self):
        """An attachment whose type has no handler comes back as ``unsupported`` with
        empty text — never an exception — so the 'just open the raw file' path
        (#34) still has intact bytes to hand back."""
        from src.attachments.ingest_eml import ingest_eml

        blob = b"\x00\x01\x02BINARY\xff"
        path = self._write_eml(
            "bin.eml", "<e2e-bin@work>", [(blob, "application", "octet-stream", "mystery.bin")]
        )
        ingest_eml([path], self.store)
        meta = self.store.list_for(message_id="<e2e-bin@work>")[0]
        res = self.store.fetch(meta.sha256, extractor="tesseract")
        self.assertEqual(res["text_status"], "unsupported")
        self.assertEqual(res["text"], "")
        # raw bytes are still retrievable regardless of extraction outcome
        self.assertEqual(self.store.get_bytes(meta.sha256), blob)

    @unittest.skipUnless(_HAVE_DOCX, "python-docx not installed")
    def test_docx_attachment_extracted_through_store(self):
        """A real .docx round-trips through ingest and yields its paragraph text —
        the office-document leg end-to-end, not just at the handler unit level."""
        from docx import Document

        from src.attachments.ingest_eml import ingest_eml

        doc = Document()
        doc.add_paragraph("Quarterly revenue rose twenty percent.")
        dbuf = io.BytesIO()
        doc.save(dbuf)
        subtype = "vnd.openxmlformats-officedocument.wordprocessingml.document"
        path = self._write_eml(
            "doc.eml", "<e2e-docx@work>", [(dbuf.getvalue(), "application", subtype, "report.docx")]
        )
        ingest_eml([path], self.store)
        meta = self.store.list_for(message_id="<e2e-docx@work>")[0]
        res = self.store.fetch(meta.sha256, extractor="tesseract")
        self.assertEqual(res["text_status"], "extracted")
        self.assertIn("Quarterly revenue rose twenty percent.", res["text"])


@unittest.skipUnless(_TESSERACT, "tesseract OCR stack (binary+pytesseract+PIL) not available")
class TestAttachmentOcrE2E(unittest.TestCase):
    """The real OCR leg: a rendered image travels through ingest and tesseract
    reads its text back. Skipped (not failed) where the engine is absent."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        from src.attachments.store import AttachmentStore

        self.store = AttachmentStore(os.path.join(self.d, "store"))
        self.addCleanup(self.store.close)

    def _write_image_eml(self, png: bytes, filename: str) -> str:
        m = EmailMessage()
        m["From"] = "a@x.com"
        m["To"] = "b@y.com"
        m["Subject"] = "scan"
        m["Message-ID"] = "<e2e-ocr@x.com>"
        m.set_content("body")
        m.add_attachment(png, maintype="image", subtype="png", filename=filename)
        path = os.path.join(self.d, "img.eml")
        with open(path, "wb") as fh:
            fh.write(bytes(m))
        return path

    def test_image_ocr_end_to_end(self):
        from src.attachments.ingest_eml import ingest_eml

        png = _png_with_text("INVOICE NUMBER")
        # RFC2047-encoded name proves decode + OCR in one pass. "Invoice.png":
        path = self._write_image_eml(png, "=?utf-8?B?SW52b2ljZS5wbmc=?=")
        ingest_eml([path], self.store)

        meta = self.store.list_for(message_id="<e2e-ocr@x.com>")[0]
        self.assertEqual(meta.filename, "Invoice.png")
        self.assertEqual(meta.mime, "image/png")

        res = self.store.fetch(meta.sha256, extractor="tesseract")
        self.assertEqual(res["text_status"], "extracted", f"OCR produced no text: {res!r}")
        # tesseract is imperfect on the bitmap font, so assert on a robust token.
        self.assertIn("invoice", res["text"].lower())

    def test_second_fetch_is_served_from_cache(self):
        """The text cache means a re-fetch returns the same extracted text without
        re-running OCR — proving the store caches the (sha, extractor) result."""
        from src.attachments.ingest_eml import ingest_eml

        path = self._write_image_eml(_png_with_text("INVOICE NUMBER"), "scan.png")
        ingest_eml([path], self.store)
        meta = self.store.list_for(message_id="<e2e-ocr@x.com>")[0]
        first = self.store.fetch(meta.sha256, extractor="tesseract")
        second = self.store.fetch(meta.sha256, extractor="tesseract")
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(second["text_status"], "extracted")


if __name__ == "__main__":
    unittest.main()
