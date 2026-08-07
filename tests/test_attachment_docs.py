"""Tests for src/indexing/attachment_docs.py — attachment text -> Documents (#80)."""

import importlib.util
import io
import os
import tempfile
import unittest
from email.message import EmailMessage


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_HAVE_OPENPYXL = _have("openpyxl")


def _xlsx_bytes(cells) -> bytes:
    """A one-sheet .xlsx whose rows are the given lists of cell values."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    for row in cells:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _write_eml(path: str, message_id: str, body: str, attachments) -> str:
    m = EmailMessage()
    m["From"] = "eric.levander@windriver.com"
    m["To"] = "fred@windriver.com"
    m["Subject"] = "Q3 MBO targets partner team.xlsx"
    m["Message-ID"] = message_id
    m.set_content(body)
    for data, maintype, subtype, filename in attachments:
        m.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    with open(path, "wb") as fh:
        fh.write(bytes(m))
    return path


class TestBuildAttachmentDocuments(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        import shutil

        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def _paths(self):
        # extractor_name="tesseract" avoids building an LLM OCR client; the xlsx /
        # csv handlers win dispatch before OCR is consulted anyway.
        from src.indexing.attachment_docs import build_attachment_documents

        return build_attachment_documents

    @unittest.skipUnless(_HAVE_OPENPYXL, "openpyxl not installed")
    def test_xlsx_cells_become_a_document_with_lineage(self):
        build = self._paths()
        xlsx = _xlsx_bytes(
            [
                ["Objective", "Weight", "Team Target FY2025"],
                ["Partner bookings", 0.50, "210,000,000"],
            ]
        )
        path = _write_eml(
            os.path.join(self.d, "mbo.eml"),
            "<mbo@windriver.com>",
            "Team\nHere are MBO targets for Q3\nLMK if anything is wrong",
            [
                (
                    xlsx,
                    "application",
                    "vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Q3 MBO targets partner team.xlsx",
                )
            ],
        )
        docs = build([path], extractor_name="tesseract")
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        # The number in the sheet is in the indexed text (the whole point).
        self.assertIn("210,000,000", doc.text)
        # Lineage payload traces the hit back to its email.
        self.assertEqual(doc.metadata["content_kind"], "attachment")
        self.assertEqual(doc.metadata["attachment_name"], "Q3 MBO targets partner team.xlsx")
        self.assertEqual(doc.metadata["parent_message_id"], "<mbo@windriver.com>")
        self.assertTrue(doc.metadata.get("thread_id"))

    def test_csv_attachment_yields_document(self):
        build = self._paths()
        path = _write_eml(
            os.path.join(self.d, "ledger.eml"),
            "<csv@x>",
            "see attached",
            [(b"item,amount\nDeal,210,000,000\n", "text", "csv", "ledger.csv")],
        )
        docs = build([path], extractor_name="tesseract")
        self.assertEqual(len(docs), 1)
        self.assertIn("210,000,000", docs[0].text)
        self.assertEqual(docs[0].metadata["attachment_name"], "ledger.csv")

    def test_email_with_no_attachment_yields_nothing(self):
        build = self._paths()
        path = _write_eml(os.path.join(self.d, "plain.eml"), "<none@x>", "just a body", [])
        self.assertEqual(build([path], extractor_name="tesseract"), [])

    def test_unsupported_binary_attachment_is_skipped(self):
        build = self._paths()
        path = _write_eml(
            os.path.join(self.d, "bin.eml"),
            "<bin@x>",
            "body",
            [(b"\x00\x01BINARY\xff", "application", "octet-stream", "mystery.bin")],
        )
        # No handler extracts text -> no document (not a crash).
        self.assertEqual(build([path], extractor_name="tesseract"), [])

    def test_malformed_eml_path_is_skipped(self):
        build = self._paths()
        bad = os.path.join(self.d, "nope.eml")
        with open(bad, "wb") as fh:
            fh.write(b"not really an email")
        # Never raises — a bad path just yields no documents.
        self.assertIsInstance(build([bad], extractor_name="tesseract"), list)


if __name__ == "__main__":
    unittest.main()


class TestAttachmentMessageKey(unittest.TestCase):
    """An email's attachment chunks must carry the SAME message_key as its body
    chunks, so one delete filter clears the whole email before a re-index (#101)."""

    def setUp(self):
        import shutil

        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def _docs(self, path):
        from src.indexing.attachment_docs import build_attachment_documents

        return build_attachment_documents([path], extractor_name="tesseract")

    def _eml(self, message_id="<mbo@windriver.com>"):
        return _write_eml(
            os.path.join(self.d, "a.eml"),
            message_id,
            "See the attached targets.",
            [(b"name,target\nfred,100\n", "text", "csv", "targets.csv")],
        )

    def test_attachment_chunks_share_the_bodys_message_key(self):
        from src.data.loaders.mail_archive_x import MailArchiveXLoader

        path = self._eml()
        body_key = MailArchiveXLoader(eml_files=[path], verbose=False).load()[0].message_key()
        docs = self._docs(path)
        self.assertTrue(docs)
        for d in docs:
            self.assertEqual(d.metadata["message_key"], body_key)

    def test_doc_ids_are_stable_across_runs(self):
        """Attachment doc ids seed the deterministic point ids, so they must not
        depend on the run."""
        path = self._eml()
        self.assertEqual([d.doc_id for d in self._docs(path)], [d.doc_id for d in self._docs(path)])

    def test_message_key_is_excluded_from_the_embedded_text(self):
        docs = self._docs(self._eml())
        self.assertIn("message_key", docs[0].excluded_embed_metadata_keys)

    def test_an_email_without_a_message_id_still_gets_a_key(self):
        """Header-less mail falls back to the content hash rather than an empty key
        that would make delete_by_message_keys a no-op."""
        path = _write_eml(
            os.path.join(self.d, "b.eml"),
            "",
            "See the attached targets.",
            [(b"name,target\nfred,100\n", "text", "csv", "targets.csv")],
        )
        docs = self._docs(path)
        self.assertTrue(docs)
        self.assertTrue(docs[0].metadata["message_key"])
