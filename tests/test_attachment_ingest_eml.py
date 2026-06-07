import os, tempfile, unittest
from email.message import EmailMessage

from src.attachments.store import AttachmentStore
from src.attachments.ingest_eml import ingest_eml


def _write_eml(path):
    m = EmailMessage()
    m["From"] = "alice@work.com"
    m["To"] = "bob@work.com"
    m["Subject"] = "Report"
    m["Message-ID"] = "<m1@work>"
    m.set_content("See attached.")
    m.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf",
                     filename="report.pdf")
    m.add_attachment(b"col\n1\n", maintype="text", subtype="csv", filename="d.csv")
    with open(path, "wb") as fh:
        fh.write(bytes(m))


class TestIngestEml(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.eml = os.path.join(self.d, "msg.eml")
        _write_eml(self.eml)
        self.store = AttachmentStore(os.path.join(self.d, "store"))

    def tearDown(self):
        self.store.close()

    def test_ingests_attachments_with_metadata(self):
        counts = ingest_eml([self.eml], self.store)
        self.assertEqual(counts["emails"], 1)
        self.assertEqual(counts["attachments"], 2)
        metas = self.store.list_for(message_id="<m1@work>")
        names = {m.filename for m in metas}
        self.assertEqual(names, {"report.pdf", "d.csv"})
        pdf = next(m for m in metas if m.filename == "report.pdf")
        self.assertEqual(pdf.mime, "application/pdf")
        self.assertEqual(pdf.source_type, "eml")
        self.assertEqual(pdf.source_ref, self.eml)
        self.assertTrue(pdf.thread_id)              # computed
        self.assertEqual(self.store.get_bytes(pdf.sha256), b"%PDF-1.4 fake")

    def test_reingest_is_idempotent(self):
        ingest_eml([self.eml], self.store)
        ingest_eml([self.eml], self.store)
        self.assertEqual(len(self.store.list_for(message_id="<m1@work>")), 2)


if __name__ == "__main__":
    unittest.main()
