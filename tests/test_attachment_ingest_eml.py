import os
import shutil
import tempfile
import unittest
from email.message import EmailMessage

from src.attachments.ingest_eml import _decode_filename, ingest_eml
from src.attachments.store import AttachmentStore
from src.data.loaders.mail_archive_x import MailArchiveXLoader


def _write_eml(path):
    m = EmailMessage()
    m["From"] = "alice@work.com"
    m["To"] = "bob@work.com"
    m["Subject"] = "Report"
    m["Message-ID"] = "<m1@work>"
    m.set_content("See attached.")
    m.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="report.pdf")
    m.add_attachment(b"col\n1\n", maintype="text", subtype="csv", filename="d.csv")
    with open(path, "wb") as fh:
        fh.write(bytes(m))


class TestIngestEml(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.eml = os.path.join(self.d, "msg.eml")
        _write_eml(self.eml)
        self.store = AttachmentStore(os.path.join(self.d, "store"))
        self.addCleanup(self.store.close)

    def tearDown(self):
        pass

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
        self.assertTrue(pdf.thread_id)  # computed
        self.assertEqual(self.store.get_bytes(pdf.sha256), b"%PDF-1.4 fake")

    def test_reingest_is_idempotent(self):
        ingest_eml([self.eml], self.store)
        ingest_eml([self.eml], self.store)
        self.assertEqual(len(self.store.list_for(message_id="<m1@work>")), 2)

    def test_thread_id_matches_indexer_with_mbox_preamble(self):
        """Regression: Mail Archive X .eml carry an mbox 'From ' preamble. The loader
        strips it (so the indexer gets the real Message-ID); ingest must too, or the
        attachment->thread join breaks. ingest's thread_id must equal the indexer's."""
        m = EmailMessage()
        m["From"] = "alice@work.com"
        m["To"] = "bob@work.com"
        m["Subject"] = "Q3 numbers"
        m["Message-ID"] = "<real-mid@work>"
        m.set_content("see attached")
        m.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="q3.pdf")
        # Real Mail Archive X preamble: an mbox 'From ' line AND an extra numeric
        # field, which breaks naive header parsing (Message-ID is lost).
        path = os.path.join(self.d, "with_preamble.eml")
        with open(path, "wb") as fh:
            fh.write(b"From <alice@work.com> Fri Oct  9 12:24:48 2025\n188035    \n" + bytes(m))

        # the indexer's thread_id for this exact file (via the loader -> to_document)
        e = MailArchiveXLoader(eml_files=[path], verbose=False).load()[0]
        indexer_tid = e.to_document(doc_id="x").metadata["thread_id"]

        ingest_eml([path], self.store)
        # joined by the REAL message-id (empty/wrong if the preamble broke the parse)
        metas = self.store.list_for(message_id="<real-mid@work>")
        self.assertTrue(metas, "attachment not found by its real Message-ID")
        self.assertEqual(metas[0].thread_id, indexer_tid)

    def test_text_attachment_mime_carries_declared_charset(self):
        """The declared charset must survive into the stored mime so extraction can
        decode non-UTF-8 text attachments correctly (instead of latin-1 mojibake)."""
        raw = (
            b"From: alice@work.com\r\nTo: bob@work.com\r\nSubject: Enc\r\n"
            b"Message-ID: <enc@work>\r\nMIME-Version: 1.0\r\n"
            b'Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
            b"--B\r\nContent-Type: text/plain\r\n\r\nbody\r\n"
            b'--B\r\nContent-Type: text/plain; charset="iso-8859-1"\r\n'
            b'Content-Disposition: attachment; filename="latin.txt"\r\n'
            b"Content-Transfer-Encoding: 8bit\r\n\r\n"
            + "h\xe9llo".encode("iso-8859-1")
            + b"\r\n--B--\r\n"
        )
        path = os.path.join(self.d, "charset.eml")
        with open(path, "wb") as fh:
            fh.write(raw)
        ingest_eml([path], self.store)
        metas = self.store.list_for(message_id="<enc@work>")
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0].mime, "text/plain; charset=iso-8859-1")

    def test_non_text_attachment_mime_stays_bare(self):
        ingest_eml([self.eml], self.store)
        pdf = next(
            m for m in self.store.list_for(message_id="<m1@work>") if m.filename == "report.pdf"
        )
        self.assertEqual(pdf.mime, "application/pdf")

    def test_encoded_word_filename_is_decoded(self):
        # RFC2047 base64 (ISO-8859-15) and quoted-printable forms
        self.assertEqual(_decode_filename("=?ISO-8859-15?B?QXRlbufjby5naWY=?="), "Atenção.gif")
        self.assertEqual(_decode_filename("=?iso-8859-1?Q?Twins1.jpg?="), "Twins1.jpg")
        # plain ascii passes through unchanged; None -> ""
        self.assertEqual(_decode_filename("plain.pdf"), "plain.pdf")
        self.assertEqual(_decode_filename(None), "")
        # Unknown charset raises inside make_header -> except branch returns raw string unchanged
        self.assertEqual(
            _decode_filename("=?not-a-charset?B?aGVsbG8=?="), "=?not-a-charset?B?aGVsbG8=?="
        )


def _raw_message(parts: bytes, headers: bytes = b"Message-ID: <p@work>\r\nSubject: Report\r\n"):
    return (
        b"From: alice@example.com\r\n" + headers + b"MIME-Version: 1.0\r\n"
        b"Content-Type: multipart/mixed; boundary=B\r\n\r\n" + parts + b"--B--\r\n"
    )


def _part(disposition: bytes, body: bytes = b"DATA") -> bytes:
    return (
        b"--B\r\nContent-Type: application/octet-stream\r\n"
        b"Content-Disposition: " + disposition + b"\r\n\r\n" + body + b"\r\n"
    )


# The stdlib parses ``filename*N`` continuation numbers with int(), which refuses
# more than 4300 digits: reading this part's filename raises ValueError.
UNREADABLE = b"attachment; filename*" + b"1" * 4301 + b"=x"


class TestHostileMessages(unittest.TestCase):
    """Every header is whatever the sender typed. Sync runs this ingest
    unattended, so a message that raises out of it stops the whole account."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.store = AttachmentStore(os.path.join(self.d, "store"))
        self.addCleanup(self.store.close)

    def _eml(self, name, raw):
        path = os.path.join(self.d, name)
        with open(path, "wb") as fh:
            fh.write(raw)
        return path

    def test_a_part_whose_headers_cannot_be_parsed_costs_only_that_part(self):
        poison = self._eml(
            "p.eml",
            _raw_message(_part(UNREADABLE) + _part(b'attachment; filename="ok.bin"', b"OK")),
        )
        counts = ingest_eml([poison], self.store)
        self.assertEqual(counts["attachments"], 1)
        self.assertEqual(counts["bad_parts"], 1, "a dropped part must be counted somewhere")
        (meta,) = self.store.list_for(message_id="<p@work>")
        self.assertEqual(meta.filename, "ok.bin")

    def test_the_next_message_is_still_ingested(self):
        poison = self._eml("p.eml", _raw_message(_part(UNREADABLE)))
        good = self._eml(
            "g.eml",
            _raw_message(
                _part(b'attachment; filename="g.bin"', b"GOOD"),
                headers=b"Message-ID: <g@work>\r\nSubject: Fine\r\n",
            ),
        )
        ingest_eml([poison, good], self.store)
        self.assertEqual(len(self.store.list_for(message_id="<g@work>")), 1)

    def test_an_unencodable_subject_does_not_cost_the_attachment(self):
        """With no Message-ID the thread id is derived from the subject. One that
        decodes to a lone surrogate made every part of the message unstorable."""
        raw = _raw_message(
            _part(b'attachment; filename="a.bin"'),
            headers=b"Subject: =?utf-7?q?+2AA-?=\r\n",
        )
        counts = ingest_eml([self._eml("s.eml", raw)], self.store)
        self.assertEqual((counts["attachments"], counts["bad_parts"]), (1, 0))
        self.assertEqual(self.store.count(), 1)

    def test_a_message_stuffed_with_parts_is_capped(self):
        """A few megabytes of tiny parts became a file and a commit each: 100,000
        of them took a minute and 400 MB of disk blocks."""
        from src.attachments.ingest_eml import MAX_PARTS_PER_MESSAGE

        many = b"".join(
            _part(b'attachment; filename="f.bin"', str(i).encode())
            for i in range(MAX_PARTS_PER_MESSAGE + 50)
        )
        counts = ingest_eml([self._eml("m.eml", _raw_message(many))], self.store)
        self.assertEqual(counts["attachments"], MAX_PARTS_PER_MESSAGE)
        self.assertEqual(self.store.count(), MAX_PARTS_PER_MESSAGE)
        self.assertEqual(counts["bad_parts"], 50, "parts past the cap are counted, not lost")

    def test_sender_typed_text_is_stored_at_a_bounded_length(self):
        """A 100 KB subject on a message with no Message-ID became the thread id
        on every one of its rows: a 153 KB email grew the index by 103 MB, and
        long filenames put a megabyte of names on a single search hit."""
        raw = _raw_message(
            _part(b'attachment; filename="' + b"n" * 5000 + b'.bin"'),
            headers=b"Subject: " + b"s" * 100_000 + b"\r\n",
        )
        ingest_eml([self._eml("big.eml", raw)], self.store)
        (meta,) = self.store.list_for()
        self.assertLessEqual(len(meta.filename), 512)
        self.assertLessEqual(len(meta.thread_id), 1024)
        self.assertLessEqual(len(meta.message_id), 1024)

    def test_an_ordinary_id_and_name_are_stored_unchanged(self):
        raw = _raw_message(_part(b'attachment; filename="report.pdf"'))
        ingest_eml([self._eml("ok.eml", raw)], self.store)
        (meta,) = self.store.list_for(message_id="<p@work>")
        self.assertEqual(meta.filename, "report.pdf")


if __name__ == "__main__":
    unittest.main()
