"""Thread-linkage: loader captures threading headers and to_document() exposes
them as embed-excluded metadata (incl. a computed thread_id).

Imports llama_index (via models/loader), so this runs in the devcontainer.
"""

import os
import tempfile
import unittest

from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.models import NormalizedEmail

_EML = b"""From: Alice <alice@example.com>
To: bob@initech.com
Subject: Re: ACP test cases
Date: Wed, 1 Jan 2025 10:00:00 +0000
Message-ID: <reply@example.com>
In-Reply-To: <root@example.com>
References: <root@example.com> <mid@example.com>
Content-Type: text/plain; charset=utf-8

My reply body.
"""


class TestLoaderCapturesThreadHeaders(unittest.TestCase):
    def test_parses_message_id_in_reply_to_and_references(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "msg.eml"), "wb") as fh:
                fh.write(_EML)
            emails = MailArchiveXLoader(root).load()

        self.assertEqual(len(emails), 1)
        email = emails[0]
        self.assertEqual(email.message_id, "<reply@example.com>")
        self.assertEqual(email.in_reply_to, "<root@example.com>")
        self.assertEqual(email.references, "<root@example.com> <mid@example.com>")


class TestToDocumentThreadMetadata(unittest.TestCase):
    def _email(self, **kw):
        base = dict(
            sender="alice@example.com",
            subject="Re: ACP",
            date=None,
            body="hi",
            source="mail_archive_x",
            source_id="x.eml",
        )
        base.update(kw)
        return NormalizedEmail(**base)

    def test_thread_id_uses_references_root(self):
        email = self._email(
            message_id="<reply@example.com>",
            in_reply_to="<root@example.com>",
            references="<root@example.com> <mid@example.com>",
        )
        doc = email.to_document(doc_id="x")
        self.assertEqual(doc.metadata["thread_id"], "root@example.com")
        self.assertEqual(doc.metadata["message_id"], "<reply@example.com>")
        self.assertEqual(doc.metadata["in_reply_to"], "<root@example.com>")

    def test_thread_metadata_excluded_from_embedding(self):
        email = self._email(message_id="<m@x>", references="<r@x>")
        doc = email.to_document(doc_id="x")
        for key in ("thread_id", "message_id", "in_reply_to", "references"):
            self.assertIn(key, doc.excluded_embed_metadata_keys)
            self.assertIn(key, doc.excluded_llm_metadata_keys)

    def test_recipients_kept_in_payload_but_excluded_from_embedding(self):
        email = self._email(recipients="bob@initech.com, carol@example.com", cc="dave@umbrella.com")
        doc = email.to_document(doc_id="x")
        # still available as payload metadata (for "involving person X" filters)
        self.assertIn("to", doc.metadata)
        self.assertIn("cc", doc.metadata)
        # but not part of the embedded / LLM text
        for key in ("to", "to_full", "cc", "cc_full"):
            self.assertIn(key, doc.excluded_embed_metadata_keys)
            self.assertIn(key, doc.excluded_llm_metadata_keys)

    def test_root_email_groups_with_its_reply(self):
        root = self._email(message_id="<root@x>").to_document("a")
        reply = self._email(
            message_id="<c@x>", in_reply_to="<root@x>", references="<root@x>"
        ).to_document("b")
        self.assertEqual(root.metadata["thread_id"], reply.metadata["thread_id"])


if __name__ == "__main__":
    unittest.main()
