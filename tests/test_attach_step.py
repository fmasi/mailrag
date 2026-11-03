"""The recipes' `attach` step fills the attachment store agents open files from."""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.attachments.build import build_store
from src.attachments.store import AttachmentStore
from src.persona.registry import load_registry
from src.persona.runner import build_handlers
from src.tui import flow


def _eml(path, with_pdf=True):
    body = (
        "From: a@example.com\nTo: b@example.com\nSubject: invoice\n"
        "Message-ID: <m1@example.com>\nMIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="B"\n\n'
        "--B\nContent-Type: text/plain\n\nsee attached\n"
    )
    if with_pdf:
        body += (
            "--B\nContent-Type: application/pdf\n"
            'Content-Disposition: attachment; filename="inv.pdf"\n'
            "Content-Transfer-Encoding: base64\n\nJVBERi0xLjQK\n"
        )
    body += "--B--\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body)


class _Case(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.root = os.path.join(self.d, "mail")
        _eml(os.path.join(self.root, "Inbox", "a.eml"))
        self.store_dir = os.path.join(self.d, "store")
        self.prof = SimpleNamespace(
            resolved_root=lambda: self.root,
            selection_rules=[{"type": "prefix", "value": "Inbox/"}],
            blacklist=None,
            collection="box",
        )

    def _store(self):
        store = AttachmentStore(self.store_dir)
        self.addCleanup(store.close)
        return store


class TestBuildStore(_Case):
    def test_stores_the_attachments_and_marks_the_store_complete(self):
        counts = build_store(self.prof, self.store_dir, classify=False, progress=False)
        self.assertEqual(counts["emails"], 1)
        self.assertEqual(counts["attachments"], 1)
        self.assertTrue(counts["complete"])
        self.assertIsNone(self._store().build_gap())

    def test_a_limited_run_is_not_complete(self):
        counts = build_store(self.prof, self.store_dir, limit=1, classify=False, progress=False)
        self.assertFalse(counts["complete"])
        self.assertEqual(self._store().build_gap(), "partial")


class TestAttachStepInTheRecipes(_Case):
    def test_every_shipped_recipe_ends_by_storing_attachments(self):
        reg = load_registry()
        for name in reg.names():
            verbs = [s.verb for s in reg.get(name).steps]
            self.assertEqual(verbs[-2:], ["index", "attach"], name)

    def test_the_handler_builds_the_collections_store(self):
        handlers = build_handlers(profile_path=os.path.join(self.d, "box.profile.json"))
        with (
            mock.patch(
                "src.attachments.location.resolve_attach_store", return_value=self.store_dir
            ),
            mock.patch("src.onboard.record_profile_for_collection") as record,
            mock.patch("src.attachments.classify.classify_blobs") as classify,
        ):
            classify.return_value.as_dict.return_value = {}
            counts = handlers["attach"](self.prof)
        record.assert_called_once()
        self.assertEqual(counts["attachments"], 1)
        self.assertTrue(counts["complete"])

    def test_the_run_log_says_it_in_a_sentence(self):
        text = flow.describe_result({"emails": 3, "attachments": 5, "complete": True}, "attach")
        self.assertEqual(text, "5 attachments from 3 emails stored; agents can open them now")
        partial = flow.describe_result({"emails": 3, "attachments": 5, "complete": False})
        self.assertIn("partial run", partial)


if __name__ == "__main__":
    unittest.main()
