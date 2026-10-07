import contextlib
import io
import types
import unittest
from unittest import mock

from src import cli


class TestAttachmentsCli(unittest.TestCase):
    def test_build_routes(self):
        prof = mock.Mock(selection_rules=[], blacklist=None)
        prof.resolved_root.return_value = "/root"
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.cli.resolve_index_files", return_value=(["/root/a.eml"], [])),
            mock.patch("src.cli.AttachmentStore") as store_cls,
            mock.patch(
                "src.cli.ingest_eml", return_value={"emails": 1, "attachments": 2, "skipped": 0}
            ) as ing,
        ):
            rc = cli.main(
                [
                    "attachments",
                    "build",
                    "--profile",
                    "p.json",
                    "--store",
                    "/tmp/st",
                    "--extractor",
                    "llm",
                ]
            )
        self.assertEqual(rc, 0)
        ing.assert_called_once()
        store_cls.assert_called_once_with("/tmp/st")

    def _build(self, *extra):
        import tempfile

        from src.attachments.store import AttachmentStore

        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        prof = mock.Mock(selection_rules=[], blacklist=None, collection="c")
        prof.resolved_root.return_value = "/root"
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.cli.resolve_index_files", return_value=([], [])),
            mock.patch("src.onboard.record_profile_for_collection"),
        ):
            rc = cli.main(
                ["attachments", "build", "--profile", "p.json", "--store", d, "--no-classify"]
                + list(extra)
            )
        self.assertEqual(rc, 0)
        store = AttachmentStore(d)
        self.addCleanup(store.close)
        store.put(
            b"x", message_id="<m>", thread_id="t", filename="a", mime="text/plain",
            size=1, source_type="eml", source_ref="/x",
        )  # fmt: skip
        return store

    def test_a_full_build_records_that_the_store_is_complete(self):
        """After it, an empty lookup means "no attachments"."""
        self.assertIsNone(self._build().build_gap())

    def test_a_limited_build_does_not(self):
        store = self._build("--limit", "5")
        store.mark_partial()
        self.assertIsNone(store.build_gap())  # non-empty and never marked partial
        self.assertIsNone(store.built_at())

    def test_get_text_routes(self):
        store = mock.Mock()
        store.fetch.return_value = {
            "sha256": "ab",
            "filename": "x.pdf",
            "mime": "application/pdf",
            "size": 3,
            "text": "hello",
            "text_status": "extracted",
            "path": "/tmp/st/blobs/ab/abc",
        }
        with mock.patch("src.cli.AttachmentStore", return_value=store):
            rc = cli.main(["attachments", "get", "abc", "--text", "--store", "/tmp/st"])
        self.assertEqual(rc, 0)
        store.fetch.assert_called_once_with("abc", extractor=None, force=False)

    def test_list_routes(self):
        store = mock.Mock()
        store.list_for.return_value = []
        with mock.patch("src.cli.AttachmentStore", return_value=store):
            rc = cli.main(["attachments", "list", "--thread-id", "t1", "--store", "/tmp/st"])
        self.assertEqual(rc, 0)
        store.list_for.assert_called_once_with(thread_id="t1", message_id=None)

    def test_list_prints_full_sha256(self):
        """list must print the full 64-char sha256 so it can be piped into `get`."""
        full_sha = "a" * 64
        meta = types.SimpleNamespace(
            sha256=full_sha,
            size=1024,
            mime="application/pdf",
            filename="report.pdf",
        )
        store = mock.Mock()
        store.list_for.return_value = [meta]
        buf = io.StringIO()
        with (
            mock.patch("src.cli.AttachmentStore", return_value=store),
            contextlib.redirect_stdout(buf),
        ):
            rc = cli.main(["attachments", "list", "--thread-id", "t1", "--store", "/tmp/st"])
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        # Full sha must be present
        self.assertIn(full_sha, output)
        # The old 12-char truncation must NOT be the entire token shown
        # (i.e. the line must contain more than just the first 12 chars)
        first_line = output.splitlines()[0]
        self.assertTrue(
            first_line.startswith(full_sha),
            f"line should start with full sha256, got: {first_line!r}",
        )

    def test_get_passes_extractor_and_force(self):
        store = mock.Mock()
        store.fetch.return_value = {
            "sha256": "ab",
            "filename": "x.png",
            "mime": "image/png",
            "size": 3,
            "text": "t",
            "text_status": "extracted",
            "path": "/p",
        }
        with mock.patch("src.cli.AttachmentStore", return_value=store):
            rc = cli.main(
                [
                    "attachments",
                    "get",
                    "abc",
                    "--text",
                    "--extractor",
                    "tesseract",
                    "--force",
                    "--store",
                    "/tmp/st",
                ]
            )
        self.assertEqual(rc, 0)
        store.fetch.assert_called_once_with("abc", extractor="tesseract", force=True)


if __name__ == "__main__":
    unittest.main()
