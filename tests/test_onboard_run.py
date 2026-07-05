import unittest
from unittest import mock

from src import onboard


class _Email:
    def __init__(self, mid):
        self.message_id = mid
        self.body = "some body text"
        self.summary = None


class _BuildResult:
    chunks = 7
    collection = "mailrag-x"


class TestRunOnboard(unittest.TestCase):
    def test_sequences_stages_and_returns_report(self):
        emails = [_Email("a"), _Email("b"), _Email("c")]
        judgments = {
            "a": {"is_noise": False, "confidence": 0.9, "summary": "s", "reason": ""},
            "b": {"is_noise": True, "confidence": 0.95, "summary": "", "reason": ""},
            "c": {"is_noise": False, "confidence": 0.1, "summary": "", "reason": "llm_error: boom"},
        }
        with (
            mock.patch.object(onboard, "_require_qdrant"),
            mock.patch.object(onboard, "load_eml_dir", return_value=emails),
            mock.patch.object(onboard, "assign_subject_fallback_thread_ids"),
            mock.patch("src.llm.cache.Pass2Cache"),
            mock.patch("src.llm.onboard_pass.generate_thread_judgments", return_value=judgments),
            mock.patch(
                "src.indexing.contextual_index.build_contextual_index", return_value=_BuildResult()
            ) as build,
            mock.patch.object(onboard, "write_manifest", return_value="/m.json"),
        ):
            report = onboard.run_onboard(
                "/some/dir", collection="mailrag-x", chunk_size=256, embedder="EMB", validate=False
            )
        self.assertEqual(report.collection, "mailrag-x")
        self.assertEqual(report.kept, 2)  # b dropped (confident noise)
        self.assertEqual(report.noise_dropped, 1)
        self.assertEqual(report.llm_failures, 1)  # c failed but kept
        self.assertEqual(report.chunks, 7)
        self.assertFalse(report.validated)
        # build_contextual_index got embed_summary=True + our chunk_size
        self.assertEqual(build.call_args.kwargs["embed_summary"], True)
        self.assertEqual(build.call_args.kwargs["chunk_size"], 256)

    def test_all_noise_raises(self):
        emails = [_Email("a")]
        judgments = {"a": {"is_noise": True, "confidence": 0.99, "summary": "", "reason": ""}}
        with (
            mock.patch.object(onboard, "_require_qdrant"),
            mock.patch.object(onboard, "load_eml_dir", return_value=emails),
            mock.patch.object(onboard, "assign_subject_fallback_thread_ids"),
            mock.patch("src.llm.cache.Pass2Cache"),
            mock.patch("src.llm.onboard_pass.generate_thread_judgments", return_value=judgments),
        ):
            with self.assertRaises(ValueError):
                onboard.run_onboard(
                    "/d", collection="c", chunk_size=256, embedder="EMB", validate=False
                )


if __name__ == "__main__":
    unittest.main()
