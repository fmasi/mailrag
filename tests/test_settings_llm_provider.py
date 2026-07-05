"""The lmstudio LLM provider must build a LlamaIndex ``OpenAILike`` so the
answer side (``Settings.llm``) uses the SAME LLM abstraction as the cleanup
client (``src.llm.client``). This is the P2 Step-3 unification."""

import unittest
from unittest.mock import MagicMock, patch


class TestLmStudioBuildsOpenAILike(unittest.TestCase):
    _RAGCONFIG_ATTRS = (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_API_BASE",
        "LLM_API_KEY",
    )

    def setUp(self):
        from src.config.settings import RAGConfig

        self._snapshot = {k: getattr(RAGConfig, k) for k in self._RAGCONFIG_ATTRS}

    def tearDown(self):
        from src.config.settings import RAGConfig

        for k, v in self._snapshot.items():
            setattr(RAGConfig, k, v)

    def test_lmstudio_provider_sets_settings_llm_to_openai_like(self):
        from src.config.settings import RAGConfig

        fake_instance = MagicMock(name="OpenAILike-instance")
        fake_cls = MagicMock(return_value=fake_instance)
        fake_module = MagicMock()
        fake_module.OpenAILike = fake_cls

        recorded: dict = {}

        class _FakeSettings:
            def __setattr__(cls_self, name, value):  # noqa: N805
                recorded[name] = value

        env = {
            "RAG_LLM_PROVIDER": "lmstudio",
            "RAG_LLM_MODEL": "gemma-3-12b",
            "RAG_LLM_API_BASE": "http://host.docker.internal:1234/v1",
        }
        with (
            patch("src.config.settings.Settings", _FakeSettings()),
            patch.dict("os.environ", env, clear=True),
            patch.dict("sys.modules", {"llama_index.llms.openai_like": fake_module}),
        ):
            RAGConfig.initialize_settings(include_llm=True, include_embeddings=False)

        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemma-3-12b")
        self.assertEqual(kwargs["api_base"], "http://host.docker.internal:1234/v1")
        self.assertTrue(kwargs["is_chat_model"])
        self.assertIs(recorded.get("llm"), fake_instance)


if __name__ == "__main__":
    unittest.main()
