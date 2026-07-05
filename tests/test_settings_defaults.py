import importlib
import unittest
from unittest.mock import MagicMock, patch


class TestDefaults(unittest.TestCase):
    # Snapshot of RAGConfig class attributes that may be mutated by
    # load_from_env() so we can restore them in tearDown and avoid
    # polluting subsequent tests.
    _RAGCONFIG_ATTRS = (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_API_BASE",
        "LLM_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_API_BASE",
        "EMBEDDING_API_KEY",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
    )

    def setUp(self):
        from src.config.settings import RAGConfig

        self._ragconfig_snapshot = {k: getattr(RAGConfig, k) for k in self._RAGCONFIG_ATTRS}

    def tearDown(self):
        from src.config.settings import RAGConfig

        for k, v in self._ragconfig_snapshot.items():
            setattr(RAGConfig, k, v)

    def test_default_collection_is_demo(self):
        from src.config import settings

        importlib.reload(settings)
        self.assertEqual(settings.RAGConfig.QDRANT_COLLECTION_NAME, "mailrag-demo")

    def test_initialize_settings_can_skip_embeddings(self):
        """include_embeddings=False must not instantiate OpenAIEmbedding and must
        not assign Settings.embed_model.  The LlamaIndex embeddings package must
        not be required (proving the lazy-import guard works).

        Uses clear=True in patch.dict to isolate from the real process environment
        so load_from_env() inside initialize_settings cannot pollute RAGConfig class
        attributes with values from a .env file loaded by another module.
        """
        from src.config.settings import RAGConfig

        # Intercept the lazy import so we can track whether OpenAIEmbedding is
        # called.  Patching sys.modules means the `from llama_index.embeddings.openai
        # import OpenAIEmbedding` inside initialize_settings will resolve to our mock.
        fake_oai_embed_cls = MagicMock()
        fake_embed_module = MagicMock()
        fake_embed_module.OpenAIEmbedding = fake_oai_embed_cls

        # Capture every attribute set on the (fake) Settings object via a
        # simple dict-backed recorder — avoids the MagicMock.__setattr__ quirk.
        recorded_sets: dict = {}

        class _FakeSettings:
            def __setattr__(cls_self, name, value):  # noqa: N805
                recorded_sets[name] = value

        # clear=True: prevent load_from_env() from reading RAG_LLM_API_KEY etc.
        # from the real environment (loaded from .env by other imported modules).
        with (
            patch("src.config.settings.Settings", _FakeSettings()),
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"llama_index.embeddings.openai": fake_embed_module}),
        ):
            RAGConfig.initialize_settings(include_llm=False, include_embeddings=False)

        # embed_model must never have been assigned on Settings
        self.assertNotIn(
            "embed_model",
            recorded_sets,
            "Settings.embed_model was assigned even though include_embeddings=False",
        )
        # OpenAIEmbedding must not have been instantiated
        fake_oai_embed_cls.assert_not_called()

    def test_initialize_settings_default_still_sets_embed_model(self):
        """Default call (include_embeddings=True) must assign Settings.embed_model.

        Uses clear=True in patch.dict to avoid leaking any env-var side effects
        (e.g. RAG_LLM_API_KEY from a .env file) into subsequent tests via
        RAGConfig class attributes set by load_from_env().
        """
        from src.config.settings import RAGConfig

        fake_embed_instance = MagicMock()
        fake_oai_embed_cls = MagicMock(return_value=fake_embed_instance)
        fake_embed_module = MagicMock()
        fake_embed_module.OpenAIEmbedding = fake_oai_embed_cls

        recorded_sets: dict = {}

        class _FakeSettings:
            def __setattr__(cls_self, name, value):  # noqa: N805
                recorded_sets[name] = value

        # clear=True: isolate from the real process environment so load_from_env()
        # inside initialize_settings cannot read RAG_LLM_API_KEY etc. from .env.
        with (
            patch("src.config.settings.Settings", _FakeSettings()),
            patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "test-key", "RAG_EMBEDDING_PROVIDER": "openai"},
                clear=True,
            ),
            patch.dict("sys.modules", {"llama_index.embeddings.openai": fake_embed_module}),
        ):
            RAGConfig.initialize_settings(include_llm=False, include_embeddings=True)

        # embed_model must have been assigned
        self.assertIn(
            "embed_model",
            recorded_sets,
            "Settings.embed_model was NOT assigned when include_embeddings=True",
        )
        self.assertIs(
            recorded_sets["embed_model"],
            fake_embed_instance,
            "Settings.embed_model was not set to the OpenAIEmbedding instance",
        )
