import unittest, importlib


class TestDefaults(unittest.TestCase):
    def test_default_collection_is_demo(self):
        from src.config import settings
        importlib.reload(settings)
        self.assertEqual(settings.RAGConfig.QDRANT_COLLECTION_NAME, "mailrag-demo")
