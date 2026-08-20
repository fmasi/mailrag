"""Shared pytest fixtures for the mailrag suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_attachment_store(tmp_path_factory, monkeypatch):
    """Never let the suite read the developer's real attachment store.

    Several tools resolve ``$RAG_ATTACH_STORE`` (or its ``~/.mailrag`` default)
    when no store is injected, and one now inspects that directory to refuse a
    pre-split shared store. Without this fixture, whether the suite passes
    depends on what happens to be in the home directory of whoever runs it —
    which is how identical code goes green on CI and red locally.
    """
    monkeypatch.setenv("RAG_ATTACH_STORE", str(tmp_path_factory.mktemp("attach_store_isolated")))
