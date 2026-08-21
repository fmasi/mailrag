"""Shared pytest fixtures for the mailrag suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_local_state(tmp_path_factory, monkeypatch):
    """Never let the suite read the developer's real mailrag state.

    Two resolutions reach outside the repo when nothing is injected: the
    attachment store (``$RAG_ATTACH_STORE``, default ``~/.mailrag``) and corpus
    profiles (``$MAILRAG_PROFILE_DIR``, default ``~``), which decide collection
    scoping. Without this fixture the suite's result depends on whose home
    directory runs it — two tests were passing only because a real store and two
    real profiles happened to exist here, and identical code went green locally
    and red anywhere else.
    """
    monkeypatch.setenv("RAG_ATTACH_STORE", str(tmp_path_factory.mktemp("attach_store_isolated")))
    monkeypatch.setenv("MAILRAG_PROFILE_DIR", str(tmp_path_factory.mktemp("profiles_isolated")))
    monkeypatch.setenv("MAILRAG_HOME", str(tmp_path_factory.mktemp("mailrag_home_isolated")))

    from src.mcp_server import scoping

    scoping.clear_cache()
    yield
    scoping.clear_cache()
