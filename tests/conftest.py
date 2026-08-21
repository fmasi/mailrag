"""Shared pytest fixtures for the mailrag suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_local_state(tmp_path_factory, monkeypatch):
    """Never let the suite read the developer's real mailrag state.

    Three resolutions reach outside the repo when nothing is injected: the
    attachment store (``$RAG_ATTACH_STORE``, default ``~/.mailrag``), corpus
    profiles (``$MAILRAG_PROFILE_DIR``, default ``~``), which decide collection
    scoping, and the raw ``.eml`` corpus root (``$MAILRAG_EML_ROOT``, default
    ``~/rag_eml``) that ``grep_email`` walks. Without this fixture the suite's
    result depends on whose home directory runs it — two tests were passing only
    because a real store and two real profiles happened to exist here, and
    identical code went green locally and red anywhere else.

    ``MAILRAG_EML_ROOT`` was missed by the first sweep and cost a second round of
    exactly that bug: a scoping test passed on a machine with a real ``~/rag_eml``
    and failed on CI, where the default root does not exist. It is pinned to a
    path that does **not** exist, which is what an unconfigured machine actually
    looks like (CI has no ``/home/runner/rag_eml``), so "no corpus root" is the
    reproducible baseline everywhere. Tests needing real files pass an explicit
    ``root``; tests about the missing-root path get it for free.
    """
    monkeypatch.setenv("RAG_ATTACH_STORE", str(tmp_path_factory.mktemp("attach_store_isolated")))
    monkeypatch.setenv("MAILRAG_PROFILE_DIR", str(tmp_path_factory.mktemp("profiles_isolated")))
    monkeypatch.setenv("MAILRAG_HOME", str(tmp_path_factory.mktemp("mailrag_home_isolated")))
    monkeypatch.setenv(
        "MAILRAG_EML_ROOT", str(tmp_path_factory.mktemp("eml_root_isolated") / "absent_rag_eml")
    )

    from src.mcp_server import scoping

    scoping.clear_cache()
    yield
    scoping.clear_cache()
