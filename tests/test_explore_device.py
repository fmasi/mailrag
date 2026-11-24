"""The scan fresh-embed path must resolve its device via pick_device(),
not a hardcoded "mps" (crashes on Linux/CPU — see issue #N)."""

import pytest


@pytest.fixture
def fake_embedder(monkeypatch):
    """Capture the (device, use_fp16) BgeM3Embedder is constructed with."""
    captured = {}

    class _FakeModel:
        def encode(self, texts, **kwargs):
            return {"dense_vecs": [[0.0, 1.0]] * len(texts), "lexical_weights": [{} for _ in texts]}

    def _fake_init(self, model_name="BAAI/bge-m3", device=None, use_fp16=True):
        captured["device"] = device
        captured["use_fp16"] = use_fp16
        self.name = model_name
        self.model = _FakeModel()
        self.dim = 2
        self.produces_sparse = True

    def _fake_encode(self, texts, batch_size=32, max_length=512):
        return [0.0] * (len(texts) * 2), [{} for _ in texts]

    monkeypatch.setattr("src.pipeline.explore.BgeM3Embedder.__init__", _fake_init)
    monkeypatch.setattr("src.pipeline.explore.BgeM3Embedder.encode", _fake_encode)
    return captured


def _thread_vectors(**kwargs):
    import src.pipeline.explore as explore
    return explore._fresh_thread_vectors(**kwargs)


def _fake_email(i, body="some body"):
    from types import SimpleNamespace
    return SimpleNamespace(body=body, message_id=f"<m{i}@x>", in_reply_to="", references="", subject="s", sender="a@b.c", noise_candidate=False, is_bulk=False, source_id=f"/tmp/{i}.eml")


def test_fresh_thread_vectors_respects_cpu_override(fake_embedder, monkeypatch):
    monkeypatch.setenv("RAG_EMBED_DEVICE", "cpu")
    emails = [_fake_email(0), _fake_email(1)]
    tid_to_idx = {"t0": [0, 1]}

    _thread_vectors(emails=emails, tid_to_idx=tid_to_idx)

    assert fake_embedder["device"] == "cpu"
    assert fake_embedder["use_fp16"] is False  # fp16 is a GPU-only win


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_fresh_thread_vectors_keeps_fp16_on_gpu(fake_embedder, monkeypatch, device):
    monkeypatch.setenv("RAG_EMBED_DEVICE", device)
    emails = [_fake_email(0)]
    tid_to_idx = {"t0": [0]}

    _thread_vectors(emails=emails, tid_to_idx=tid_to_idx)

    assert fake_embedder["device"] == device
    assert fake_embedder["use_fp16"] is True  # Apple-Silicon path unchanged
