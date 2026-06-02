"""Pick a torch device for FlagEmbedding, portable across mac/linux/CI."""
import os

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False

def pick_device() -> str:
    """RAG_EMBED_DEVICE overrides; else cuda > mps > cpu."""
    forced = os.getenv("RAG_EMBED_DEVICE", "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        return forced
    if _has_cuda():
        return "cuda"
    if _has_mps():
        return "mps"
    return "cpu"
