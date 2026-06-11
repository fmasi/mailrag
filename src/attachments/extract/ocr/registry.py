"""Resolve an OCR provider name to a provider instance, and read the configured default.

Provider names: 'tesseract' (local OCR), 'llm' (local vision model via LM Studio/Gemma),
'cloud' (reserved off-device seam, opt-in, not implemented).
"""
from __future__ import annotations

import os

from src.attachments.extract.ocr.base import ChainedOcr, OcrProvider
from src.attachments.extract.ocr.tesseract import TesseractOcr

_DEFAULT = "llm"   # privacy-first: local vision-LLM is the default reader


def default_extractor_name() -> str:
    return os.getenv("RAG_ATTACH_EXTRACTOR", _DEFAULT).strip() or _DEFAULT


def resolve(name: str) -> OcrProvider:
    name = (name or _DEFAULT).strip().lower()
    if name == "tesseract":
        return TesseractOcr()
    if name == "cloud":
        raise NotImplementedError(
            "cloud OCR is opt-in and not implemented yet (off-device; reserved seam)")
    if name == "llm":
        from src.llm.client import make_client, default_model
        from src.attachments.extract.ocr.llm_vision import LlmVision
        try:
            client, model = make_client(), default_model()
        except Exception:
            client, model = None, ""
        return ChainedOcr([LlmVision(client=client, model=model), TesseractOcr()])
    raise ValueError(f"unknown extractor '{name}'")
