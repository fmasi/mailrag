"""Local vision-LLM OCR provider (gemma-4 via LM Studio). Fully on-device.

Returns a structured description + verbatim transcription. Images are downscaled to
bound vision tokens; scanned PDFs are rendered to images (page-capped, truncation
logged). If the model/endpoint is unavailable the result signals OCR_UNAVAILABLE so a
ChainedOcr falls through to tesseract.
"""
from __future__ import annotations

import io
import os

from src.attachments.extract.ocr.base import OcrResult
from src.attachments.extract.result import Status

_NAME = "llm_vision"
_MAX_EDGE = 2048
_PROMPT = (
    "You are reading an email attachment image. Reply in EXACTLY this format:\n"
    "DESCRIPTION: <one or two factual sentences describing the image>\n"
    "TEXT:\n"
    "<verbatim transcription of all readable text, or \"(none)\" if there is none>"
)


def _max_pages() -> int:
    try:
        return max(1, int(os.getenv("RAG_ATTACH_LLM_MAX_PAGES", "10")))
    except ValueError:
        return 10


def _downscale_png(data: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    im.thumbnail((_MAX_EDGE, _MAX_EDGE))
    out = io.BytesIO(); im.convert("RGB").save(out, format="PNG")
    return out.getvalue()


class LlmVision:
    def __init__(self, client, model: str, chat_vision=None, log=None):
        self._client = client
        self._model = model
        if chat_vision is None:
            from src.llm.client import chat_vision as _cv
            chat_vision = _cv
        self._chat_vision = chat_vision
        self._log = log or (lambda *_: None)

    def read(self, data: bytes, mime: str, filename: str) -> OcrResult:
        if not self._model:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        name = (filename or "").lower()
        try:
            if mime == "application/pdf" or name.endswith(".pdf"):
                pages = self._render_pdf(data)
            else:
                pages = [_downscale_png(data)]
        except Exception:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)  # libs (PIL/poppler) missing
        try:
            chunks = [self._chat_vision(self._client, self._model, _PROMPT, p, "image/png")
                      for p in pages]
        except Exception:
            # endpoint down / connection / model error -> let the chain fall through
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        text = "\n\n".join(chunks).strip()
        return OcrResult(text, Status.EXTRACTED if text else Status.EMPTY, _NAME)

    def _render_pdf(self, data: bytes):
        import pdf2image
        images = pdf2image.convert_from_bytes(data)
        cap = _max_pages()
        if len(images) > cap:
            self._log(f"llm_vision: PDF has {len(images)} pages; sending first {cap}")
            images = images[:cap]
        out = []
        for im in images:
            im = im.convert("RGB")
            im.thumbnail((_MAX_EDGE, _MAX_EDGE))
            buf = io.BytesIO(); im.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
