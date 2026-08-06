"""Which model, at which quantisation, from where — recorded, not assumed.

A model id alone does not identify a judge. The same ``gemma-4-26b-a4b-it-mlx``
at a different quantisation produces different output, and a local endpoint
differs again from a hosted one in cost, latency and sometimes in result. A cache
of 25,000 judgments over one corpus is only comparable with itself if you can
tell which judge produced each row — otherwise a shift in noise rate is
unattributable: prompt, rubric, or a model file swapped months ago?

So provenance is captured as data next to the judgments, not written to a log
that scrolls away. LM Studio's native API reports the loaded quantisation; where
it is unavailable the fields degrade to empty rather than to a guess.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import urlparse

# Hosts that mean "this ran on my machine". Anything else is treated as remote,
# which is the safe direction: mislabelling a hosted endpoint as local would
# understate both cost and data exposure.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


@dataclass(frozen=True)
class Provenance:
    """Everything needed to say which judge produced a result."""

    model: str = ""
    quant: str = ""
    arch: str = ""
    endpoint: str = ""
    source: str = ""  # "local" | "remote"

    def label(self) -> str:
        """One line for a run header: ``model @ quant (local, http://…)``."""
        bits = self.model or "(unknown model)"
        if self.quant:
            bits += f" @ {self.quant}"
        where = self.source or "unknown"
        if self.endpoint:
            where += f", {self.endpoint}"
        return f"{bits} ({where})"

    def as_dict(self) -> dict:
        return asdict(self)


def classify_source(api_base: str) -> str:
    """``"local"`` when the endpoint is on this machine, else ``"remote"``."""
    try:
        host = (urlparse(str(api_base)).hostname or "").lower()
    except (ValueError, TypeError):
        return "unknown"
    if not host:
        return "unknown"
    return "local" if host in _LOCAL_HOSTS else "remote"


def _lmstudio_model_info(api_base: str, model: str, api_key: str = "") -> dict:
    """Ask LM Studio's native API about *model*. Returns {} if unavailable.

    Deliberately best-effort: this is metadata capture, and must never be the
    reason a sweep fails to start.
    """
    try:
        root = str(api_base).rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        req = urllib.request.Request(f"{root}/api/v0/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - operator-configured
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        for entry in payload.get("data", []) or []:
            if entry.get("id") == model:
                return entry
        return {}
    except Exception:  # noqa: BLE001 — provenance is metadata; it must never
        # be the reason a sweep fails to start. A caller passing something
        # unexpected (a mock, a non-string) degrades to "unknown", not a crash.
        return {}


def describe_backend(model: str = "", api_base: str = "", api_key: str = "") -> Provenance:
    """Best-effort provenance for the configured LLM backend.

    Falls back to the environment for anything not passed in, so callers can just
    call it. Never raises.
    """
    # str() because a caller may hand us a client attribute of any type; this
    # helper degrades rather than raising.
    api_base = str(api_base or os.getenv("RAG_LLM_API_BASE", "")).strip()
    model = str(model or os.getenv("RAG_LLM_MODEL", "")).strip()
    if not api_key:
        # Resolve through the SAME path the client uses. Reading the env var raw
        # would send "Bearer keychain:mailrag.llm.token" — the reference, not the
        # secret — and the 401 would degrade provenance to "unknown" silently,
        # which is precisely the failure this module exists to prevent.
        try:
            from src.llm.client import _resolve_api_key  # noqa: PLC0415

            api_key = _resolve_api_key()
        except Exception:  # noqa: BLE001 — metadata capture never blocks a run
            api_key = os.getenv("RAG_LLM_API_KEY", "").strip()

    info = _lmstudio_model_info(api_base, model, api_key) if api_base and model else {}
    return Provenance(
        model=model,
        quant=str(info.get("quantization") or ""),
        arch=str(info.get("arch") or ""),
        endpoint=api_base,
        source=classify_source(api_base),
    )


def model_fingerprint(prov: Optional[Provenance]) -> str:
    """The value stored alongside a judgment: ``model@quant`` (quant if known).

    Kept in the existing ``model`` column so old rows stay readable and a plain
    model id still compares equal to itself.
    """
    if prov is None:
        return ""
    if prov.quant:
        return f"{prov.model}@{prov.quant}"
    return prov.model
