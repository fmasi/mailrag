"""Single seam for constructing a Qdrant client from explicit args + QDRANT_* env.

The one place that turns ``(url, api_key, prefer_grpc)`` — explicit args overriding
the ``QDRANT_*`` environment — into a ``QdrantClient``. The former call sites
(``query/hybrid``, ``storage/persist``, ``ingest/hybrid_qdrant``) delegate here so
connection logic, env-var names and the grpc flag cannot drift apart (issue #45; the
``QDRANT_URL`` host gotcha of issue #29 is exactly the class of bug this prevents).

Deliberately lean: only stdlib plus a lazy ``qdrant_client`` import, so the
lightweight offline query env can import it without pulling heavy llama-index
integrations (see the note in ``src/query/hybrid.py``).
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def get_qdrant_client(url=None, api_key=None, prefer_grpc=None):
    """Construct a ``QdrantClient``.

    Precedence (explicit arg > environment > default):

      * ``url``         — arg > ``$QDRANT_URL`` > ``ValueError``. Whitespace stripped.
      * ``api_key``     — arg > ``$QDRANT_API_KEY`` > ``None`` (empty string -> ``None``).
      * ``prefer_grpc`` — arg (bool) > ``$QDRANT_PREFER_GRPC`` (truthy set) > ``False``.

    The ``url`` arg is the explicit override used by the host CLI to target
    ``localhost`` without inheriting a container-oriented ``QDRANT_URL`` from ``.env``.
    """
    from qdrant_client import QdrantClient

    url = (url or os.environ.get("QDRANT_URL") or "").strip()
    if not url:
        raise ValueError("QDRANT_URL environment variable is not set")

    if api_key is None:
        api_key = os.environ.get("QDRANT_API_KEY")
    api_key = (api_key or "").strip() or None

    if prefer_grpc is None:
        prefer_grpc = (os.environ.get("QDRANT_PREFER_GRPC") or "").strip().lower() in _TRUTHY

    return QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
