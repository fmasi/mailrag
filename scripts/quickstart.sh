#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Starting Qdrant (docker compose)…"
docker compose up -d qdrant
echo "Waiting for Qdrant on :6333…"
until curl -sf http://localhost:6333/readyz >/dev/null 2>&1; do sleep 1; done
echo "Running the thread-aware demo…"
python main.py
