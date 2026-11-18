#!/usr/bin/env bash
# Start the PathSnap FastAPI backend, reachable from the evaluator laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn api.app:app --host 0.0.0.0 --port "${PATHSNAP_API_PORT:-8000}"
