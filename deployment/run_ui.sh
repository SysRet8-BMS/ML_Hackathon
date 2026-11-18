#!/usr/bin/env bash
# Start the PathSnap Streamlit evaluator page.
# Set PATHSNAP_API_URL to the backend laptop, e.g. http://192.168.1.10:8000
set -euo pipefail
cd "$(dirname "$0")/.."
exec streamlit run ui/app.py
