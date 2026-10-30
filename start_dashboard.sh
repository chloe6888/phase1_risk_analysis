#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -r requirements-dashboard.txt
fi

mkdir -p data/private
if [[ ! -f "data/private/holdings_snapshot.csv" ]]; then
  echo "No private holdings CSV found."
  echo "Upload your private holdings CSV in the dashboard when ready."
fi

exec "$PYTHON_BIN" -m streamlit run dashboard.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --server.headless true
