#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt
mkdir -p data/private

if [[ ! -f "data/private/holdings_snapshot.csv" ]]; then
  echo "No private holdings CSV found."
  echo "Copy your personal CSV to data/private/holdings_snapshot.csv or upload it in the dashboard."
fi

exec python -m streamlit run dashboard.py --server.address 127.0.0.1
