#!/usr/bin/env bash
# Start the unsupervised defect-detection web server on Linux / macOS.
# Override the interpreter with: PY=/path/to/python ./start_web.sh
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://${HOST}:${PORT}/"

if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "${URL}health" >/dev/null 2>&1; then
  echo "[INFO] Service is already running: ${URL}"
  exit 0
fi

echo "Starting service: ${URL}"
exec "${PY}" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}"
