#!/usr/bin/env bash
# Start Explain This Repo: sets up Python on the first run, then opens http://127.0.0.1:8000
#   ./run.sh               start
#   ./run.sh --rebuild-ui  rebuild the React frontend first (needs Node.js)
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then set -a; . ./.env; set +a; fi

PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 "$HOME/miniforge3/bin/python3" python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Explain This Repo needs Python 3.10 or newer. On a Mac: brew install python@3.12" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating a virtual environment with $($PY --version)..."
  "$PY" -m venv .venv
fi
if [ ! -f .venv/.installed ] || [ backend/requirements.txt -nt .venv/.installed ]; then
  echo "Installing Python packages (first run only)..."
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r backend/requirements.txt
  touch .venv/.installed
fi

if [ "${1:-}" = "--rebuild-ui" ] || [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "Building the web interface..."
    (cd frontend && npm install --silent --no-audit --no-fund && npm run build --silent)
  else
    echo "frontend/dist is missing and Node.js isn't installed. Install Node (brew install node) and run ./run.sh again." >&2
    exit 1
  fi
fi

PORT="${ETR_PORT:-8000}"
URL="http://127.0.0.1:$PORT"
echo "Explain This Repo is running at $URL  (Ctrl+C to stop)"
( sleep 2; command -v open >/dev/null 2>&1 && open "$URL" ) >/dev/null 2>&1 &
exec .venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$PORT" --log-level warning
