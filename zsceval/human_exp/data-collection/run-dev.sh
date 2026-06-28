#!/usr/bin/env bash
# Run data-collection backend (Flask-SocketIO) + frontend (Vite) together.
#   bash zsceval/human_exp/data-collection/run-dev.sh
# Ctrl+C tears both processes down.

set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-5050}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"
CONDA_ENV="${CONDA_ENV:-neurocontroller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

# conda activation
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "${CONDA_BASE:-}" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniconda3"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cleanup() {
  trap - EXIT INT TERM
  echo "[run-dev] shutting down…"
  # kill entire process group
  kill 0 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Backend — run from backend/ dir; PYTHONPATH includes repo root for zsceval.*
echo "[run-dev] starting backend on http://localhost:${BACKEND_PORT}"
cd "$BACKEND_DIR"
PYTHONPATH="$REPO_ROOT:$BACKEND_DIR" python app.py "$BACKEND_PORT" &

# Wait for backend health
echo -n "[run-dev] waiting for backend"
for _ in $(seq 1 40); do
  if curl -fs "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo " ready."
    break
  fi
  echo -n "."
  sleep 0.5
done
if ! curl -fs "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
  echo; echo "ERROR: backend did not start." >&2; exit 1
fi

# Frontend
echo "[run-dev] starting frontend on http://localhost:${FRONTEND_PORT}"
cd "$SCRIPT_DIR/frontend"
VITE_BACKEND_PORT="$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT" --strictPort
