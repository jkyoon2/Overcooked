#!/usr/bin/env bash
# Run the human_exp backend (uvicorn) + frontend (Vite) together.
#   bash zsceval/human_exp/run-dev.sh
#   BACKEND_PORT=8002 FRONTEND_PORT=5174 bash zsceval/human_exp/run-dev.sh
# Ctrl+C tears both processes down.

set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
CONDA_ENV="${CONDA_ENV:-neurocontroller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- conda activation ---------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "${CONDA_BASE:-}" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniconda3"
fi
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "ERROR: could not locate conda.sh under '$CONDA_BASE'." >&2
  echo "       Set CONDA_BASE or install miniconda before running." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- port pre-flight ----------------------------------------------------------
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\.)${port}$"
  else
    netstat -tln 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\.)${port}$"
  fi
}

show_port_owner() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print}'
  fi
}

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if port_in_use "$port"; then
    echo "ERROR: port $port is already in use:" >&2
    show_port_owner "$port" >&2
    echo "       Pick a different port, e.g.: BACKEND_PORT=8002 FRONTEND_PORT=5174 $0" >&2
    exit 1
  fi
done

# --- backend (background) -----------------------------------------------------
echo "[run-dev] starting uvicorn on http://localhost:${BACKEND_PORT}"
uvicorn backend.main:app --reload --port "$BACKEND_PORT" --log-level info &
BACKEND_PID=$!

# Sweep every descendant of this script — handles uvicorn's reloader child,
# npm → node → vite, and any signal delivered only to the script PID (e.g.
# `kill -INT <pid>`) rather than the whole foreground process group.
sweep_descendants() {
  local sig="$1"
  local parent="$2"
  local children
  children=$(pgrep -P "$parent" 2>/dev/null || true)
  for child in $children; do
    sweep_descendants "$sig" "$child"
    kill "-$sig" "$child" 2>/dev/null || true
  done
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  echo "[run-dev] shutting down…"
  sweep_descendants TERM "$$"
  # Brief grace period before SIGKILL on anything still alive.
  sleep 1
  sweep_descendants KILL "$$"
  exit "$rc"
}
trap cleanup EXIT INT TERM

# --- wait for backend to answer /health ---------------------------------------
echo -n "[run-dev] waiting for backend "
for _ in $(seq 1 60); do
  if curl -fs "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo "ready."
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo
    echo "ERROR: uvicorn exited before answering /health." >&2
    exit 1
  fi
  echo -n "."
  sleep 0.3
done
if ! curl -fs "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
  echo
  echo "ERROR: backend did not become healthy within 18s." >&2
  exit 1
fi

# --- frontend (foreground) ----------------------------------------------------
echo "[run-dev] starting vite on http://localhost:${FRONTEND_PORT}"
cd frontend
VITE_BACKEND_PORT="$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT" --strictPort
