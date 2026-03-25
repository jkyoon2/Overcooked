#!/bin/bash
set -euo pipefail

# Kill HSP stage-2 training processes:
# - train_hsp_stage_2.sh launcher
# - train_adaptive.py runs with experiment_name hsp-S2-s*
# - corresponding worker processes (setproctitle)

layout=${1:-}
population_size=${2:-}

echo "Killing HSP stage-2 processes (layout='${layout:-ALL}', pop='${population_size:-ALL}')" >&2

patterns=(
  "train_hsp_stage_2.sh"
  "train/train_adaptive.py.*hsp-S2-s"
  "adaptive-Overcooked.*-hsp-S2-s"
)

if [[ -n "${layout}" ]]; then
  patterns+=(
    "train_hsp_stage_2.sh.*${layout}"
    "train/train_adaptive.py.*${layout}.*hsp-S2-s"
    "adaptive-Overcooked.*${layout}.*-hsp-S2-s"
  )
fi

if [[ -n "${population_size}" ]]; then
  patterns+=(
    "train/train_adaptive.py.*hsp-S2-s${population_size}"
    "adaptive-Overcooked.*-hsp-S2-s${population_size}"
  )
fi

all_pids=()
for pat in "${patterns[@]}"; do
  while IFS= read -r pid; do
    all_pids+=("$pid")
  done < <(pgrep -f "$pat" || true)
done

readarray -t pids < <(printf "%s\n" "${all_pids[@]:-}" | sed '/^$/d' | sort -u)

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No matching HSP stage-2 processes found." >&2
  exit 0
fi

echo "Found PIDs: ${pids[*]}" >&2

# First terminate process groups.
for pid in "${pids[@]}"; do
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ' || true)
  [[ -n "${pgid}" ]] || continue
  kill -TERM "-${pgid}" 2>/dev/null || true
done

sleep 3

# Force kill leftovers.
for pid in "${pids[@]}"; do
  if ps -p "$pid" >/dev/null 2>&1; then
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ' || true)
    [[ -n "${pgid}" ]] || continue
    kill -KILL "-${pgid}" 2>/dev/null || true
  fi
done

echo "Done." >&2

