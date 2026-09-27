#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
WORKERS="${WORKERS:-4}"
LAYOUTS="${LAYOUTS:-2_forced_hard 2_forced_hard_4}"
SEEDS="${SEEDS:-}"

LOG_DIR="${REPO_ROOT}/json_trajectory/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/requested_level_${TIMESTAMP}.log"

echo "Starting trajectory generation at $(date)" | tee "${LOG_FILE}"
echo "LAYOUTS=${LAYOUTS}" | tee -a "${LOG_FILE}"
echo "WORKERS=${WORKERS}" | tee -a "${LOG_FILE}"
echo "SEEDS=${SEEDS:-layout defaults}" | tee -a "${LOG_FILE}"
echo "LOG=${LOG_FILE}" | tee -a "${LOG_FILE}"

ARGS=(
    --layouts ${LAYOUTS}
    --device "${DEVICE}"
    --workers "${WORKERS}"
)
if [[ -n "${SEEDS}" ]]; then
    ARGS+=(--seeds ${SEEDS} --keep_out_of_scope)
fi

${PYTHON_BIN} -m zsceval.analysis.json.generate_requested_level_trajectories "${ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "Done at $(date)" | tee -a "${LOG_FILE}"
