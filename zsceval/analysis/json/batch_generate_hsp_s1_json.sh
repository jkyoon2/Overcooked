#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
DEVICE="${DEVICE:-cpu}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

LAYOUTS=(
    "1_incentivized_hard"
    "1_forced_hard"
    "2_incentivized_hard"
)

SEEDS=(1 2)

CHECKPOINTS=(
    2520000
    3020000
    3520000
    4020000
    4520000
    5020000
)

total=$(( ${#LAYOUTS[@]} * ${#SEEDS[@]} * ${#CHECKPOINTS[@]} ))
count=0

for layout in "${LAYOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for checkpoint in "${CHECKPOINTS[@]}"; do
            count=$((count + 1))
            output_path="${REPO_ROOT}/json_trajectory/${layout}_${seed}_${checkpoint}.json"

            if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
                echo "[${count}/${total}] skip existing ${output_path}"
                continue
            fi

            echo "[${count}/${total}] layout=${layout} seed=${seed} checkpoint=${checkpoint}"
            "${PYTHON_BIN}" -m zsceval.analysis.json.generate_realtime_state_json \
                --layout_name "${layout}" \
                --algorithm_name rmappo \
                --experiment_name hsp-S1 \
                --policy_seed "${seed}" \
                --checkpoint "${checkpoint}" \
                --policy_type shared \
                --results_root "${RESULTS_ROOT}" \
                --device "${DEVICE}"
        done
    done
done
