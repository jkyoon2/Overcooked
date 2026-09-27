#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

source /home/juliecandoit98/miniconda3/etc/profile.d/conda.sh
conda activate neurocontroller

LAYOUT="1_forced_easy"
CONFIG="${CONFIG:-zsceval/scripts/overcooked/config/diffusion_policy_1_forced_easy_split3_1_1_long.yaml}"
SEED="${SEED:-1}"
EXP="${EXP:-diffusion_1_forced_easy_nobs4_split3_1_1_long}"
LOG_DIR="${LOG_DIR:-logs/diffusion_policy/${LAYOUT}/${EXP}/seed${SEED}}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-Overcooked-Diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
RESUME_FROM_PREVIOUS="${RESUME_FROM_PREVIOUS:-0}"
RESUME_EXP="${RESUME_EXP:-diffusion_1_forced_easy_nobs4}"

mkdir -p "${LOG_DIR}"

if [[ "${USE_WANDB}" == "1" && "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    if ! python - <<'PY' >/dev/null 2>&1
import sys
import wandb

try:
    api = wandb.Api()
    has_key = bool(getattr(api, "api_key", None))
except Exception:
    has_key = False

sys.exit(0 if has_key else 1)
PY
    then
        echo "W&B is not logged in. Run: conda run -n neurocontroller wandb login --relogin --verify"
        exit 1
    fi
fi

launch_agent() {
    local agent_id="$1"
    local gpu_id="$2"
    local log_file="${LOG_DIR}/agent${agent_id}_gpu${gpu_id}.log"
    local pid_file="${LOG_DIR}/agent${agent_id}_gpu${gpu_id}.pid"

    local wandb_args=()
    if [[ "${USE_WANDB}" == "1" ]]; then
        wandb_args+=(--use-wandb)
        wandb_args+=(--wandb-project "${WANDB_PROJECT}")
        wandb_args+=(--wandb-name "${LAYOUT}_agent${agent_id}_${EXP}_seed${SEED}")
        if [[ -n "${WANDB_ENTITY}" ]]; then
            wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
        fi
    fi

    local resume_args=()
    if [[ "${RESUME_FROM_PREVIOUS}" == "1" ]]; then
        local resume_path="results/Overcooked/${LAYOUT}/diffusion_policy/${RESUME_EXP}/agent${agent_id}/seed${SEED}/policy.pt"
        if [[ -f "${resume_path}" ]]; then
            resume_args+=(--resume-from "${resume_path}")
        else
            echo "resume checkpoint not found for agent${agent_id}: ${resume_path}; starting from scratch"
        fi
    fi

    nohup setsid env CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE}" python zsceval/scripts/overcooked/train/train_diffusion_policy.py \
        --config "${CONFIG}" \
        --data-root trajectories \
        --layout "${LAYOUT}" \
        --agent-id "${agent_id}" \
        --seed "${SEED}" \
        --experiment-name "${EXP}" \
        "${resume_args[@]}" \
        "${wandb_args[@]}" \
        > "${log_file}" 2>&1 < /dev/null &

    echo "$!" > "${pid_file}"
    echo "agent${agent_id} gpu${gpu_id} pid=$(cat "${pid_file}") log=${log_file}"
}

launch_agent 0 0
launch_agent 1 1
