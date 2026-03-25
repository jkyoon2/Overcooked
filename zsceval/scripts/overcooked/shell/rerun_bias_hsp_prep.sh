#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash zsceval/scripts/overcooked/shell/rerun_bias_hsp_prep.sh [options]

Reruns selected bias-agent seeds for one layout, then refreshes:
  1. bias checkpoint extraction
  2. bias eval template generation
  3. bias evaluation
  4. HSP-S2 yaml generation

Defaults are tailored for the current tto recovery workflow.

Options:
  --layout <name>               Default: tto
  --seeds <csv>                 Default: 2,3,5,6
  --env-name <name>             Default: Overcooked
  --wandb-name <name>           Default: $WANDB_NAME or juliejung98
  --hsp-k <n>                   Default: 6
  --mep-population-size <n>     Default: 5
  --hsp-population-size <n>     Default: 12
  --conda-sh <path>             Default: $HOME/miniconda3/etc/profile.d/conda.sh
  --conda-env <name>            Default: zsceval
  --no-conda-activate           Skip conda activation inside this script
  --dry-run                     Print commands without running them
  -h, --help                    Show this help

Examples:
  bash zsceval/scripts/overcooked/shell/rerun_bias_hsp_prep.sh

  bash zsceval/scripts/overcooked/shell/rerun_bias_hsp_prep.sh \
    --layout tto --seeds 2,3,5,6
EOF
}

split_csv() {
    local csv="$1"
    local -n out_ref="$2"
    local item
    out_ref=()
    IFS=',' read -r -a out_ref <<<"$csv"
    for item in "${!out_ref[@]}"; do
        out_ref[$item]="$(echo "${out_ref[$item]}" | xargs)"
    done
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERCOOKED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${OVERCOOKED_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"

layout="tto"
seeds_csv="2,3,5,6"
env_name="Overcooked"
wandb_name="${WANDB_NAME:-juliejung98}"
hsp_k=6
mep_population_size=5
hsp_population_size=12
conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
conda_env="zsceval"
activate_conda=1
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --layout)
            layout="${2:-}"
            shift 2
            ;;
        --seeds)
            seeds_csv="${2:-}"
            shift 2
            ;;
        --env-name)
            env_name="${2:-}"
            shift 2
            ;;
        --wandb-name)
            wandb_name="${2:-}"
            shift 2
            ;;
        --hsp-k)
            hsp_k="${2:-6}"
            shift 2
            ;;
        --mep-population-size)
            mep_population_size="${2:-5}"
            shift 2
            ;;
        --hsp-population-size)
            hsp_population_size="${2:-12}"
            shift 2
            ;;
        --conda-sh)
            conda_sh="${2:-}"
            shift 2
            ;;
        --conda-env)
            conda_env="${2:-}"
            shift 2
            ;;
        --no-conda-activate)
            activate_conda=0
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

split_csv "${seeds_csv}" rerun_seeds
if (( ${#rerun_seeds[@]} == 0 )); then
    echo "[ERROR] --seeds produced an empty seed list." >&2
    exit 1
fi

run_cmd() {
    echo "[$(date '+%F %T')] $*"
    if (( dry_run )); then
        return 0
    fi
    "$@"
}

cd "${OVERCOOKED_DIR}"

if (( activate_conda && ! dry_run )); then
    if [[ ! -f "${conda_sh}" ]]; then
        echo "[ERROR] Missing conda activation script: ${conda_sh}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${conda_sh}"
    conda activate "${conda_env}"
fi

export WANDB_NAME="${wandb_name}"
export PYTHONUNBUFFERED=1

echo "[INFO] Layout: ${layout}"
echo "[INFO] Rerun bias seeds: ${rerun_seeds[*]}"
echo "[INFO] HSP k=${hsp_k}, MEP population size=${mep_population_size}, HSP population size=${hsp_population_size}"
if (( dry_run )); then
    echo "[INFO] Dry-run mode enabled."
fi

for seed in "${rerun_seeds[@]}"; do
    run_cmd bash "${OVERCOOKED_DIR}/shell/train_bias_agents.sh" "${layout}" "${seed}" "${seed}"
done

run_cmd python "${SCRIPTS_DIR}/extract_models/extract_bias_agents_models.py" \
    "${layout}" "${env_name}" --wandb_name "${wandb_name}"

run_cmd python "${SCRIPTS_DIR}/prep/gen_bias_agent_eval_yml.py" "${layout}"

run_cmd bash "${OVERCOOKED_DIR}/shell/eval_bias_agents_events.sh" "${layout}"

if (( ! dry_run )); then
    usable_count="$(
        python - "${PROJECT_ROOT}" "${layout}" <<'PY'
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
layout = sys.argv[2]
root = project_root / "zsceval" / "scripts" / "overcooked" / "eval" / "results" / layout / "bias"
usable = 0
for path in sorted(root.glob("eval-hsp*_final.json")):
    data = json.loads(path.read_text())
    sparse_keys = [key for key in data if key.endswith("-eval_ep_sparse_r")]
    if sparse_keys and data[sparse_keys[0]] > 0.1:
        usable += 1
print(usable)
PY
    )"
    echo "[INFO] Usable bias policies after eval: ${usable_count}"
    if (( usable_count < hsp_k )); then
        echo "[ERROR] Need at least hsp-k=${hsp_k} usable bias policies before HSP yaml generation, got ${usable_count}." >&2
        echo "[ERROR] HSP-S2 yaml generation was skipped to avoid producing an invalid population config." >&2
        exit 1
    fi
fi

run_cmd python "${SCRIPTS_DIR}/prep/gen_hsp_S2_ymls.py" \
    -l "${layout}" -k "${hsp_k}" -s "${mep_population_size}" -S "${hsp_population_size}"

echo "[INFO] Bias refresh and HSP prep completed for layout ${layout}."
echo "[INFO] Next step: bash zsceval/scripts/overcooked/shell/train_hsp_stage_2.sh ${layout} ${hsp_population_size} 1 5"
