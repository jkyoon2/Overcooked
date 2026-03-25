#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash zsceval/scripts/overcooked/shell/run_full_hsp_pipeline.sh [options]

Runs the full HSP-S2 preparation/training pipeline for one or more layouts:
  1. bias agent training
  2. bias agent extract + evaluate
  3. MEP S1 training
  4. MEP S1 extract
  5. HSP S2 yaml generation
  6. HSP S2 training

Default layouts are: ttt,tto,too,ooo

Options:
  --layouts <csv|all>            Layout list. Default: all
  --skip-layouts <csv>           Explicitly skip layouts.
  --gpus <csv|auto>              GPU list, e.g. 0,1. Default: auto
  --parallel <n>                 Max concurrent layout pipelines. Default: number of selected GPUs
  --skip-active-layouts          Skip layouts with active train processes. Default: enabled
  --no-skip-active-layouts       Disable active-layout skipping
  --free-gpu-mem-max-mb <mb>     Max memory used for an auto-selected idle GPU. Default: 512
  --bias-seed-begin <n>          Default: 1
  --bias-seed-end <n>            Default: 6
  --mep-population-size <n>      Default: 5
  --mep-seed-begin <n>           Default: 1
  --mep-seed-end <n>             Default: 1
  --mep-extract-seed <n>         Default: 1
  --hsp-k <n>                    Number of selected bias agents. Default: 6
  --hsp-population-size <n>      Default: 12
  --hsp-seed-begin <n>           Default: 1
  --hsp-seed-end <n>             Default: 5
  --from-stage <name>            One of: bias_train,bias_extract_eval,mep_train,mep_extract,hsp_gen,hsp_train
  --to-stage <name>              One of: bias_train,bias_extract_eval,mep_train,mep_extract,hsp_gen,hsp_train
  --wandb-name <name>            Default: $WANDB_NAME or juliejung98
  --conda-sh <path>              Default: $HOME/miniconda3/etc/profile.d/conda.sh
  --conda-env <name>             Default: zsceval
  --log-root <path>              Default: zsceval/scripts/overcooked/log/pipeline
  --poll-seconds <n>             Scheduler poll interval. Default: 10
  --dry-run                      Print commands without running them
  -h, --help                     Show this help

Examples:
  bash zsceval/scripts/overcooked/shell/run_full_hsp_pipeline.sh

  bash zsceval/scripts/overcooked/shell/run_full_hsp_pipeline.sh --gpus 1

  bash zsceval/scripts/overcooked/shell/run_full_hsp_pipeline.sh \
    --layouts too \
    --from-stage hsp_train --to-stage hsp_train \
    --hsp-seed-begin 5 --hsp-seed-end 5 \
    --gpus 1
EOF
}

stage_rank() {
    case "$1" in
        bias_train) echo 1 ;;
        bias_extract_eval) echo 2 ;;
        mep_train) echo 3 ;;
        mep_extract) echo 4 ;;
        hsp_gen) echo 5 ;;
        hsp_train) echo 6 ;;
        *)
            echo "[ERROR] Unknown stage: $1" >&2
            return 1
            ;;
    esac
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

contains_item() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERCOOKED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${OVERCOOKED_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"

layouts_csv="all"
skip_layouts_csv=""
gpus_csv="auto"
parallel=0
skip_active_layouts=1
free_gpu_mem_max_mb=512

bias_seed_begin=1
bias_seed_end=6

mep_population_size=5
mep_seed_begin=1
mep_seed_end=1
mep_extract_seed=1

hsp_k=6
hsp_population_size=12
hsp_seed_begin=1
hsp_seed_end=5

from_stage="bias_train"
to_stage="hsp_train"

env_name="Overcooked"
wandb_name="${WANDB_NAME:-juliejung98}"
conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
conda_env="zsceval"
log_root="${OVERCOOKED_DIR}/log/pipeline"
poll_seconds=10
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --layouts)
            layouts_csv="${2:-}"
            shift 2
            ;;
        --skip-layouts)
            skip_layouts_csv="${2:-}"
            shift 2
            ;;
        --gpus)
            gpus_csv="${2:-}"
            shift 2
            ;;
        --parallel)
            parallel="${2:-0}"
            shift 2
            ;;
        --skip-active-layouts)
            skip_active_layouts=1
            shift
            ;;
        --no-skip-active-layouts)
            skip_active_layouts=0
            shift
            ;;
        --free-gpu-mem-max-mb)
            free_gpu_mem_max_mb="${2:-512}"
            shift 2
            ;;
        --bias-seed-begin)
            bias_seed_begin="${2:-1}"
            shift 2
            ;;
        --bias-seed-end)
            bias_seed_end="${2:-6}"
            shift 2
            ;;
        --mep-population-size)
            mep_population_size="${2:-5}"
            shift 2
            ;;
        --mep-seed-begin)
            mep_seed_begin="${2:-1}"
            shift 2
            ;;
        --mep-seed-end)
            mep_seed_end="${2:-1}"
            shift 2
            ;;
        --mep-extract-seed)
            mep_extract_seed="${2:-1}"
            shift 2
            ;;
        --hsp-k)
            hsp_k="${2:-6}"
            shift 2
            ;;
        --hsp-population-size)
            hsp_population_size="${2:-12}"
            shift 2
            ;;
        --hsp-seed-begin)
            hsp_seed_begin="${2:-1}"
            shift 2
            ;;
        --hsp-seed-end)
            hsp_seed_end="${2:-5}"
            shift 2
            ;;
        --from-stage)
            from_stage="${2:-bias_train}"
            shift 2
            ;;
        --to-stage)
            to_stage="${2:-hsp_train}"
            shift 2
            ;;
        --wandb-name)
            wandb_name="${2:-${wandb_name}}"
            shift 2
            ;;
        --conda-sh)
            conda_sh="${2:-${conda_sh}}"
            shift 2
            ;;
        --conda-env)
            conda_env="${2:-${conda_env}}"
            shift 2
            ;;
        --log-root)
            log_root="${2:-${log_root}}"
            shift 2
            ;;
        --poll-seconds)
            poll_seconds="${2:-10}"
            shift 2
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

from_stage_rank="$(stage_rank "${from_stage}")"
to_stage_rank="$(stage_rank "${to_stage}")"
if (( from_stage_rank > to_stage_rank )); then
    echo "[ERROR] --from-stage must come before or equal to --to-stage" >&2
    exit 1
fi

if (( bias_seed_begin > bias_seed_end )); then
    echo "[ERROR] Invalid bias seed range: ${bias_seed_begin}..${bias_seed_end}" >&2
    exit 1
fi
if (( mep_seed_begin > mep_seed_end )); then
    echo "[ERROR] Invalid MEP seed range: ${mep_seed_begin}..${mep_seed_end}" >&2
    exit 1
fi
if (( hsp_seed_begin > hsp_seed_end )); then
    echo "[ERROR] Invalid HSP seed range: ${hsp_seed_begin}..${hsp_seed_end}" >&2
    exit 1
fi
if (( (hsp_population_size - hsp_k) % 3 != 0 )); then
    echo "[ERROR] (hsp_population_size - hsp_k) must be divisible by 3: S=${hsp_population_size}, k=${hsp_k}" >&2
    exit 1
fi
if (( bias_seed_end - bias_seed_begin + 1 < hsp_k )); then
    echo "[ERROR] Need at least hsp_k=${hsp_k} bias seeds in the configured bias seed range." >&2
    exit 1
fi

all_layouts=(ttt tto too ooo)
if [[ "${layouts_csv}" == "all" ]]; then
    layouts=("${all_layouts[@]}")
else
    split_csv "${layouts_csv}" layouts
fi

if [[ -n "${skip_layouts_csv}" ]]; then
    split_csv "${skip_layouts_csv}" skip_layouts
else
    skip_layouts=()
fi

filtered_layouts=()
for layout in "${layouts[@]}"; do
    if ! contains_item "${layout}" "${all_layouts[@]}"; then
        echo "[ERROR] Unsupported layout: ${layout}" >&2
        exit 1
    fi
    if contains_item "${layout}" "${skip_layouts[@]}"; then
        echo "[INFO] Skipping layout ${layout} because it is listed in --skip-layouts."
        continue
    fi
    filtered_layouts+=("${layout}")
done
layouts=("${filtered_layouts[@]}")

is_layout_active() {
    local layout="$1"
    ps -eo args= | rg -q \
        "(train_bias_agents\\.sh|train_mep_stage_1\\.sh|train_hsp_stage_2\\.sh).*(^|[[:space:]])${layout}([[:space:]]|$)|Overcooked_${layout}-(hsp-S2|mep-S1|hsp-S1)"
}

if (( skip_active_layouts )); then
    filtered_layouts=()
    for layout in "${layouts[@]}"; do
        if is_layout_active "${layout}"; then
            echo "[INFO] Skipping layout ${layout} because an active train process was detected."
            continue
        fi
        filtered_layouts+=("${layout}")
    done
    layouts=("${filtered_layouts[@]}")
fi

if (( ${#layouts[@]} == 0 )); then
    echo "[INFO] No layouts left to schedule."
    exit 0
fi

detect_auto_gpus() {
    local -n out_ref="$1"
    local line idx mem
    out_ref=()
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[ERROR] nvidia-smi not found but --gpus auto was requested." >&2
        return 1
    fi
    while IFS=',' read -r idx mem; do
        idx="$(echo "${idx}" | xargs)"
        mem="$(echo "${mem}" | xargs)"
        if [[ -n "${idx}" && -n "${mem}" ]] && (( mem <= free_gpu_mem_max_mb )); then
            out_ref+=("${idx}")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}

if [[ "${gpus_csv}" == "auto" ]]; then
    detect_auto_gpus gpus
else
    split_csv "${gpus_csv}" gpus
fi

if (( ${#gpus[@]} == 0 )); then
    echo "[ERROR] No available GPUs were selected." >&2
    exit 1
fi

if (( parallel <= 0 || parallel > ${#gpus[@]} )); then
    parallel="${#gpus[@]}"
fi
slot_gpus=("${gpus[@]:0:${parallel}}")

mkdir -p "${log_root}"
mkdir -p "${log_root}/locks"

should_run_stage() {
    local stage_name="$1"
    local stage_id
    stage_id="$(stage_rank "${stage_name}")"
    (( stage_id >= from_stage_rank && stage_id <= to_stage_rank ))
}

run_layout_pipeline() {
    local layout="$1"
    local gpu="$2"
    local lock_dir="${log_root}/locks/${layout}.lock"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local pipeline_log="${log_root}/${layout}_gpu${gpu}_${ts}.log"

    if ! mkdir "${lock_dir}" 2>/dev/null; then
        echo "[WARN] Layout ${layout} is already locked; skipping duplicate launch." | tee -a "${pipeline_log}"
        return 0
    fi

    (
        set -euo pipefail
        trap 'rm -rf "${lock_dir}"' EXIT

        run_cmd() {
            echo "[$(date '+%F %T')] [${layout}] [gpu${gpu}] $*"
            if (( dry_run )); then
                return 0
            fi
            "$@"
        }

        if (( ! dry_run )); then
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
        export CUDA_VISIBLE_DEVICES="${gpu}"

        echo "[$(date '+%F %T')] [${layout}] [gpu${gpu}] pipeline start"
        echo "[$(date '+%F %T')] [${layout}] [gpu${gpu}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

        if should_run_stage bias_train; then
            run_cmd bash "${OVERCOOKED_DIR}/shell/train_bias_agents.sh" \
                "${layout}" "${bias_seed_begin}" "${bias_seed_end}"
        fi

        if should_run_stage bias_extract_eval; then
            run_cmd python "${SCRIPTS_DIR}/extract_models/extract_bias_agents_models.py" \
                "${layout}" "${env_name}" --wandb_name "${wandb_name}"
            run_cmd bash "${OVERCOOKED_DIR}/shell/eval_bias_agents_events.sh" "${layout}"
        fi

        if should_run_stage mep_train; then
            run_cmd bash "${OVERCOOKED_DIR}/shell/train_mep_stage_1.sh" \
                "${layout}" "${mep_population_size}" "${mep_seed_begin}" "${mep_seed_end}"
        fi

        if should_run_stage mep_extract; then
            run_cmd python "${SCRIPTS_DIR}/extract_models/extract_pop_S1_models.py" \
                "${layout}" "${env_name}" --algo mep \
                --population_size "${mep_population_size}" \
                --experiment_name "mep-S1-s${mep_population_size}" \
                --seed "${mep_extract_seed}"
        fi

        if should_run_stage hsp_gen; then
            run_cmd python "${SCRIPTS_DIR}/prep/gen_hsp_S2_ymls.py" \
                -l "${layout}" -k "${hsp_k}" -s "${mep_population_size}" -S "${hsp_population_size}"
        fi

        if should_run_stage hsp_train; then
            run_cmd bash "${OVERCOOKED_DIR}/shell/train_hsp_stage_2.sh" \
                "${layout}" "${hsp_population_size}" "${hsp_seed_begin}" "${hsp_seed_end}"
        fi

        echo "[$(date '+%F %T')] [${layout}] [gpu${gpu}] pipeline done"
    ) 2>&1 | tee "${pipeline_log}"
}

echo "[INFO] Layouts to run: ${layouts[*]}"
echo "[INFO] GPU slots: ${slot_gpus[*]}"
echo "[INFO] Stage window: ${from_stage} -> ${to_stage}"
echo "[INFO] Logs: ${log_root}"
if (( dry_run )); then
    echo "[INFO] Dry-run mode enabled."
fi

pending_layouts=("${layouts[@]}")
free_gpus=("${slot_gpus[@]}")
running_pids=()
failures=()

declare -A pid_to_gpu
declare -A pid_to_layout

launch_next() {
    local layout="$1"
    local gpu="$2"
    echo "[INFO] Launching layout ${layout} on GPU ${gpu}"
    run_layout_pipeline "${layout}" "${gpu}" &
    local pid=$!
    running_pids+=("${pid}")
    pid_to_gpu["${pid}"]="${gpu}"
    pid_to_layout["${pid}"]="${layout}"
}

while (( ${#pending_layouts[@]} > 0 || ${#running_pids[@]} > 0 )); do
    while (( ${#pending_layouts[@]} > 0 && ${#free_gpus[@]} > 0 )); do
        next_layout="${pending_layouts[0]}"
        pending_layouts=("${pending_layouts[@]:1}")
        next_gpu="${free_gpus[0]}"
        free_gpus=("${free_gpus[@]:1}")
        launch_next "${next_layout}" "${next_gpu}"
    done

    if (( ${#running_pids[@]} == 0 )); then
        break
    fi

    sleep "${poll_seconds}"
    new_running_pids=()
    for pid in "${running_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            new_running_pids+=("${pid}")
            continue
        fi

        status=0
        if wait "${pid}"; then
            status=0
        else
            status=$?
        fi

        finished_layout="${pid_to_layout[${pid}]}"
        finished_gpu="${pid_to_gpu[${pid}]}"
        free_gpus+=("${finished_gpu}")
        unset pid_to_layout["${pid}"]
        unset pid_to_gpu["${pid}"]

        if (( status == 0 )); then
            echo "[INFO] Layout ${finished_layout} finished successfully on GPU ${finished_gpu}"
        else
            echo "[ERROR] Layout ${finished_layout} failed on GPU ${finished_gpu} with status ${status}" >&2
            failures+=("${finished_layout}@gpu${finished_gpu}")
        fi
    done
    running_pids=("${new_running_pids[@]}")
done

if (( ${#failures[@]} > 0 )); then
    echo "[ERROR] Pipeline finished with failures: ${failures[*]}" >&2
    exit 1
fi

echo "[INFO] Pipeline finished successfully."
