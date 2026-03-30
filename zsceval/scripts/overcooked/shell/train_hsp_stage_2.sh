#!/bin/bash
set -euo pipefail

env="Overcooked"
layout=${1:-}
population_size=${2:-}
seed_begin=${3:-1}
seed_max=${4:-5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERCOOKED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${OVERCOOKED_DIR}/../../.." && pwd)"
cd "${OVERCOOKED_DIR}"

if [[ -z "${layout}" || -z "${population_size}" ]]; then
    echo "Usage: bash zsceval/scripts/overcooked/shell/train_hsp_stage_2.sh <layout> <population_size:{12|24|36}> [seed_begin] [seed_max]"
    exit 1
fi

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

if [[ "${population_size}" == "12" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 2.5e7 5e7"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 4e7 5e7"
    fi
    reward_shaping_horizon="5e7"
    num_env_steps="5e7"
    final_checkpoint_step="50000000"
    pop="hsp"
    mep_exp="mep-S1-s5"
elif [[ "${population_size}" == "24" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 4e7 8e7"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 6.4e7 8e7"
    fi
    reward_shaping_horizon="8e7"
    num_env_steps="8e7"
    final_checkpoint_step="80000000"
    pop="hsp"
    mep_exp="mep-S1-s10"
elif [[ "${population_size}" == "36" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 5e7 1e8"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 8e7 1e8"
    fi
    reward_shaping_horizon="1e8"
    num_env_steps="1e8"
    final_checkpoint_step="100000000"
    pop="hsp"
    mep_exp="mep-S1-s15"
fi

num_agents=2
algo="adaptive"
exp="hsp-S2-s${population_size}"
stage="S2"
policy_pool_root="${PROJECT_ROOT}/zsceval/scripts/policy_pool"
policy_mode="shared"
role_csv="individual,individual"
agent_policy_names="ppo ppo"
results_root="${PROJECT_ROOT}/results"

export POLICY_POOL="${policy_pool_root}"

n_training_threads=100
n_render_rollout_threads=1
render_episodes=1
wandb_name="${WANDB_NAME:-juliejung98}"
log_dir="${OVERCOOKED_DIR}/log"
mkdir -p "${log_dir}"

ulimit -n 65536

echo "env=${env}, layout=${layout}, algo=${algo}, pop=${pop}, exp=${exp}, stage=${stage}, mode=${policy_mode}, roles=${role_csv}, seeds=${seed_begin}..${seed_max}"
for seed in $(seq "${seed_begin}" "${seed_max}");
do
    yml_path="${POLICY_POOL}/${layout}/hsp/s2/train-s${population_size}-${pop}_${mep_exp}-${seed}.yml"
    if [[ ! -f "${yml_path}" ]]; then
        echo "[ERROR] Missing population yaml for seed ${seed}: ${yml_path}"
        exit 1
    fi

    run_dir="${results_root}/${env}/${layout}/${policy_mode}/${algo}/${exp}/seed${seed}"
    final_actor="${run_dir}/models/hsp_adaptive/actor_periodic_${final_checkpoint_step}.pt"
    if [[ -f "${final_actor}" ]]; then
        echo "seed=${seed}, final checkpoint exists at ${final_actor}; skipping"
        continue
    fi

    ts=$(date +%Y%m%d_%H%M%S)
    run_log="${log_dir}/train_hsp_stage2_${layout}_s${population_size}_seed${seed}_${ts}.log"
    echo "seed=${seed}, run_dir=${run_dir}"
    echo "log=${run_log}"

    python train/train_adaptive.py --env_name "${env}" --algorithm_name "${algo}" --experiment_name "${exp}" --layout_name "${layout}" --num_agents "${num_agents}" \
    --seed "${seed}" --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps "${num_env_steps}" --reward_shaping_horizon "${reward_shaping_horizon}" \
    --overcooked_version "${version}" \
    --n_rollout_threads "${n_training_threads}" --dummy_batch_size 1 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --stage 2 \
    --policy_mode "${policy_mode}" --agent_policy_names ${agent_policy_names} \
    --reward_shaping_role individual --reward_shaping_roles "${role_csv}" \
    --save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads "$((population_size * 2))" --eval_episodes 5 \
    --population_yaml_path "${yml_path}" \
    --population_size "${population_size}" --adaptive_agent_name hsp_adaptive --use_agent_policy_id \
    --use_render --save_gifs --n_render_rollout_threads "${n_render_rollout_threads}" --render_episodes "${render_episodes}" \
    --use_proper_time_limits \
    --wandb_name "${wandb_name}" 2>&1 | tee "${run_log}"
done
