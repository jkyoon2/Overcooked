#!/bin/bash
set -euo pipefail

env="Overcooked"
layout=${1:-}
population_size=${2:-}
seed_begin=${3:-1}
seed_max=${4:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERCOOKED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${OVERCOOKED_DIR}/../../.." && pwd)"
cd "${OVERCOOKED_DIR}"

if [[ -z "${layout}" || -z "${population_size}" ]]; then
    echo "Usage: bash zsceval/scripts/overcooked/shell/train_mep_stage_1.sh <layout> <population_size:{5|10|15}> [seed_begin] [seed_max]"
    exit 1
fi

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 6e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps="1e7"
num_agents=2
algo="mep"
stage="S1"
exp="mep-${stage}-s${population_size}"
policy_mode="shared"
agent_policy_names="ppo ppo"
role_csv="individual,individual"
train_batch=125
eval_threads=$((population_size * 2))
n_render_rollout_threads=1
render_episodes=1
wandb_name="${WANDB_NAME:-juliejung98}"

policy_pool_root="${PROJECT_ROOT}/zsceval/scripts/policy_pool"
population_yaml="${policy_pool_root}/${layout}/mep/s1/train-s${population_size}.yml"
results_root="${PROJECT_ROOT}/results"
log_dir="${OVERCOOKED_DIR}/log"
mkdir -p "${log_dir}"

export POLICY_POOL="${policy_pool_root}"

if [[ ! -f "${population_yaml}" ]]; then
    python "${PROJECT_ROOT}/zsceval/scripts/prep/gen_pop_ymls.py" "${layout}" mep -s "${population_size}"
fi

if [[ ! -f "${population_yaml}" ]]; then
    echo "[ERROR] Missing population yaml: ${population_yaml}"
    exit 1
fi

ulimit -n 65536

echo "env=${env}, layout=${layout}, algo=${algo}, exp=${exp}, mode=${policy_mode}, seeds=${seed_begin}..${seed_max}"
for seed in $(seq "${seed_begin}" "${seed_max}");
do
    run_dir="${results_root}/${env}/${layout}/${policy_mode}/${algo}/${exp}/seed${seed}"
    ts=$(date +%Y%m%d_%H%M%S)
    run_log="${log_dir}/train_mep_stage1_${layout}_s${population_size}_seed${seed}_${ts}.log"
    echo "seed=${seed}, run_dir=${run_dir}"
    echo "log=${run_log}"

    python train/train_mep.py --env_name "${env}" --algorithm_name "${algo}" --experiment_name "${exp}" --layout_name "${layout}" --num_agents "${num_agents}" --agent_policy_names ${agent_policy_names} \
    --seed "${seed}" --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps "${num_env_steps}" --reward_shaping_horizon "${reward_shaping_horizon}" \
    --train_env_batch "${train_batch}" --n_rollout_threads "${train_batch}" --dummy_batch_size 1 \
    --overcooked_version "${version}" \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --stage 1 --policy_mode "${policy_mode}" \
    --reward_shaping_role individual --reward_shaping_roles "${role_csv}" \
    --mep_entropy_alpha 0.01 \
    --population_yaml_path "${population_yaml}" \
    --population_size "${population_size}" --adaptive_agent_name mep_adaptive \
    --save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads "${eval_threads}" --eval_episodes 10 \
    --use_render --save_gifs --n_render_rollout_threads "${n_render_rollout_threads}" --render_episodes "${render_episodes}" \
    --use_proper_time_limits \
    --wandb_name "${wandb_name}" 2>&1 | tee "${run_log}"
done
