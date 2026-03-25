#!/bin/bash
set -euo pipefail

env="Overcooked"

layout=${1:-}
seed_begin=${2:-4}
seed_max=${3:-6}

if [[ -z "${layout}" ]]; then
    echo "Usage: bash zsceval/scripts/overcooked/shell/train_agent_pool.sh <layout> [seed_begin] [seed_max]"
    exit 1
fi

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi
version="new"

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps_shared="1e7"
num_env_steps_separated="2e7"
agent_policy_names="ppo ppo"

num_agents=2
algo="rmappo"
exp="agent_pool_sp"
ulimit -n 65536

policy_modes=("separated" "shared")

echo "env=${env}, layout=${layout}, algo=${algo}, exp=${exp}, modes=${policy_modes[*]}, seeds=${seed_begin}..${seed_max}"

for policy_mode in "${policy_modes[@]}"; do
    if [[ "${policy_mode}" == "shared" ]]; then
        role_csv="individual,individual"
        num_env_steps="${num_env_steps_shared}"
    else
        role_csv="supplier,cook"
        num_env_steps="${num_env_steps_separated}"
    fi

    reward_shaping_horizon="${num_env_steps}"

    echo "========== policy_mode=${policy_mode}, roles=${role_csv}, num_env_steps=${num_env_steps} =========="
    for seed in $(seq "${seed_begin}" "${seed_max}"); do
        echo "policy_mode=${policy_mode}, seed=${seed}"
        # Resume-aware entrypoint: if periodic checkpoints exist in the seed run_dir, continue from latest.
        python train/train_sp_resume.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} --agent_policy_names ${agent_policy_names} \
        --seed ${seed} --n_training_threads 1 --n_rollout_threads 50 --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
        --policy_mode ${policy_mode} \
        --reward_shaping_roles ${role_csv} \
        --reward_shaping_role individual \
        --overcooked_version ${version} \
        --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
        --use_proper_time_limits \
        --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
        --use_render --save_gifs --n_render_rollout_threads 1 --render_episodes 1 \
        --wandb_name "juliejung98"
    done
done
