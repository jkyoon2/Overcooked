#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <layout> <gpu_id>" >&2
  exit 2
fi

layout="$1"
gpu_id="$2"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../../../.." && pwd)"

cd "${root_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${root_dir}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python_bin="${PYTHON:-/home/juliecandoit98/miniconda3/envs/zsceval/bin/python}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] start layout=${layout} seeds=11-15 gpu=${CUDA_VISIBLE_DEVICES}"

for seed in 11 12 13 14 15; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] start layout=${layout} seed=${seed} gpu=${CUDA_VISIBLE_DEVICES}"
  "${python_bin}" -u zsceval/scripts/overcooked/train/train_sp.py \
    --env_name Overcooked \
    --algorithm_name rmappo \
    --experiment_name sp \
    --layout_name "${layout}" \
    --num_agents 4 \
    --agent_policy_names ppo ppo ppo ppo \
    --seed "${seed}" \
    --n_training_threads 1 \
    --n_rollout_threads 50 \
    --dummy_batch_size 2 \
    --num_mini_batch 1 \
    --episode_length 400 \
    --num_env_steps 1e7 \
    --reward_shaping_horizon 1e8 \
    --reward_shaping_role individual \
    --reward_shaping_roles individual,individual,individual,individual \
    --overcooked_version new \
    --ppo_epoch 15 \
    --entropy_coefs 0.2 0.05 0.01 \
    --entropy_coef_horizons 0 5e6 1e7 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
    --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 \
    --log_interval 10 \
    --use_eval \
    --eval_interval 20 \
    --n_eval_rollout_threads 10 \
    --share_policy \
    --wandb_name juliejung98
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] done layout=${layout} seed=${seed}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] done layout=${layout} seeds=11-15"
