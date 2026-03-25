#!/bin/bash
set -euo pipefail

env="Overcooked"
layout=${1:-}
seed_begin_override=${2:-}
seed_max_override=${3:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERCOOKED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OVERCOOKED_DIR}"

if [[ -z "${layout}" ]]; then
    echo "Usage: bash zsceval/scripts/overcooked/shell/train_bias_agents.sh <layout> [seed_begin] [seed_max]"
    exit 1
fi

join_by_comma() {
    local IFS=","
    echo "$*"
}

validate_weight_count() {
    local label=$1
    local expected=$2
    shift 2
    local actual=$#
    if [[ ${actual} -ne ${expected} ]]; then
        echo "[ERROR] ${label} expects ${expected} values, got ${actual}" >&2
        exit 1
    fi
}

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps="1e7"

num_agents=2
algo="rmappo"
stage="S1"
exp="hsp-${stage}"
policy_mode="shared"
agent_policy_names="ppo ppo"
n_rollout_threads=50
n_eval_rollout_threads=10
n_render_rollout_threads=1
render_episodes=1
wandb_name="${WANDB_NAME:-juliejung98}"
log_dir="${OVERCOOKED_DIR}/log"
mkdir -p "${log_dir}"

old_layouts=("random0" "random0_medium" "random1" "random3" "small_corridor" "unident_s")
version="new"
for old_layout in "${old_layouts[@]}"; do
    if [[ "${layout}" == "${old_layout}" ]]; then
        version="old"
        break
    fi
done

if [[ "${version}" == "old" ]]; then
    # old SHAPED_INFOS + sparse:
    # 0 put_onion_on_X
    # 1 put_dish_on_X
    # 2 put_soup_on_X
    # 3 pickup_onion_from_X
    # 4 pickup_onion_from_O
    # 5 pickup_dish_from_X
    # 6 pickup_dish_from_D
    # 7 pickup_soup_from_X
    # 8 USEFUL_DISH_PICKUP
    # 9 SOUP_PICKUP
    # 10 PLACEMENT_IN_POT
    # 11 delivery
    # 12 STAY
    # 13 MOVEMENT
    # 14 IDLE_MOVEMENT
    # 15 IDLE_INTERACT_X
    # 16 IDLE_INTERACT_EMPTY
    # 17 sparse_reward
    w1_values=(
        "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "1"
    )
    case "${layout}" in
        random0)
            w0_values=(
                "0" "0" "0" "0" "[0:10]" "0" "[0:10]" "[-20:0]" "3" "5" "3" "0" "[-0.1:0:0.1]" "0" "0" "0" "0" "[0.1:1]"
            )
            seed_begin=1
            seed_max=30
            ;;
        random0_medium)
            w0_values=(
                "0" "0" "0" "[-20:0]" "[-20:0:10]" "0" "[0:10]" "[-20:0]" "3" "5" "3" "0" "[-0.1:0:0.1]" "0" "0" "0" "0" "[0.1:1]"
            )
            seed_begin=1
            seed_max=54
            ;;
        small_corridor)
            w0_values=(
                "0" "0" "0" "0" "[-20:0:5]" "0" "[-20:0:5]" "0" "3" "5" "3" "[-20:0]" "[-0.1:0]" "0" "0" "0" "0" "[0.1:1]"
            )
            seed_begin=1
            seed_max=124
            ;;
        *)
            w0_values=(
                "0" "0" "0" "0" "[-20:0:10]" "0" "[-20:0:10]" "0" "3" "5" "3" "[-20:0]" "[-0.1:0:0.1]" "0" "0" "0" "0" "[0.1:1]"
            )
            seed_begin=1
            seed_max=176
            ;;
    esac
else
    # new SHAPED_INFOS + sparse:
    # 0 put_onion_on_X
    # 1 put_tomato_on_X
    # 2 put_dish_on_X
    # 3 put_soup_on_X
    # 4 pickup_onion_from_X
    # 5 pickup_onion_from_O
    # 6 pickup_tomato_from_X
    # 7 pickup_tomato_from_T
    # 8 pickup_dish_from_X
    # 9 pickup_dish_from_D
    # 10 pickup_soup_from_X
    # 11 USEFUL_DISH_PICKUP
    # 12 SOUP_PICKUP
    # 13 USEFUL_SOUP_PICKUP
    # 14 USELESS_SOUP_PICKUP
    # 15 PLACEMENT_IN_POT
    # 16 viable_placement
    # 17 optimal_placement
    # 18 catastrophic_placement
    # 19 useless_placement
    # 20 potting_onion
    # 21 potting_tomato
    # 22 cook
    # 23 useful_cook
    # 24 useless_cook
    # 25 delivery
    # 26 deliver_size_two_order
    # 27 deliver_size_three_order
    # 28 deliver_useless_order
    # 29 STAY
    # 30 MOVEMENT
    # 31 IDLE_MOVEMENT
    # 32 IDLE_INTERACT
    # 33 sparse_reward
    #
    # w1 is a fixed dense-shaping baseline aligned with train_sp.sh.
    # w0 is sampled from a broader bias family over recipe/utility-sensitive events.
    w1_values=(
        "0" "0" "0" "0" "0" "0.1" "0.1" "0" "0" "0.1" "0"
        "3" "0" "10" "-2" "3" "2" "2" "-2" "-2"
        "5" "5" "0" "20" "-5" "0" "7" "20" "-5"
        "-0.01" "-0.01" "-0.1" "-0.1" "30"
    )
    w0_values=(
        "0" "0" "0" "0" "0" "0.1" "0.1" "0" "0" "0.1" "0"
        "3" "0" "[0:5:10:15]" "[-10:-5:-2:0]" "3" "2" "2" "-2" "-2"
        "[0:5:10]" "[0:5:10]" "0" "[0:10:20:30]" "-5" "0" "[0:7:14]" "[0:20:30]" "[-10:-5:0]"
        "-0.01" "-0.01" "-0.1" "-0.1" "30"
    )
    seed_begin=1
    seed_max=72
fi

if [[ -n "${seed_begin_override}" ]]; then
    seed_begin=${seed_begin_override}
fi
if [[ -n "${seed_max_override}" ]]; then
    seed_max=${seed_max_override}
fi

validate_weight_count "w0" $(( ${#w1_values[@]} )) "${w0_values[@]}"
if [[ "${version}" == "old" ]]; then
    validate_weight_count "w1" 18 "${w1_values[@]}"
else
    validate_weight_count "w1" 34 "${w1_values[@]}"
fi

w0="$(join_by_comma "${w0_values[@]}")"
w1="$(join_by_comma "${w1_values[@]}")"

ulimit -n 65536

echo "env=${env}, layout=${layout}, algo=${algo}, exp=${exp}, version=${version}, mode=${policy_mode}, seeds=${seed_begin}..${seed_max}"
for seed in $(seq "${seed_begin}" "${seed_max}");
do
    ts=$(date +%Y%m%d_%H%M%S)
    run_log="${log_dir}/train_bias_agents_${layout}_seed${seed}_${ts}.log"
    echo "seed=${seed}, log=${run_log}"
    python train/train_bias_agent.py --env_name "${env}" --algorithm_name "${algo}" --experiment_name "${exp}" --layout_name "${layout}" --num_agents "${num_agents}" --agent_policy_names ${agent_policy_names} \
    --seed "${seed}" --n_training_threads 1 --n_rollout_threads "${n_rollout_threads}" --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps "${num_env_steps}" --reward_shaping_horizon "${reward_shaping_horizon}" \
    --overcooked_version "${version}" \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 "${w0}" --w1 "${w1}" --policy_mode "${policy_mode}" --random_index \
    --reward_shaping_role individual --reward_shaping_roles "individual,individual" \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads "${n_eval_rollout_threads}" \
    --use_render --save_gifs --n_render_rollout_threads "${n_render_rollout_threads}" --render_episodes "${render_episodes}" \
    --wandb_name "${wandb_name}" 2>&1 | tee "${run_log}"
done
