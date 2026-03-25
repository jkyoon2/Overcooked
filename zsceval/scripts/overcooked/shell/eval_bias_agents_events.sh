#!/bin/bash
set -euo pipefail

env="Overcooked"
layout=${1:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERCOOKED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${OVERCOOKED_DIR}/.." && pwd)"
cd "${OVERCOOKED_DIR}"

if [[ -z "${layout}" ]]; then
    echo "Usage: bash zsceval/scripts/overcooked/shell/eval_bias_agents_events.sh <layout>"
    exit 1
fi

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

num_agents=2
algo="population"
policy_version="hsp"
policy_pool="${POLICY_POOL:-${SCRIPTS_DIR}/policy_pool}"
mkdir -p "${policy_pool}"
export POLICY_POOL="$(cd "${policy_pool}" && pwd)"

template_path="${POLICY_POOL}/${layout}/hsp/s1/eval_template.yml"
if [[ ! -f "${template_path}" ]]; then
    python "${SCRIPTS_DIR}/prep/gen_bias_agent_eval_yml.py" "${layout}" --policy_pool_path "${POLICY_POOL}"
fi

policy_dir="${POLICY_POOL}/${layout}/hsp/s1/${policy_version}"
if [[ ! -d "${policy_dir}" ]]; then
    echo "[ERROR] Missing bias policy dir: ${policy_dir}"
    exit 1
fi

mapfile -t actor_files < <(find "${policy_dir}" -maxdepth 1 -name "hsp*_final_w0_actor.pt" | sort -V)
echo "env=${env}, layout=${layout}, eval bias agents in ${policy_dir}"
echo "found ${#actor_files[@]} final bias agents"

yml_dir="eval/eval_policy_pool/${layout}/bias"
mkdir -p "${yml_dir}"
mkdir -p "eval/results/${layout}/bias"

for actor_file in "${actor_files[@]}"; do
    actor_base="$(basename "${actor_file}" .pt)"
    agent0_policy_name="${actor_base%_actor}"
    agent1_policy_name="${agent0_policy_name/_w0/_w1}"
    exp="eval-${agent0_policy_name%%_w0}"
    yml="${yml_dir}/${exp}.yml"

    sed -e "s/agent0/${agent0_policy_name}/g" -e "s/agent1/${agent1_policy_name}/g" -e "s/pop/${policy_version}/g" "${template_path}" > "${yml}"

    echo "########################################"
    echo "evaluate ${agent0_policy_name}-${agent1_policy_name}"
    python eval/eval.py --env_name "${env}" --algorithm_name "${algo}" --experiment_name "${exp}" --layout_name "${layout}" \
    --num_agents "${num_agents}" --seed 1 --episode_length 400 --n_eval_rollout_threads 80 --eval_episodes 80 --eval_stochastic --dummy_batch_size 2 \
    --use_proper_time_limits \
    --use_wandb \
    --population_yaml_path "${yml}" --population_size 2 \
    --agent0_policy_name "${agent0_policy_name}" \
    --agent1_policy_name "${agent1_policy_name}" --overcooked_version "${version}" --eval_result_path "eval/results/${layout}/bias/${exp}.json"
    echo "########################################"
done
