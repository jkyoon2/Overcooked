#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <wait_pgid> <wait_label> <target_layout> <gpu_id>" >&2
  exit 2
fi

wait_pgid="$1"
wait_label="$2"
target_layout="$3"
gpu_id="$4"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../../../.." && pwd)"
train_script="${script_dir}/train_sp_2agent_seed1_10.sh"

cd "${root_dir}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] queued target=${target_layout} gpu=${gpu_id}; waiting for ${wait_label} pgid=${wait_pgid}"

while ps -eo pgid= | awk -v pgid="${wait_pgid}" '$1 == pgid {found=1} END {exit !found}'; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] waiting target=${target_layout}; ${wait_label} pgid=${wait_pgid} still active"
  sleep 300
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] dependency finished: ${wait_label}; starting target=${target_layout} gpu=${gpu_id}"
exec "${train_script}" "${target_layout}" "${gpu_id}"
