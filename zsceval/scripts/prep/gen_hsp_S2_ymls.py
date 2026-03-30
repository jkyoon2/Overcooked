from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from zsceval.utils.bias_agent_vars import LAYOUTS_EXPS
from zsceval.utils.hsp_pipeline import (
    overcooked_event_types,
    overcooked_version_for_layout,
    resolve_overcooked_eval_results_root,
    resolve_policy_pool_root,
)


def compute_metric(events: dict, event_types: list, num_agents: int):
    event_types_bi = [
        [f"{w0_i}-{k}_by_agent{a_i}" for a_i in range(num_agents) if a_i != w0_i]
        for w0_i in range(num_agents)
        for k in event_types
    ]
    event_types_bi = sum(event_types_bi, start=[])

    def empty_event_count():
        return {k: 0 for k in event_types_bi}

    ec = defaultdict(empty_event_count)
    for exp in events.keys():
        exp_i = int(exp.split("_")[0][3:])
        exp_ec = events[exp]
        for k in event_types_bi:
            ec[exp_i][k] += exp_ec.get(k, 0)
    exps = sorted(ec.keys())
    logger.info(f"exps: {exps}")
    event_np = np.array([[ec[i][k] for k in event_types_bi] for i in exps])
    df = pd.DataFrame(event_np, index=exps, columns=event_types_bi)
    logger.info(f"event df shape {df.shape}")
    event_ratio_np = event_np / (event_np.max(axis=0) + 1e-3).reshape(1, -1)

    return exps, event_ratio_np, df


def select_policies(runs, metric_np, K):
    S = []
    n = len(runs)
    if n == 0:
        return []
    if K > n:
        logger.warning(f"Requested K={K} but only {n} bias policies are available. Clamping K to {n}.")
        K = n
    S.append(np.random.randint(0, n))
    for _ in range(1, K):
        v = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            if i not in S:
                for j in S:
                    v[i] += abs(metric_np[i] - metric_np[j]).sum()
            else:
                v[i] = -1e9
        x = v.argmax()
        S.append(x)
    S = sorted([runs[i] for i in S])
    return S


MEP_EXPS = {
    5: "mep-S1-s5",
    10: "mep-S1-s10",
    15: "mep-S1-s15",
}


def parse_args():
    parser = argparse.ArgumentParser(description="hsp", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env", type=str, default="Overcooked")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("-l", "--layout", type=str, required=True, help="layout name")
    parser.add_argument("-k", type=int, default=6, help="number of selected policies")
    parser.add_argument("-s", type=int, default=5, help="population size of S1")
    parser.add_argument("-S", type=int, default=12, help="population size of training")
    parser.add_argument("--eval_result_dir", type=str, default=None)
    parser.add_argument("--policy_pool_path", type=str, default=None)
    parser.add_argument("--bias_agent_version", type=str, default="hsp")
    parser.add_argument("--adaptive_policy_config", type=str, default="rnn_policy_config.pkl")
    parser.add_argument("--bias_policy_config", type=str, default="rnn_policy_config.pkl")
    parser.add_argument("--population_policy_config", type=str, default="rnn_policy_config.pkl")
    parser.add_argument("--count-only", action="store_true", help="Print usable bias-agent count and exit.")

    args = parser.parse_args()
    return args


def load_event_profiles(eval_result_dir: Path, layout: str, policy_version: str, event_types: list[str], num_agents: int):
    events = dict()
    logfiles = sorted(eval_result_dir.glob(f"eval*{policy_version}*.json"))
    logfiles = [path for path in logfiles if "mid" not in path.name]
    logger.success(f"{len(logfiles)} eval files")

    exclude = set(f"hsp{x}" for x in LAYOUTS_EXPS.get(layout, []))

    def build_agent_names(policy_tag: str) -> list[str]:
        if policy_tag.endswith(("_init", "_mid", "_final")):
            return [f"{policy_tag}_w{a_i}" for a_i in range(num_agents)]
        return [f"{policy_tag}_final_w{a_i}" for a_i in range(num_agents)]

    for logfile in logfiles:
        with logfile.open(encoding="utf-8") as f:
            eval_result = json.load(f)
        hsp_exp_name = logfile.stem.split("eval-")[1]
        hsp_policy_id = hsp_exp_name.split("_")[0]
        if hsp_exp_name in exclude or hsp_policy_id in exclude:
            continue

        agents = build_agent_names(hsp_exp_name)
        full_exp_name = "-".join(agents)
        if eval_result.get(f"{full_exp_name}-eval_ep_sparse_r", 0.0) <= 0.1:
            logger.warning(f"exp {hsp_exp_name} has near-zero sparse reward")
            exclude.update({hsp_exp_name, hsp_policy_id})
            continue

        event_dict = defaultdict(list)
        agent_names = build_agent_names(hsp_exp_name)
        for event_name in event_types:
            for pair in permutations(agent_names):
                pair_name = "-".join(pair)
                w0_i = next((a_i for a_i, a_name in enumerate(pair) if "w0" in a_name), -1)
                if w0_i < 0:
                    continue
                for a_i, a_name in enumerate(pair):
                    if "w0" in a_name:
                        continue
                    event_dict[f"{w0_i}-{event_name}_by_agent{a_i}"].append(
                        eval_result.get(f"{pair_name}-eval_ep_{event_name}_by_agent{a_i}", 0.0)
                    )
        for key, values in event_dict.items():
            event_dict[key] = float(np.mean(values)) if values else 0.0
        events[hsp_exp_name] = event_dict

    for excluded in exclude:
        events.pop(excluded, None)
    return events


if __name__ == "__main__":
    args = parse_args()
    layout = args.layout
    overcooked_version = overcooked_version_for_layout(layout)
    K = args.k
    policy_version = args.bias_agent_version
    np.random.seed(0)
    random.seed(0)

    if args.env.lower() == "overcooked":
        event_types = overcooked_event_types(layout)
    else:
        event_types = [
            "actual_pass",
            "shot",
            "catch",
            "assist",
            "possession",
            "score",
        ]

    eval_result_dir = resolve_overcooked_eval_results_root(args.eval_result_dir) / layout / "bias"
    logger.info(f"eval result dir {eval_result_dir}")
    events = load_event_profiles(
        eval_result_dir=eval_result_dir,
        layout=layout,
        policy_version=policy_version,
        event_types=event_types,
        num_agents=args.num_agents,
    )
    logger.info(f"filtered exp num {len(events.keys())}")
    if args.count_only:
        print(len(events))
        raise SystemExit(0)
    if not events:
        raise RuntimeError(f"No usable bias evaluation results found in {eval_result_dir}")
    if len(events) < K:
        raise RuntimeError(
            f"Need at least k={K} usable bias agents for layout={layout}, but found only {len(events)} in {eval_result_dir}"
        )

    exps, metric_np, df = compute_metric(events, event_types, args.num_agents)
    df.to_excel(eval_result_dir / f"event_count_{policy_version}.xlsx", sheet_name="Events")

    runs = select_policies(exps, metric_np, K)
    logger.success(f"selected runs: {runs}")

    # generate HSP training config
    policy_pool_root = resolve_policy_pool_root(args.policy_pool_path)
    (policy_pool_root / layout / "hsp" / "s2").mkdir(parents=True, exist_ok=True)
    mep_exp = MEP_EXPS[args.s]
    for seed in range(1, 6):
        target_path = policy_pool_root / layout / "hsp" / "s2" / f"train-s{args.S}-{args.bias_agent_version}_{mep_exp}-{seed}.yml"
        with target_path.open("w", encoding="utf-8") as f:
            f.write(
                f"""\
hsp_adaptive:
    policy_config_path: {layout}/policy_config/{args.adaptive_policy_config}
    featurize_type: ppo
    train: True
"""
            )
            assert (args.S - int(K)) % 3 == 0, (args.S, K)
            POP_SIZE = (args.S - int(K)) // 3
            TOTAL_SIZE = args.s
            for p_i in range(1, POP_SIZE + 1):
                pt_i = (TOTAL_SIZE // 5 * (seed - 1) + p_i - 1) % TOTAL_SIZE + 1
                f.write(
                    f"""\
mep{p_i}_1:
    policy_config_path: {layout}/policy_config/{args.population_policy_config}
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, "mep", "s1", mep_exp, f"mep{pt_i}_init_actor.pt")}
mep{p_i}_2:
    policy_config_path: {layout}/policy_config/{args.population_policy_config}
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, "mep", "s1", mep_exp, f"mep{pt_i}_mid_actor.pt")}
mep{p_i}_3:
    policy_config_path: {layout}/policy_config/{args.population_policy_config}
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, "mep", "s1", mep_exp, f"mep{pt_i}_final_actor.pt")}
"""
                )
            for i, run_i in enumerate(runs):
                f.write(
                    f"""\
hsp{i+1}_final:
    policy_config_path: {layout}/policy_config/{args.bias_policy_config}
    featurize_type: ppo
    train: False
    model_path:
        actor: {layout}/hsp/s1/{policy_version}/hsp{run_i}_final_w0_actor.pt\n"""
                )
        logger.info(f"Wrote {target_path}")
