from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from loguru import logger

from zsceval.analysis.core import policy_loader, runner
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw_trajectories"
SUMMARY_DIR = DATA_DIR / "summary_tables"


def build_parser() -> argparse.ArgumentParser:
    parser = get_config()
    parser = get_overcooked_args(parser)

    parser.add_argument("--analysis_task", type=str, choices=["xp", "tsne", "curve"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seed list for XP matrix.")
    parser.add_argument("--ego_seed", type=int, default=None, help="Ego seed for tsne/curve.")
    parser.add_argument("--partner_seed", type=int, default=None, help="Partner seed for tsne/curve.")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step for XP/TSNE.")
    parser.add_argument("--partner_step", type=int, default=None, help="Fixed partner step for curve.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per pair.")
    parser.add_argument("--results_root", type=str, default=None, help="Results root directory.")
    parser.add_argument("--output_name", type=str, default=None, help="Output file name override.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--policy_type", type=str, choices=["shared", "separated"], default="shared")
    parser.add_argument("--max_steps", type=int, default=None, help="Limit checkpoints for curve.")
    parser.add_argument(
        "--target_error_types",
        nargs="+",
        type=str,
        default=None,
        help="List of error types to count as violations (e.g., COLLISION ONION_COUNTER_REGRAB).",
    )

    return parser


def main():
    args = build_parser().parse_args()
    args.use_wandb = False
    args.use_render = False
    args.use_eval = False
    args.share_policy = args.policy_type == "shared"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.cuda = device.type == "cuda"
    run_dir = str(DATA_DIR)
    envs = runner.make_eval_env(args, run_dir)

    if args.analysis_task == "xp":
        run_xp_matrix(args, envs, device)
    elif args.analysis_task == "tsne":
        run_tsne_data(args, envs, device)
    elif args.analysis_task == "curve":
        run_learning_curve(args, envs, device)

    envs.close()


def run_xp_matrix(all_args, envs, device):
    if not all_args.seeds:
        raise ValueError("--seeds is required for xp task.")

    policy_cache: Dict[Tuple[int, Optional[int], Optional[int]], Tuple[object, Optional[int]]] = {}
    results = []

    for seed_i in all_args.seeds:
        for seed_j in all_args.seeds:
            policy_a, step_a = _get_policy(
                all_args, envs, seed_i, device, agent_id=0, step=all_args.checkpoint_step, cache=policy_cache
            )
            policy_b, step_b = _get_policy(
                all_args, envs, seed_j, device, agent_id=1, step=all_args.checkpoint_step, cache=policy_cache
            )
            summaries, _ = runner.run_pair_episodes(
                policy_a,
                policy_b,
                envs,
                num_episodes=all_args.episodes,
                all_args=all_args,
                deterministic=True,
                seed=all_args.seed,
            )

            mean_return = sum(s.episode_return for s in summaries) / len(summaries)
            mean_violation = sum(s.violation_rate for s in summaries) / len(summaries)
            results.append(
                {
                    "seed_ego": seed_i,
                    "seed_partner": seed_j,
                    "return": mean_return,
                    "violation_rate": mean_violation,
                    "step_ego": step_a,
                    "step_partner": step_b,
                    "episodes": len(summaries),
                }
            )

    df = pd.DataFrame(results)
    output_name = all_args.output_name or f"xp_result_{all_args.layout_name}_{all_args.algorithm_name}.csv"
    output_path = SUMMARY_DIR / output_name
    df.to_csv(output_path, index=False)
    logger.info(f"Saved XP matrix data to {output_path}")


def run_tsne_data(all_args, envs, device):
    if all_args.ego_seed is None or all_args.partner_seed is None:
        raise ValueError("--ego_seed and --partner_seed are required for tsne task.")

    policy_cache: Dict[Tuple[int, Optional[int], Optional[int]], Tuple[object, Optional[int]]] = {}
    sp_policy_a, sp_step = _get_policy(
        all_args, envs, all_args.ego_seed, device, agent_id=0, step=all_args.checkpoint_step, cache=policy_cache
    )
    sp_policy_b, _ = _get_policy(
        all_args, envs, all_args.ego_seed, device, agent_id=1, step=all_args.checkpoint_step, cache=policy_cache
    )
    xp_policy_b, xp_step = _get_policy(
        all_args, envs, all_args.partner_seed, device, agent_id=1, step=all_args.checkpoint_step, cache=policy_cache
    )

    _, sp_traj = runner.run_pair_episodes(
        sp_policy_a,
        sp_policy_b,
        envs,
        num_episodes=all_args.episodes,
        all_args=all_args,
        deterministic=True,
        collect_trajectory=True,
        seed=all_args.seed,
    )
    for record in sp_traj:
        record.update(
            {
                "pair_type": "sp",
                "seed_ego": all_args.ego_seed,
                "seed_partner": all_args.ego_seed,
                "step_ego": sp_step,
                "step_partner": sp_step,
            }
        )

    _, xp_traj = runner.run_pair_episodes(
        sp_policy_a,
        xp_policy_b,
        envs,
        num_episodes=all_args.episodes,
        all_args=all_args,
        deterministic=True,
        collect_trajectory=True,
        seed=all_args.seed,
    )
    for record in xp_traj:
        record.update(
            {
                "pair_type": "xp",
                "seed_ego": all_args.ego_seed,
                "seed_partner": all_args.partner_seed,
                "step_ego": sp_step,
                "step_partner": xp_step,
            }
        )

    df = pd.DataFrame(sp_traj + xp_traj)
    output_name = all_args.output_name or f"tsne_data_{all_args.layout_name}_{all_args.algorithm_name}.pkl"
    output_path = RAW_DIR / output_name
    df.to_pickle(output_path)
    logger.info(f"Saved t-SNE data to {output_path}")


def run_learning_curve(all_args, envs, device):
    if all_args.ego_seed is None or all_args.partner_seed is None:
        raise ValueError("--ego_seed and --partner_seed are required for curve task.")

    steps = policy_loader.list_checkpoints(
        layout=all_args.layout_name,
        algo=all_args.algorithm_name,
        experiment=all_args.experiment_name,
        seed=all_args.ego_seed,
        agent_id=None if all_args.share_policy else 0,
        share_policy=all_args.share_policy,
        env_name=all_args.env_name,
        results_root=all_args.results_root,
        policy_type=all_args.policy_type,
    )
    if all_args.partner_step is None:
        partner_steps = policy_loader.list_checkpoints(
            layout=all_args.layout_name,
            algo=all_args.algorithm_name,
            experiment=all_args.experiment_name,
            seed=all_args.partner_seed,
            agent_id=None if all_args.share_policy else 1,
            share_policy=all_args.share_policy,
            env_name=all_args.env_name,
            results_root=all_args.results_root,
            policy_type=all_args.policy_type,
        )
        if not partner_steps:
            raise ValueError("No partner checkpoints found for learning curve.")
        partner_steps_set = set(partner_steps)
        steps = [step for step in steps if step in partner_steps_set]

    if all_args.max_steps:
        steps = steps[: all_args.max_steps]

    if not steps:
        if all_args.partner_step is None:
            raise ValueError("No overlapping checkpoints found for learning curve.")
        raise ValueError("No checkpoints found for learning curve.")

    partner_policy = None
    partner_step = None
    if all_args.partner_step is not None:
        partner_policy, partner_step = _get_policy(
            all_args, envs, all_args.partner_seed, device, agent_id=1, step=all_args.partner_step, cache={}
        )

    results = []
    for step in steps:
        policy_cache: Dict[Tuple[int, Optional[int], Optional[int]], Tuple[object, Optional[int]]] = {}
        sp_policy_a, _ = _get_policy(
            all_args, envs, all_args.ego_seed, device, agent_id=0, step=step, cache=policy_cache
        )
        sp_policy_b, _ = _get_policy(
            all_args, envs, all_args.ego_seed, device, agent_id=1, step=step, cache=policy_cache
        )

        if all_args.partner_step is None:
            partner_policy, partner_step = _get_policy(
                all_args, envs, all_args.partner_seed, device, agent_id=1, step=step, cache=policy_cache
            )

        sp_summaries, _ = runner.run_pair_episodes(
            sp_policy_a,
            sp_policy_b,
            envs,
            num_episodes=all_args.episodes,
            all_args=all_args,
            deterministic=True,
            seed=all_args.seed,
        )
        xp_summaries, _ = runner.run_pair_episodes(
            sp_policy_a,
            partner_policy,
            envs,
            num_episodes=all_args.episodes,
            all_args=all_args,
            deterministic=True,
            seed=all_args.seed,
        )

        sp_return = sum(s.episode_return for s in sp_summaries) / len(sp_summaries)
        xp_return = sum(s.episode_return for s in xp_summaries) / len(xp_summaries)
        sp_violation = sum(s.violation_rate for s in sp_summaries) / len(sp_summaries)
        xp_violation = sum(s.violation_rate for s in xp_summaries) / len(xp_summaries)

        results.append(
            {
                "step": step,
                "sp_return": sp_return,
                "xp_return": xp_return,
                "sp_violation": sp_violation,
                "xp_violation": xp_violation,
                "partner_step": partner_step,
            }
        )

    df = pd.DataFrame(results)
    output_name = all_args.output_name or f"learning_curve_{all_args.layout_name}_{all_args.algorithm_name}.csv"
    output_path = SUMMARY_DIR / output_name
    df.to_csv(output_path, index=False)
    logger.info(f"Saved learning curve data to {output_path}")


def _get_policy(
    all_args,
    envs,
    seed: int,
    device,
    agent_id: int,
    step: Optional[int],
    cache: Dict[Tuple[int, Optional[int], Optional[int]], Tuple[object, Optional[int]]],
):
    key = (seed, agent_id if not all_args.share_policy else None, step)
    if key in cache:
        return cache[key]

    policy, resolved_step = policy_loader.load_agent(
        all_args=all_args,
        envs=envs,
        layout=all_args.layout_name,
        algo=all_args.algorithm_name,
        experiment=all_args.experiment_name,
        seed=seed,
        step=step,
        agent_id=None if all_args.share_policy else agent_id,
        share_policy=all_args.share_policy,
        results_root=all_args.results_root,
        device=device,
        policy_type=all_args.policy_type,
    )
    cache[key] = (policy, resolved_step)
    return cache[key]


if __name__ == "__main__":
    main()
