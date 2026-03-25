from __future__ import annotations

"""
Collect (s, a, w, z) strategy dataset by pairwise rollout of trained policies.

Minimal implementation spec
1) Policy pool (simple):
   - --policy_type shared    -> individual x seeds       => (len(seeds) * 1)^2
   - --policy_type separated -> supplier,cook x seeds    => (len(seeds) * 2)^2
   - --policy_type mixed     -> supplier,cook,individual => (len(seeds) * 3)^2

2) PolicySpec (minimal):
   - recipe_code, role, seed

3) Output files:
   - transitions.pkl: list[dict], one row per agent-step
   - episodes.pkl: list[dict], one row per agent-episode
   - manifest.json
"""

import argparse
import itertools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from loguru import logger

from zsceval.analysis.core import policy_loader, runner
from zsceval.analysis.core.strategy_laber import (
    encode_w_id,
    infer_recipe_code_from_layout,
    infer_role_from_episode,
    normalize_recipe_code,
    normalize_role,
)
from zsceval.analysis.core.subtask_labeler import backfill_subtasks
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args


@dataclass(frozen=True)
class PolicySpec:
    recipe_code: str
    role: str
    seed: int

    @property
    def policy_id(self) -> str:
        return f"seed{self.seed}:{self.role}"


def build_parser() -> argparse.ArgumentParser:
    parser = get_config()
    parser = get_overcooked_args(parser)

    parser.add_argument("--policy_type", type=str, choices=["shared", "separated", "mixed"], default="separated")
    parser.add_argument("--pool_seeds", type=int, nargs="+", default=[1, 2, 3], help="Default: 1 2 3")
    parser.add_argument("--recipe_code", type=str, default=None, help="One of TTT/TTO/TOO/OOO. Default: infer from layout.")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Default: latest")
    parser.add_argument("--episodes_per_pair", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--only_successful_episodes", action="store_true")
    parser.add_argument("--backfill_horizon", type=int, default=40, help="Max backfill length for z labeling")
    parser.add_argument("--results_root", type=str, default=None, help="Where trained checkpoints are read from")
    parser.add_argument("--output_root", type=str, default=None, help="Where collected dataset is saved")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max_pairs", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.use_wandb = False
    args.use_render = False
    args.use_eval = False
    args.share_policy = args.policy_type == "shared"
    prepare_runtime_args(args)

    recipe_code = normalize_recipe_code(args.recipe_code) if args.recipe_code else infer_recipe_code_from_layout(args.layout_name)
    policy_pool = build_policy_pool(policy_type=args.policy_type, recipe_code=recipe_code, seeds=args.pool_seeds)
    pairings = list(itertools.product(policy_pool, policy_pool))
    if args.max_pairs is not None:
        pairings = pairings[: args.max_pairs]
    if not pairings:
        raise ValueError("No pairings to rollout.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    args.cuda = device.type == "cuda"

    eval_run_dir = str((Path(__file__).resolve().parent / "data" / "strategy_datasets" / "_tmp_eval_run").resolve())
    envs = runner.make_eval_env(args, eval_run_dir)
    try:
        step_records = collect_pairwise_step_records(args=args, envs=envs, device=device, pairings=pairings)
    finally:
        envs.close()

    transitions, episodes = build_transition_rows(
        step_records=step_records,
        recipe_code=recipe_code,
        backfill_horizon=args.backfill_horizon,
        only_successful_episodes=args.only_successful_episodes,
    )
    output_dir = save_dataset(transitions=transitions, episodes=episodes, args=args, recipe_code=recipe_code, pool_size=len(policy_pool))
    logger.info(f"Saved strategy dataset to {output_dir}")


def prepare_runtime_args(args) -> None:
    """
    Fill minimal runtime defaults expected by Overcooked env/eval stack.
    """
    num_agents = int(getattr(args, "num_agents", 2))

    # train_agent_pool.sh uses episode_length=400, while global default config is 200.
    # If left at 200 default, per-pair timestep counts appear halved in analysis.
    if int(getattr(args, "episode_length", 200)) == 200:
        args.episode_length = 400

    # Match train_agent_pool.sh model architecture defaults.
    if not getattr(args, "cnn_layers_params", None):
        args.cnn_layers_params = "32,3,1,1 64,3,1,1 32,3,1,1"
    if getattr(args, "algorithm_name", "").lower() == "rmappo":
        if not getattr(args, "use_recurrent_policy", False) and not getattr(args, "use_naive_recurrent_policy", False):
            args.use_recurrent_policy = True

    # Overcooked_Env.__init__ checks len(all_args.agent_policy_names)
    # and crashes when the attribute exists but is None.
    policy_names = getattr(args, "agent_policy_names", None)
    if isinstance(policy_names, str):
        policy_names = [p.strip() for p in policy_names.split(",") if p.strip()]
    elif isinstance(policy_names, (list, tuple)):
        policy_names = [str(p).strip() for p in policy_names if str(p).strip()]
    else:
        policy_names = []

    if len(policy_names) == 1 and num_agents > 1:
        policy_names = policy_names * num_agents
    if len(policy_names) != num_agents:
        policy_names = ["ppo"] * num_agents
    args.agent_policy_names = policy_names

    reward_shaping_role = getattr(args, "reward_shaping_role", None)
    if reward_shaping_role is None:
        reward_shaping_role = "individual"
    reward_shaping_role = str(reward_shaping_role).strip().lower()
    args.reward_shaping_role = reward_shaping_role

    reward_roles = getattr(args, "reward_shaping_roles_list", None)
    if reward_roles is None:
        reward_roles = getattr(args, "reward_shaping_roles", None)
    if reward_roles is None:
        parsed_roles = [reward_shaping_role] * num_agents
    elif isinstance(reward_roles, str):
        parsed_roles = [r.strip().lower() for r in reward_roles.split(",") if r.strip()]
    elif isinstance(reward_roles, (list, tuple)):
        parsed_roles = [str(r).strip().lower() for r in reward_roles if str(r).strip()]
    else:
        parsed_roles = [reward_shaping_role] * num_agents

    if len(parsed_roles) == 1 and num_agents > 1:
        parsed_roles = parsed_roles * num_agents
    if len(parsed_roles) != num_agents:
        parsed_roles = [reward_shaping_role] * num_agents
    args.reward_shaping_roles_list = parsed_roles
    args.reward_shaping_roles = ",".join(parsed_roles)


def build_policy_pool(policy_type: str, recipe_code: str, seeds: Sequence[int]) -> List[PolicySpec]:
    if policy_type == "shared":
        roles = ["individual"]
    elif policy_type == "separated":
        roles = ["supplier", "cook"]
    elif policy_type == "mixed":
        roles = ["supplier", "cook", "individual"]
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    return [PolicySpec(recipe_code=recipe_code, role=normalize_role(role), seed=int(seed)) for seed in seeds for role in roles]


def resolve_checkpoint_target(global_policy_type: str, role: str) -> Tuple[str, bool, Optional[str]]:
    """
    Returns:
      policy_family: 'shared' or 'separated'
      share_policy: bool
      ckpt_role: role subdir for separated checkpoints, else None
    """
    if global_policy_type == "shared":
        return "shared", True, None
    if global_policy_type == "separated":
        return "separated", False, role
    if global_policy_type == "mixed":
        if role == "individual":
            return "shared", True, None
        return "separated", False, role
    raise ValueError(f"Unsupported policy_type: {global_policy_type}")


def collect_pairwise_step_records(
    args,
    envs,
    device: torch.device,
    pairings: Sequence[Tuple[PolicySpec, PolicySpec]],
) -> List[dict]:
    cache: Dict[Tuple[str, int, str, Optional[int]], Tuple[object, Optional[int]]] = {}
    all_records: List[dict] = []

    for pair_id, (ego_spec, partner_spec) in enumerate(pairings):
        logger.info(
            f"[{pair_id + 1}/{len(pairings)}] pair={ego_spec.policy_id} x {partner_spec.policy_id}, "
            f"episodes={args.episodes_per_pair}"
        )

        ego_policy, ego_step = get_policy_from_cache(cache, args, envs, device, ego_spec, agent_id=0)
        partner_policy, partner_step = get_policy_from_cache(cache, args, envs, device, partner_spec, agent_id=1)

        _, step_records = runner.run_pair_episodes(
            ego_policy,
            partner_policy,
            envs,
            num_episodes=args.episodes_per_pair,
            all_args=args,
            deterministic=not args.stochastic,
            collect_trajectory=True,
            flatten_obs=True,
            collect_step_data=True,
            seed=args.seed,
        )

        for rec in step_records:
            rec.update(
                {
                    "pair_id": pair_id,
                    "ego_policy_id": ego_spec.policy_id,
                    "partner_policy_id": partner_spec.policy_id,
                    "ego_seed": ego_spec.seed,
                    "partner_seed": partner_spec.seed,
                    "ego_role_fixed": ego_spec.role,
                    "partner_role_fixed": partner_spec.role,
                    "ego_step": ego_step,
                    "partner_step": partner_step,
                }
            )
            all_records.append(rec)

    return all_records


def get_policy_from_cache(
    cache: Dict[Tuple[str, int, str, Optional[int]], Tuple[object, Optional[int]]],
    args,
    envs,
    device: torch.device,
    spec: PolicySpec,
    agent_id: int,
):
    policy_family, share_policy, ckpt_role = resolve_checkpoint_target(args.policy_type, spec.role)
    cache_agent_id = None if share_policy else agent_id
    cache_key = (policy_family, spec.seed, spec.role, cache_agent_id)
    if cache_key in cache:
        return cache[cache_key]

    policy, resolved_step = policy_loader.load_agent(
        all_args=args,
        envs=envs,
        layout=args.layout_name,
        algo=args.algorithm_name,
        experiment=args.experiment_name,
        seed=spec.seed,
        step=args.checkpoint_step,
        agent_id=cache_agent_id,
        share_policy=share_policy,
        results_root=args.results_root,
        device=device,
        policy_type=policy_family,
        role=ckpt_role,
    )
    cache[cache_key] = (policy, resolved_step)
    return cache[cache_key]


def build_transition_rows(
    step_records: Sequence[dict],
    recipe_code: str,
    backfill_horizon: int,
    only_successful_episodes: bool,
) -> Tuple[List[dict], List[dict]]:
    grouped: Dict[Tuple[int, int, int], List[dict]] = {}
    for rec in step_records:
        grouped.setdefault((int(rec["pair_id"]), int(rec["env_index"]), int(rec["episode_id"])), []).append(rec)

    transitions: List[dict] = []
    episodes: List[dict] = []

    for (pair_id, env_index, episode_id), ep_steps in grouped.items():
        ep_steps = sorted(ep_steps, key=lambda x: int(x["timestep"]))
        if not bool(ep_steps[-1].get("done", False)):
            continue

        for agent_id in (0, 1):
            agent_steps = [extract_agent_step(step, agent_id=agent_id) for step in ep_steps]
            success = any(_is_delivery_step(step) for step in agent_steps)
            if only_successful_episodes and not success:
                continue

            fixed_role = normalize_role(ep_steps[0]["ego_role_fixed"] if agent_id == 0 else ep_steps[0]["partner_role_fixed"])
            posthoc_role = infer_role_from_episode(agent_steps)
            w_fixed_id = encode_w_id(recipe_code, fixed_role)
            w_posthoc_id = encode_w_id(recipe_code, posthoc_role)
            z_ids = backfill_subtasks(agent_steps, backfill_horizon=backfill_horizon)

            ep_sparse_return = float(sum(step["sparse_r"] for step in agent_steps))
            ep_shaped_return = float(sum(step["shaped_r"] for step in agent_steps))
            ep_total_return = ep_sparse_return + ep_shaped_return

            episode_uid = f"pair{pair_id}_env{env_index}_ep{episode_id}_agent{agent_id}"
            episodes.append(
                {
                    "episode_uid": episode_uid,
                    "pair_id": pair_id,
                    "env_index": env_index,
                    "episode_id": episode_id,
                    "agent_id": agent_id,
                    "ego_policy_id": ep_steps[0]["ego_policy_id"],
                    "partner_policy_id": ep_steps[0]["partner_policy_id"],
                    "w_fixed_id": w_fixed_id,
                    "w_posthoc_id": w_posthoc_id,
                    "success": success,
                    "length": len(agent_steps),
                    "sparse_return": ep_sparse_return,
                    "shaped_return": ep_shaped_return,
                    "episode_return": ep_total_return,
                }
            )

            for local_t, (step, z_id) in enumerate(zip(agent_steps, z_ids)):
                transitions.append(
                    {
                        "episode_uid": episode_uid,
                        "pair_id": pair_id,
                        "env_index": env_index,
                        "episode_id": episode_id,
                        "agent_id": agent_id,
                        "timestep": local_t,
                        "ego_policy_id": ep_steps[0]["ego_policy_id"],
                        "partner_policy_id": ep_steps[0]["partner_policy_id"],
                        "s": step["state"],
                        "a": step["action"],
                        "w_id": w_posthoc_id,
                        "w_fixed_id": w_fixed_id,
                        "w_posthoc_id": w_posthoc_id,
                        "z_id": z_id,
                        "sparse_r": step["sparse_r"],
                        "shaped_r": step["shaped_r"],
                        "reward": step["sparse_r"] + step["shaped_r"],
                        "done": bool(step["done"]),
                        "success": success,
                    }
                )

    return transitions, episodes


def extract_agent_step(step_record: Mapping[str, object], agent_id: int) -> dict:
    shaped_info_timestep = step_record.get("shaped_info_timestep") or []
    shaped_info = {}
    if isinstance(shaped_info_timestep, Sequence) and len(shaped_info_timestep) > agent_id:
        maybe_dict = shaped_info_timestep[agent_id]
        if isinstance(maybe_dict, Mapping):
            shaped_info = dict(maybe_dict)

    event_infos = step_record.get("event_infos") or {}
    event_flags = {}
    if isinstance(event_infos, Mapping):
        for key, vals in event_infos.items():
            if isinstance(vals, Sequence) and len(vals) > agent_id:
                event_flags[key] = bool(vals[agent_id])

    shaped_r = 0.0
    sparse_r = 0.0
    shaped_by_agent = step_record.get("shaped_r_by_agent") or []
    sparse_by_agent = step_record.get("sparse_r_by_agent") or []
    if isinstance(shaped_by_agent, Sequence) and len(shaped_by_agent) > agent_id:
        shaped_r = float(shaped_by_agent[agent_id])
    if isinstance(sparse_by_agent, Sequence) and len(sparse_by_agent) > agent_id:
        sparse_r = float(sparse_by_agent[agent_id])

    state_key = "obs_ego" if agent_id == 0 else "obs_partner"
    action_key = "action_ego" if agent_id == 0 else "action_partner"
    return {
        "state": step_record.get(state_key),
        "action": int(step_record.get(action_key, 0)),
        "done": bool(step_record.get("done", False)),
        "shaped_info": shaped_info,
        "event_flags": event_flags,
        "shaped_r": shaped_r,
        "sparse_r": sparse_r,
    }


def _is_delivery_step(agent_step: Mapping[str, object]) -> bool:
    shaped_info = agent_step.get("shaped_info", {})
    if isinstance(shaped_info, Mapping) and float(shaped_info.get("delivery", 0)) > 0:
        return True
    event_flags = agent_step.get("event_flags", {})
    if isinstance(event_flags, Mapping):
        return bool(event_flags.get("soup_delivery", False))
    return False


def save_dataset(
    transitions: List[dict],
    episodes: List[dict],
    args,
    recipe_code: str,
    pool_size: int,
) -> Path:
    default_root = Path(__file__).resolve().parent / "data" / "strategy_datasets"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_root.resolve()
    dataset_name = args.dataset_name or f"{args.layout_name}_{args.policy_type}_{args.algorithm_name}_{args.experiment_name}"
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "transitions.pkl").open("wb") as f:
        pickle.dump(transitions, f)
    with (output_dir / "episodes.pkl").open("wb") as f:
        pickle.dump(episodes, f)

    manifest = {
        "layout_name": args.layout_name,
        "recipe_code": recipe_code,
        "policy_type": args.policy_type,
        "algorithm_name": args.algorithm_name,
        "experiment_name": args.experiment_name,
        "pool_seeds": list(args.pool_seeds),
        "pool_size": pool_size,
        "pair_count": pool_size * pool_size,
        "episode_length": int(getattr(args, "episode_length", -1)),
        "checkpoint_step": args.checkpoint_step,
        "episodes_per_pair": args.episodes_per_pair,
        "only_successful_episodes": bool(args.only_successful_episodes),
        "backfill_horizon": args.backfill_horizon,
        "w_id_definition": "w_id is set to w_posthoc_id",
        "w_fixed_id_definition": "w derived from rollout source policy role (seed+role pool spec)",
        "w_posthoc_id_definition": "w derived from posthoc role inference on collected events",
        "num_transitions": len(transitions),
        "num_agent_episodes": len(episodes),
        "transition_columns": sorted(list(transitions[0].keys())) if transitions else [],
        "episode_columns": sorted(list(episodes[0].keys())) if episodes else [],
        "transition_column_desc": transition_column_desc(),
        "episode_column_desc": episode_column_desc(),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output_dir


def transition_column_desc() -> Dict[str, str]:
    return {
        "episode_uid": "Unique episode key: pair/env/episode/agent",
        "pair_id": "Pair index in cartesian policy pairing list",
        "env_index": "Vectorized env worker index",
        "episode_id": "Episode index inside env worker",
        "agent_id": "0 or 1",
        "timestep": "Step index inside the agent episode",
        "ego_policy_id": "Policy id assigned to ego slot (agent0)",
        "partner_policy_id": "Policy id assigned to partner slot (agent1)",
        "s": "Flattened local observation vector",
        "a": "Discrete action id",
        "w_id": "Alias of w_posthoc_id",
        "w_fixed_id": "Recipe-role id from source pool label",
        "w_posthoc_id": "Recipe-role id inferred from rollout events",
        "z_id": "Subtask id (with backfill)",
        "sparse_r": "Sparse reward at this step",
        "shaped_r": "Shaped reward at this step",
        "reward": "sparse_r + shaped_r",
        "done": "Episode done flag at this step",
        "success": "Episode contains at least one delivery",
    }


def episode_column_desc() -> Dict[str, str]:
    return {
        "episode_uid": "Unique episode key: pair/env/episode/agent",
        "pair_id": "Pair index in cartesian policy pairing list",
        "env_index": "Vectorized env worker index",
        "episode_id": "Episode index inside env worker",
        "agent_id": "0 or 1",
        "ego_policy_id": "Policy id assigned to ego slot (agent0)",
        "partner_policy_id": "Policy id assigned to partner slot (agent1)",
        "w_fixed_id": "Recipe-role id from source pool label",
        "w_posthoc_id": "Recipe-role id inferred from rollout events",
        "success": "Episode contains at least one delivery",
        "length": "Number of timesteps in this agent episode",
        "sparse_return": "Sum of sparse rewards over episode",
        "shaped_return": "Sum of shaped rewards over episode",
        "episode_return": "sparse_return + shaped_return",
    }


if __name__ == "__main__":
    main()
