from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from zsceval.analysis.collect_strategy_dataset import prepare_runtime_args
from zsceval.analysis.core import policy_loader
from zsceval.config import get_config
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as OvercookedNew
from zsceval.overcooked_config import OLD_LAYOUTS, get_overcooked_args


ACTION_NAME_BY_INDEX = {
    0: "north",
    1: "south",
    2: "east",
    3: "west",
    4: "stay",
    5: "interact",
}

ORIENTATION_NAME_BY_VECTOR = {
    (0, -1): "north",
    (0, 1): "south",
    (1, 0): "east",
    (-1, 0): "west",
    (0, 0): "stay",
}

OLD_ORDER_NAMES = {"onion", "tomato", "any"}
ROLE_CHOICES = ("supplier", "cook", "individual")
POLICY_TYPE_CHOICES = ("shared", "separated")
AUTO_OLD_DYNAMICS_LAYOUTS = {
    "random0_medium",
    "random3",
    "small_corridor",
    "multiplayer_schelling_3",
    "multiplayer_schelling",
}


@dataclass(frozen=True)
class AgentSpec:
    seed: Optional[int]
    policy_type: str
    role: Optional[str]
    step: Optional[int]
    actor_path: Optional[Path] = None


def build_parser() -> argparse.ArgumentParser:
    parser = get_config()
    parser = get_overcooked_args(parser)

    parser.add_argument("--policy_seed", type=int, default=None, help="Seed number for all loaded policies.")
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=POLICY_TYPE_CHOICES,
        default="shared",
        help="Checkpoint family for all loaded policies.",
    )
    parser.add_argument(
        "--ego_role",
        type=str,
        choices=ROLE_CHOICES,
        default=None,
        help="Role subdir for separated player 0 checkpoints.",
    )
    parser.add_argument(
        "--partner_role",
        type=str,
        choices=ROLE_CHOICES,
        default=None,
        help="Role subdir for separated player 1 checkpoints.",
    )
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint number to load.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Actor checkpoint path(s) to load directly. Provide one path to broadcast to every "
            "agent, or one path per agent."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Sample policy actions instead of using deterministic rollout.",
    )
    parser.add_argument(
        "--max_ticks",
        type=int,
        default=None,
        help="Optional hard cap on exported ticks per episode.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Root directory where checkpoints are loaded from.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional JSON output path. Defaults to json_trajectory/<layout>_<seed>_<checkpoint>.json.",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.overcooked_version = "new"
    args.use_wandb = False
    args.use_render = False
    args.use_eval = False
    args.share_policy = False
    prepare_runtime_args(args)
    apply_dynamics_defaults(args)
    validate_args(args)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    args.cuda = device.type == "cuda"

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_single_env(args, run_dir=str(output_path.parent))
    try:
        agent_specs = build_agent_specs(args, num_agents=env.all_args.num_agents)
        align_policy_encoding_with_checkpoints(args, env, agent_specs)
        policies = [
            load_policy(args, env, device, spec, agent_id=agent_id)[0]
            for agent_id, spec in enumerate(agent_specs)
        ]

        trajectory_json = generate_episode_trajectory(
            env=env,
            policies=policies,
            deterministic=not args.stochastic,
            max_ticks=args.max_ticks,
        )
    finally:
        env.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trajectory_json, f, indent=2)

    final_score = final_score_from_trajectory(trajectory_json)
    logger.info("Saved realtime JSON trajectory to {}", output_path)
    logger.info("Final score: {}", final_score)
    print(f"FINAL_SCORE={final_score}")


def final_score_from_trajectory(trajectory_json: Dict[str, Any]) -> int:
    dynamic_states = trajectory_json.get("dynamicState", [])
    if not dynamic_states:
        return 0
    return int(dynamic_states[-1].get("score", 0))


def apply_dynamics_defaults(args) -> None:
    # Keep using the overcooked_new environment, but opt specific layouts into
    # old pot dynamics so the last ingredient starts cooking automatically.
    old_dynamics_layouts = set(OLD_LAYOUTS) | AUTO_OLD_DYNAMICS_LAYOUTS
    args.old_dynamics = bool(getattr(args, "old_dynamics", False) or args.layout_name in old_dynamics_layouts)
    logger.info(
        "Trajectory generation config: layout={} overcooked_version={} old_dynamics={}",
        args.layout_name,
        args.overcooked_version,
        args.old_dynamics,
    )


def align_policy_encoding_with_checkpoints(args, env, agent_specs: Sequence[AgentSpec]) -> None:
    if not getattr(args, "old_dynamics", False):
        return

    channels_by_flag = {}
    original_featurize_type = getattr(env, "featurize_type", tuple(["ppo"] * env.all_args.num_agents))
    for flag in (False, True):
        env.old_dynamics = flag
        env.reset_featurize_type(featurize_type=original_featurize_type)
        channels_by_flag[flag] = env.observation_space[0].shape[-1]

    target_flags = set()
    for agent_id, spec in enumerate(agent_specs):
        actor_channels = infer_checkpoint_input_channels(args, spec, agent_id)
        matching_flags = [flag for flag, channels in channels_by_flag.items() if channels == actor_channels]
        if not matching_flags:
            raise RuntimeError(
                f"Could not match checkpoint input channels ({actor_channels}) to env observation channels "
                f"{channels_by_flag} for layout={args.layout_name}"
            )
        target_flags.add(matching_flags[0])

    if len(target_flags) != 1:
        raise RuntimeError(
            f"Checkpoint observation encodings disagree across agents: flags={sorted(target_flags)} "
            f"for layout={args.layout_name}"
        )

    policy_encoding_old_dynamics = target_flags.pop()
    env.old_dynamics = policy_encoding_old_dynamics
    env.reset_featurize_type(featurize_type=original_featurize_type)
    logger.info(
        "Environment alignment: transition_old_dynamics={} policy_encoding_old_dynamics={} obs_channels={}",
        args.old_dynamics,
        env.old_dynamics,
        env.observation_space[0].shape[-1],
    )


def infer_checkpoint_input_channels(args, spec: AgentSpec, agent_id: int) -> int:
    actor_path = resolve_actor_checkpoint_path(args, spec, agent_id)
    state_dict = torch.load(actor_path, map_location="cpu")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return int(value.shape[1])
    raise RuntimeError(f"Could not infer CNN input channels from checkpoint: {actor_path}")


def resolve_actor_checkpoint_path(args, spec: AgentSpec, agent_id: int) -> Path:
    if spec.actor_path is not None:
        return spec.actor_path

    share_policy = spec.policy_type == "shared"
    ckpt_role = None if share_policy else spec.role
    if spec.seed is None:
        raise ValueError("--policy_seed is required when --checkpoint_path is not provided.")
    if spec.step is None:
        raise ValueError("--checkpoint is required when --checkpoint_path is not provided.")
    models_dir = policy_loader.build_models_dir(
        layout=args.layout_name,
        algo=args.algorithm_name,
        experiment=args.experiment_name,
        seed=spec.seed,
        env_name=args.env_name,
        results_root=args.results_root,
        policy_type=spec.policy_type,
        role=ckpt_role,
    )
    candidate_names = [f"actor_periodic_{spec.step}.pt"]
    if not share_policy:
        candidate_names.insert(0, f"actor_agent{agent_id}_periodic_{spec.step}.pt")

    for name in candidate_names:
        candidate = models_dir / name
        if candidate.exists():
            return candidate

    tried = ", ".join(str(models_dir / name) for name in candidate_names)
    raise FileNotFoundError(f"Actor checkpoint not found. Tried: {tried}")


def validate_args(args) -> None:
    has_direct_paths = bool(args.checkpoint_path)
    if not has_direct_paths:
        if args.policy_seed is None:
            raise ValueError("--policy_seed is required unless --checkpoint_path is provided.")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --checkpoint_path is provided.")

    if args.policy_type == "separated":
        if int(args.num_agents) != 2:
            raise ValueError("--policy_type=separated is currently supported only for --num_agents=2.")
        if not has_direct_paths and (args.ego_role is None or args.partner_role is None):
            raise ValueError("--ego_role and --partner_role are required when --policy_type=separated.")


def build_agent_specs(args, num_agents: int) -> List[AgentSpec]:
    direct_actor_paths = resolve_direct_actor_paths(args.checkpoint_path, num_agents=num_agents)

    if args.policy_type == "shared":
        return [
            AgentSpec(
                seed=args.policy_seed,
                policy_type=args.policy_type,
                role=None,
                step=resolve_spec_step(args, direct_actor_paths[agent_id] if direct_actor_paths else None),
                actor_path=direct_actor_paths[agent_id] if direct_actor_paths else None,
            )
            for agent_id in range(num_agents)
        ]

    return [
        AgentSpec(
            seed=args.policy_seed,
            policy_type=args.policy_type,
            role=args.ego_role,
            step=resolve_spec_step(args, direct_actor_paths[0] if direct_actor_paths else None),
            actor_path=direct_actor_paths[0] if direct_actor_paths else None,
        ),
        AgentSpec(
            seed=args.policy_seed,
            policy_type=args.policy_type,
            role=args.partner_role,
            step=resolve_spec_step(args, direct_actor_paths[1] if direct_actor_paths else None),
            actor_path=direct_actor_paths[1] if direct_actor_paths else None,
        ),
    ]


def resolve_output_path(args) -> Path:
    if args.output_path:
        return Path(args.output_path).expanduser()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "json_trajectory"
    seed_label = args.policy_seed if args.policy_seed is not None else "custom"
    checkpoint_label = args.checkpoint
    if checkpoint_label is None and args.checkpoint_path:
        checkpoint_label = infer_checkpoint_step(Path(args.checkpoint_path[0]).expanduser())
    if checkpoint_label is None:
        checkpoint_label = "custom"
    file_name = f"{args.layout_name}_{seed_label}_{checkpoint_label}.json"
    return output_dir / file_name


def resolve_direct_actor_paths(checkpoint_paths: Optional[Sequence[str]], num_agents: int) -> Optional[List[Path]]:
    if not checkpoint_paths:
        return None

    paths = [Path(path).expanduser() for path in checkpoint_paths]
    if len(paths) == 1:
        paths = paths * num_agents
    elif len(paths) != num_agents:
        raise ValueError(
            f"--checkpoint_path expects either 1 path or {num_agents} paths, got {len(paths)}."
        )

    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Checkpoint path(s) not found: {missing_text}")

    return paths


def resolve_spec_step(args, actor_path: Optional[Path]) -> Optional[int]:
    if args.checkpoint is not None:
        return int(args.checkpoint)
    if actor_path is not None:
        return infer_checkpoint_step(actor_path)
    return None


def infer_checkpoint_step(path: Path) -> Optional[int]:
    match = re.search(r"(?:^|_)periodic_(\d+)$", path.stem)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"_(\d+)$", path.stem)
    if match is not None:
        return int(match.group(1))
    return None


def make_single_env(all_args, run_dir: str):
    return OvercookedNew(all_args, run_dir, evaluation=True, rank=0)


def load_policy(args, env, device: torch.device, spec: AgentSpec, agent_id: int):
    share_policy = spec.policy_type == "shared"
    if spec.actor_path is not None:
        ckpt_agent_id = None if share_policy else agent_id
        policy = policy_loader._build_policy(
            args,
            env,
            agent_id=ckpt_agent_id,
            share_policy=share_policy,
            device=device,
        )
        policy.actor.load_state_dict(torch.load(spec.actor_path, map_location=device))
        return policy, spec.step

    ckpt_role = None if share_policy else spec.role
    ckpt_agent_id = None if share_policy else agent_id
    return policy_loader.load_agent(
        all_args=args,
        envs=env,
        layout=args.layout_name,
        algo=args.algorithm_name,
        experiment=args.experiment_name,
        seed=spec.seed,
        step=spec.step,
        agent_id=ckpt_agent_id,
        share_policy=share_policy,
        results_root=args.results_root,
        device=device,
        policy_type=spec.policy_type,
        role=ckpt_role,
        load_critic=False,
    )


def generate_episode_trajectory(
    env,
    policies: Sequence[object],
    deterministic: bool,
    max_ticks: Optional[int],
) -> Dict[str, Any]:
    static_info = build_static_info(env)
    obs, available_actions = reset_eval_env(env)
    state = env.base_env.state
    score = 0
    num_agents = len(policies)
    previous_actions = ["stay"] * num_agents
    current_actions = ["stay"] * num_agents
    dynamic_states = [
        build_dynamic_state(
            env=env,
            state=state,
            score=score,
            current_actions=current_actions,
            previous_actions=previous_actions,
        )
    ]

    for policy in policies:
        policy.prep_rollout()

    rnn_states = [
        np.zeros((1, env.all_args.recurrent_N, env.all_args.hidden_size), dtype=np.float32)
        for _ in range(num_agents)
    ]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(num_agents)]

    tick_count = 0
    done = False
    while not done and (max_ticks is None or tick_count < max_ticks):
        actions: List[int] = []
        current_actions = []
        for agent_id, policy in enumerate(policies):
            avail = None if available_actions is None else available_actions[agent_id]
            action, rnn_states[agent_id] = select_action(
                policy=policy,
                obs=obs[agent_id],
                rnn_states=rnn_states[agent_id],
                masks=masks[agent_id],
                available_actions=avail,
                deterministic=deterministic,
            )
            actions.append(action)
            current_actions.append(action_index_to_name(action))

        obs, info, available_actions, done = step_eval_env(env, actions)
        score += int(sum(int(v) for v in info.get("sparse_r_by_agent", [])))
        state = env.base_env.state
        dynamic_states.append(
            build_dynamic_state(
                env=env,
                state=state,
                score=score,
                current_actions=current_actions,
                previous_actions=previous_actions,
            )
        )

        previous_actions = list(current_actions)
        tick_count += 1
        if done:
            for mask in masks:
                mask[:] = 0
        else:
            for mask in masks:
                mask[:] = 1

    return {
        "staticInfo": static_info,
        "dynamicState": dynamic_states,
    }


def reset_eval_env(env) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    reset_result = env.reset()
    if not isinstance(reset_result, tuple):
        raise ValueError(f"Unexpected reset result type: {type(reset_result).__name__}")

    if len(reset_result) == 3:
        obs, _, available_actions = reset_result
        obs_array = np.stack(obs)
        return obs_array, available_actions

    if len(reset_result) == 2:
        obs, info = reset_result
        if not isinstance(info, dict):
            raise ValueError("Expected dict info from env.reset().")
        all_agent_obs = info.get("all_agent_obs")
        if all_agent_obs is None:
            all_agent_obs = np.expand_dims(obs, axis=0)
        return np.asarray(all_agent_obs), info.get("available_actions")

    raise ValueError(f"Unexpected reset result length: {len(reset_result)}")


def step_eval_env(env, actions: Sequence[int]) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray], bool]:
    action_array = np.asarray(actions, dtype=np.int64).reshape(-1, 1)
    obs, _, _, dones, info, available_actions = env.step(action_array)
    obs_array = np.stack(obs)
    done = bool(np.all(np.asarray(dones, dtype=bool)))
    return obs_array, info, available_actions, done


def select_action(
    policy,
    obs: np.ndarray,
    rnn_states: np.ndarray,
    masks: np.ndarray,
    available_actions: Optional[np.ndarray],
    deterministic: bool,
) -> Tuple[int, np.ndarray]:
    obs_batch = np.expand_dims(obs, axis=0)
    available_actions_batch = None
    if available_actions is not None:
        available_actions_batch = np.expand_dims(available_actions, axis=0)
    action, next_rnn_states = policy.act(
        obs_batch,
        rnn_states,
        masks,
        available_actions_batch,
        deterministic=deterministic,
    )
    return int(np.asarray(to_numpy(action)).squeeze()), to_numpy(next_rnn_states)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_static_info(env) -> Dict[str, Any]:
    mdp = env.base_env.mdp
    grid = terrain_to_grid(mdp.terrain_mtx)
    width = len(grid[0]) if grid else 0
    height = len(grid)
    soup_capacity = soup_ingredient_capacity(env)
    all_orders = all_possible_orders_to_json(env)

    return {
        "layoutName": str(mdp.layout_name),
        "width": width,
        "height": height,
        "grid": grid,
        "startPlayerPositions": [
            {"id": player_id, "position": position_to_json(position)}
            for player_id, position in enumerate(mdp.start_player_positions)
        ],
        "allOrders": all_orders,
        "cookTime": resolve_cook_time(env),
        "soupIngredientCapacity": soup_capacity,
        "deliveryReward": resolve_delivery_reward(env),
    }


def build_dynamic_state(
    env,
    state,
    score: int,
    current_actions: Sequence[str],
    previous_actions: Sequence[str],
) -> Dict[str, Any]:
    objects = sorted(state.objects.values(), key=lambda obj: (obj.position[1], obj.position[0], obj.name))
    return {
        "timestep": int(getattr(env.base_env, "t", getattr(state, "timestep", 0))),
        "score": int(score),
        "orders": current_orders_to_json(env.base_env.mdp, state, soup_ingredient_capacity(env)),
        "players": [
            player_to_json(
                env=env,
                player=player,
                player_id=player_id,
                current_action=current_actions[player_id],
                previous_action=previous_actions[player_id],
            )
            for player_id, player in enumerate(state.players)
        ],
        "objects": [object_to_json(obj, env) for obj in objects],
    }


def player_to_json(env, player, player_id: int, current_action: str, previous_action: str) -> Dict[str, Any]:
    held_object = player.get_object() if player.has_object() else None
    return {
        "id": player_id,
        "position": position_to_json(player.position),
        "orientation": orientation_to_name(player.orientation),
        "heldObject": None if held_object is None else object_to_json(held_object, env),
        "action": {
            "current": current_action,
            "previous": previous_action,
        },
    }


def object_to_json(obj, env) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": str(obj.name),
        "position": position_to_json(obj.position),
    }

    if obj.name != "soup":
        return payload

    if hasattr(obj, "ingredients"):
        ingredients = [str(ingredient) for ingredient in obj.ingredients]
        is_idle = bool(getattr(obj, "is_idle", False))
        is_cooking = bool(getattr(obj, "is_cooking", False))
        is_ready = bool(getattr(obj, "is_ready", False))
        cook_time = resolve_cook_time(env)
        explicit_cook_time = getattr(obj, "_cook_time", None)
        if explicit_cook_time is not None:
            cook_time = int(explicit_cook_time)
        elif not is_idle:
            cook_time = int(obj.cook_time)
        raw_cooking_tick = getattr(obj, "_cooking_tick", getattr(obj, "cooking_tick", -1))
        cooking_tick = max(0, int(raw_cooking_tick))
    else:
        soup_type, num_items, cooking_tick = obj.state
        ingredients = [str(soup_type)] * int(num_items)
        cook_time = resolve_cook_time(env)
        is_ready = int(num_items) >= soup_ingredient_capacity(env) and int(cooking_tick) >= int(cook_time)
        is_cooking = int(num_items) >= soup_ingredient_capacity(env) and not is_ready

    payload.update(
        {
            "ingredients": ingredients,
            "isCooking": is_cooking,
            "isReady": is_ready,
            "cookTime": cook_time,
            "cookingTick": cooking_tick,
        }
    )
    return payload


def current_orders_to_json(mdp, state, soup_capacity: int) -> List[Dict[str, Any]]:
    if hasattr(state, "all_orders"):
        orders = list(state.all_orders)
        return [{"ingredients": order_to_ingredients(order, soup_capacity)} for order in orders]

    order_list = getattr(state, "order_list", None)
    if order_list is None:
        order_list = infer_old_order_names(mdp)
    return [{"ingredients": order_to_ingredients(order, soup_capacity)} for order in order_list]


def all_possible_orders_to_json(env) -> List[Dict[str, Any]]:
    state = env.base_env.state
    soup_capacity = soup_ingredient_capacity(env)

    if hasattr(state, "all_orders"):
        return [{"ingredients": order_to_ingredients(order, soup_capacity)} for order in state.all_orders]

    order_names = infer_old_order_names(env.base_env.mdp)
    unique_order_names = list(dict.fromkeys(order_names))
    return [{"ingredients": order_to_ingredients(order, soup_capacity)} for order in unique_order_names]


def infer_old_order_names(mdp) -> List[str]:
    if getattr(mdp, "start_order_list", None):
        filtered = [str(order) for order in mdp.start_order_list if str(order) in OLD_ORDER_NAMES]
        if filtered:
            return filtered

    grid = terrain_to_grid(mdp.terrain_mtx)
    flat = {cell for row in grid for cell in row}
    orders = []
    if "O" in flat:
        orders.append("onion")
    if "T" in flat:
        orders.append("tomato")
    if not orders:
        orders.append("onion")
    return orders


def order_to_ingredients(order: Any, soup_capacity: int) -> List[str]:
    if hasattr(order, "ingredients"):
        return [str(ingredient) for ingredient in order.ingredients]

    if isinstance(order, dict) and "ingredients" in order:
        return [str(ingredient) for ingredient in order["ingredients"]]

    if isinstance(order, str):
        if order == "any":
            return ["any"]
        if order in {"onion", "tomato"}:
            return [order] * int(soup_capacity)
        return [order]

    return [str(order)]


def resolve_cook_time(env) -> int:
    mdp = env.base_env.mdp
    if hasattr(mdp, "soup_cooking_time"):
        return int(mdp.soup_cooking_time)

    state = env.base_env.state
    if hasattr(state, "all_orders") and state.all_orders:
        return int(state.all_orders[0].time)

    return 20


def resolve_delivery_reward(env) -> int:
    mdp = env.base_env.mdp
    if hasattr(mdp, "delivery_reward"):
        return int(mdp.delivery_reward)

    state = env.base_env.state
    if hasattr(state, "all_orders") and state.all_orders:
        return int(state.all_orders[0].value)

    return 20


def soup_ingredient_capacity(env) -> int:
    mdp = env.base_env.mdp
    if hasattr(mdp, "num_items_for_soup"):
        return int(mdp.num_items_for_soup)
    if hasattr(mdp, "max_num_items_for_soup"):
        return int(mdp.max_num_items_for_soup)
    return 3


def terrain_to_grid(terrain) -> List[List[str]]:
    grid = []
    for row in terrain:
        if isinstance(row, str):
            grid.append(list(row))
        else:
            grid.append([str(cell) for cell in row])
    return grid


def position_to_json(position: Sequence[int]) -> Dict[str, int]:
    return {"x": int(position[0]), "y": int(position[1])}


def orientation_to_name(orientation: Sequence[int]) -> str:
    return ORIENTATION_NAME_BY_VECTOR.get(tuple(int(v) for v in orientation), "unknown")


def action_index_to_name(action_index: int) -> str:
    if action_index not in ACTION_NAME_BY_INDEX:
        raise ValueError(f"Unsupported action index: {action_index}")
    return ACTION_NAME_BY_INDEX[action_index]


if __name__ == "__main__":
    main()
