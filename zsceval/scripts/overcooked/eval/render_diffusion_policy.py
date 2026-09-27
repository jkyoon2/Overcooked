#!/usr/bin/env python
"""Render trained diffusion policies in Overcooked and save GIF rollouts."""

import argparse
import glob
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from zsceval.algorithms.diffusion_policy.policy import DiffusionOvercookedPolicy
from zsceval.algorithms.diffusion_policy.scheduler import DDPMScheduler
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import Recipe


RECIPE_TO_ID = {"TTT": 0, "TTO": 1, "TOO": 2, "OOO": 3, "NULL": 4}
RECIPE_INGREDIENTS = {
    "TTT": ["tomato", "tomato", "tomato"],
    "TTO": ["tomato", "tomato", "onion"],
    "TOO": ["tomato", "onion", "onion"],
    "OOO": ["onion", "onion", "onion"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="1_forced_easy")
    parser.add_argument("--recipe", choices=sorted(RECIPE_TO_ID), default="TTT")
    parser.add_argument("--agent0-checkpoint", required=True)
    parser.add_argument("--agent1-checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--old-dynamics", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]
    metadata = checkpoint["metadata"]

    policy = DiffusionOvercookedPolicy(
        obs_shape=tuple(metadata["obs_shape"]),
        n_obs=data_cfg["n_obs"],
        horizon=data_cfg["horizon"],
        n_action_steps=data_cfg["n_action_steps"],
        action_dim=data_cfg["action_dim"],
        obs_emb_dim=model_cfg["obs_emb_dim"],
        goal_emb_dim=model_cfg["goal_emb_dim"],
        down_dims=model_cfg["down_dims"],
        num_diffusion_steps=diffusion_cfg["num_diffusion_steps"],
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    if "scheduler" in checkpoint:
        policy.scheduler = DDPMScheduler.from_config(checkpoint["scheduler"]["config"])
    policy.to(device)
    policy.eval()
    return policy, config


def make_env_args(args):
    return argparse.Namespace(
        layout_name=args.layout,
        episode_length=args.episode_length,
        num_agents=2,
        use_render=True,
        n_render_rollout_threads=1,
        algorithm_name="diffusion_policy",
        agent_policy_names=["diffusion", "diffusion"],
        overcooked_version="old" if args.old_dynamics else "new",
        old_dynamics=args.old_dynamics,
        reward_shaping_role="individual",
        reward_shaping_roles="individual,individual",
        reward_shaping_roles_list=["individual", "individual"],
        reward_shaping_horizon=1e8,
        initial_reward_shaping_factor=1.0,
        reward_shaping_factor=1.0,
        random_start_prob=0.0,
        use_hsp=False,
        use_phi=False,
        use_agent_policy_id=False,
        random_index=False,
        use_available_actions=True,
        onion_regrab_window=8,
        idle_backtrack_window=2,
        store_traj=False,
        use_random_terrain_state=False,
        use_random_player_pos=False,
        num_initial_state=5,
        replay_return_threshold=0.75,
        use_task_v_out=False,
        policy_mode="separated",
    )


def apply_recipe(env, recipe):
    if recipe == "NULL":
        return
    ingredients = RECIPE_INGREDIENTS[recipe]
    recipe_config = {
        "all_orders": [{"ingredients": ingredients}],
        "recipe_values": [20],
        "recipe_times": [20],
    }
    Recipe.configure(recipe_config)
    all_orders = recipe_config["all_orders"]
    env.base_mdp.recipe_config = recipe_config
    env.base_mdp.start_all_orders = all_orders
    env.base_env.mdp.recipe_config = recipe_config
    env.base_env.mdp.start_all_orders = all_orders


def normalized_agent_obs(info):
    # Overcooked PPO observations are lossless encodings multiplied by 255.
    # The diffusion policy was trained on the underlying lossless scale.
    return info["all_agent_obs"].astype(np.float32) / 255.0


@torch.no_grad()
def plan_chunk(policy, obs_history, z_goal_id, device, num_inference_steps, guidance_scale):
    obs_hist = torch.from_numpy(np.stack(obs_history, axis=0)).float().unsqueeze(0).to(device)
    z_goal = torch.tensor([z_goal_id], dtype=torch.long, device=device)
    _, logits = policy.sample_action_chunk(
        obs_hist,
        z_goal,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        return_logits=True,
    )
    return logits.squeeze(0).detach().cpu().numpy()


def masked_argmax(logits, available_actions):
    masked_logits = np.asarray(logits, dtype=np.float32).copy()
    mask = np.asarray(available_actions).astype(bool)
    masked_logits[~mask] = -1.0e9
    return int(masked_logits.argmax())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    policies = []
    configs = []
    for ckpt in [args.agent0_checkpoint, args.agent1_checkpoint]:
        policy, config = load_policy(ckpt, device)
        policies.append(policy)
        configs.append(config)

    n_obs = configs[0]["data"]["n_obs"]
    n_action_steps = configs[0]["data"]["n_action_steps"]
    num_inference_steps = args.num_inference_steps
    if num_inference_steps is None:
        num_inference_steps = configs[0]["diffusion"].get("inference_steps", 50)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            f"results/Overcooked/{args.layout}/diffusion_policy/"
            f"rollouts_{args.recipe}_seed{args.seed}"
        )
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = Overcooked(make_env_args(args), run_dir=str(run_dir), evaluation=True, rank=0)
    apply_recipe(env, args.recipe)
    z_goal_id = RECIPE_TO_ID[args.recipe]

    gif_paths = []
    for episode in range(args.episodes):
        _, info = env.reset(reset_choose=True)
        agent_obs = normalized_agent_obs(info)
        histories = [
            deque([agent_obs[agent_id]] * n_obs, maxlen=n_obs) for agent_id in range(2)
        ]
        chunk_logits = [None, None]
        chunk_offsets = [n_action_steps, n_action_steps]

        sparse_return = 0.0
        for _ in range(args.episode_length):
            available_actions = info["available_actions"]
            actions = []
            for agent_id in range(2):
                if chunk_offsets[agent_id] >= n_action_steps:
                    chunk_logits[agent_id] = plan_chunk(
                        policies[agent_id],
                        histories[agent_id],
                        z_goal_id,
                        device,
                        num_inference_steps,
                        args.guidance_scale,
                    )
                    chunk_offsets[agent_id] = 0
                action = masked_argmax(
                    chunk_logits[agent_id][chunk_offsets[agent_id]],
                    available_actions[agent_id],
                )
                actions.append(action)

            action_array = np.asarray([[actions[0]], [actions[1]]], dtype=np.int64)
            _, _, _, done, info, _ = env.step(action_array)
            sparse_return += float(info.get("sparse_r", 0.0))

            agent_obs = normalized_agent_obs(info)
            for agent_id in range(2):
                histories[agent_id].append(agent_obs[agent_id])
                chunk_offsets[agent_id] += 1

            if all(done):
                break

        traj_dir = run_dir / "gifs" / f"rank_0_traj_num_{env.traj_num}"
        matches = sorted(glob.glob(str(traj_dir / "reward_*.gif")))
        gif_path = matches[-1] if matches else str(traj_dir)
        gif_paths.append(gif_path)
        print(f"episode={episode + 1} sparse_return={sparse_return:.3f} gif={gif_path}")

    print("Saved GIFs:")
    for path in gif_paths:
        print(path)


if __name__ == "__main__":
    main()
