from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from zsceval.analysis.core import metrics
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked as OvercookedOld
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as OvercookedNew
from zsceval.utils.train_util import setup_seed


@dataclass
class EpisodeSummary:
    episode_return: float
    violation_count: int
    violation_rate: float
    episode_length: int


def make_eval_env(all_args, run_dir: str):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "Overcooked":
                raise NotImplementedError(f"Unsupported env: {all_args.env_name}")
            if all_args.overcooked_version == "old":
                env = OvercookedOld(all_args, run_dir, evaluation=True, rank=rank)
            else:
                env = OvercookedNew(all_args, run_dir, evaluation=True, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocDummyBatchVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
        all_args.dummy_batch_size,
    )


def run_pair_episodes(
    agent_a,
    agent_b,
    envs,
    num_episodes: int,
    all_args,
    deterministic: bool = True,
    collect_trajectory: bool = False,
    ego_agent_id: int = 0,
    partner_agent_id: int = 1,
    flatten_obs: bool = True,
    collect_step_data: bool = False,
    seed: Optional[int] = None,
) -> Tuple[List[EpisodeSummary], List[Dict[str, Any]]]:
    if seed is not None:
        setup_seed(seed)

    target_error_types = getattr(all_args, "target_error_types", None)

    if hasattr(agent_a, "prep_rollout"):
        agent_a.prep_rollout()
    if hasattr(agent_b, "prep_rollout"):
        agent_b.prep_rollout()

    obs, available_actions, infos = _reset_envs(envs)
    n_envs = obs.shape[0]
    num_agents = obs.shape[1]
    if num_agents <= max(ego_agent_id, partner_agent_id):
        raise ValueError(f"Invalid agent ids for num_agents={num_agents}.")

    rnn_states_a = np.zeros((n_envs, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    rnn_states_b = np.zeros_like(rnn_states_a)
    masks_a = np.ones((n_envs, 1), dtype=np.float32)
    masks_b = np.ones_like(masks_a)

    step_infos: List[List[dict]] = [[] for _ in range(n_envs)]
    episode_counts = [0 for _ in range(n_envs)]
    summaries: List[EpisodeSummary] = []
    trajectory_records: List[Dict[str, Any]] = []
    episode_ids = [0 for _ in range(n_envs)]
    timesteps = [0 for _ in range(n_envs)]

    with torch.no_grad():
        while len(summaries) < num_episodes:
            obs_a = obs[:, ego_agent_id]
            obs_b = obs[:, partner_agent_id]
            avail_a = available_actions[:, ego_agent_id] if available_actions is not None else None
            avail_b = available_actions[:, partner_agent_id] if available_actions is not None else None

            action_a, rnn_states_a = agent_a.act(obs_a, rnn_states_a, masks_a, avail_a, deterministic=deterministic)
            action_b, rnn_states_b = agent_b.act(obs_b, rnn_states_b, masks_b, avail_b, deterministic=deterministic)

            action_a = _ensure_action_shape(_t2n(action_a))
            action_b = _ensure_action_shape(_t2n(action_b))
            actions = np.stack([action_a, action_b], axis=1)

            obs_next, _, _, dones, infos, available_actions = envs.step(actions)
            obs_next = _normalize_obs(obs_next, infos)

            done_flags = _normalize_dones(dones)
            for env_i, info in enumerate(infos):
                step_infos[env_i].append(info)
                if collect_trajectory:
                    state = obs[env_i, ego_agent_id]
                    if flatten_obs:
                        state = state.reshape(-1)
                    record = {
                        "state": state,
                        "violation": metrics.is_safety_violation(info, target_error_types=target_error_types),
                        "episode_id": episode_ids[env_i],
                        "timestep": timesteps[env_i],
                        "env_index": env_i,
                    }
                    if collect_step_data:
                        ego_obs = obs[env_i, ego_agent_id]
                        partner_obs = obs[env_i, partner_agent_id]
                        next_ego_obs = obs_next[env_i, ego_agent_id]
                        next_partner_obs = obs_next[env_i, partner_agent_id]
                        if flatten_obs:
                            ego_obs = ego_obs.reshape(-1)
                            partner_obs = partner_obs.reshape(-1)
                            next_ego_obs = next_ego_obs.reshape(-1)
                            next_partner_obs = next_partner_obs.reshape(-1)
                        record.update(
                            {
                                "obs_ego": ego_obs,
                                "obs_partner": partner_obs,
                                "next_obs_ego": next_ego_obs,
                                "next_obs_partner": next_partner_obs,
                                "action_ego": int(np.asarray(action_a[env_i]).squeeze()),
                                "action_partner": int(np.asarray(action_b[env_i]).squeeze()),
                                "done": bool(done_flags[env_i]),
                                "event_infos": copy.deepcopy(info.get("event_infos")),
                                "shaped_info_timestep": copy.deepcopy(info.get("shaped_info_timestep")),
                                "shaped_r_by_agent": copy.deepcopy(info.get("shaped_r_by_agent")),
                                "sparse_r_by_agent": copy.deepcopy(info.get("sparse_r_by_agent")),
                            }
                        )
                    trajectory_records.append(record)
                timesteps[env_i] += 1

                if done_flags[env_i]:
                    summary = _summarize_episode(step_infos[env_i], info, target_error_types=target_error_types)
                    summaries.append(summary)
                    step_infos[env_i] = []
                    episode_counts[env_i] += 1
                    episode_ids[env_i] += 1
                    timesteps[env_i] = 0
                    rnn_states_a[env_i] = 0
                    rnn_states_b[env_i] = 0
                    masks_a[env_i] = 0
                    masks_b[env_i] = 0
                else:
                    masks_a[env_i] = 1
                    masks_b[env_i] = 1

            obs = obs_next

    return summaries[:num_episodes], trajectory_records


def _summarize_episode(
    step_infos: List[dict],
    last_info: dict,
    target_error_types: Optional[List[str]] = None,
) -> EpisodeSummary:
    safety = metrics.calculate_safety_violation(step_infos, target_error_types=target_error_types)
    episode_return = metrics.extract_episode_return(last_info)
    return EpisodeSummary(
        episode_return=episode_return,
        violation_count=safety["count"],
        violation_rate=safety["rate"],
        episode_length=len(step_infos),
    )


def _reset_envs(envs):
    reset_result = envs.reset()
    if len(reset_result) == 2:
        obs, infos = reset_result
        obs = _normalize_obs(obs, infos)
        available_actions = None
        if infos and isinstance(infos[0], dict) and "available_actions" in infos[0]:
            available_actions = np.array([info["available_actions"] for info in infos])
        return obs, available_actions, infos
    if len(reset_result) == 4:
        obs, _, available_actions, infos = reset_result
        obs = _normalize_obs(obs, infos)
        return obs, available_actions, infos
    obs, _, available_actions = reset_result
    obs = np.stack(obs)
    return obs, available_actions, None


def _normalize_obs(obs, infos):
    if infos and isinstance(infos[0], dict) and "all_agent_obs" in infos[0]:
        return np.array([info["all_agent_obs"] for info in infos])
    return np.stack(obs)


def _normalize_dones(dones) -> np.ndarray:
    if isinstance(dones, np.ndarray):
        if dones.ndim > 1:
            # 에이전트 별 done을 하나로 합침 (모두 True여야 에피소드 종료인 경우)
            done_flags = np.all(dones, axis=1)
            # np.atleast_1d를 감싸서 스칼라(0-d)가 되는 것을 막음
            return np.atleast_1d(np.squeeze(done_flags)).astype(bool)
        return np.atleast_1d(np.squeeze(dones)).astype(bool)
    return np.atleast_1d(np.array(dones, dtype=bool))


def _t2n(x):
    return x.detach().cpu().numpy()


def _ensure_action_shape(actions: np.ndarray) -> np.ndarray:
    if actions.ndim == 1:
        return actions[:, None]
    return actions
