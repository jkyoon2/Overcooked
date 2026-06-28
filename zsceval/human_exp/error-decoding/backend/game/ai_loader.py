from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, cast

import numpy as np
import torch

from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
)


CheckpointId = Literal[
    "tomato",
    "onion",
    "ttt",
    "tto",
    "too",
    "ooo",
    "tto_mep1",
    "tto_mep2",
    "tto_mep3",
    "tto_mep4",
    "too_mep1",
    "too_mep2",
    "too_mep3",
    "too_mep4",
]
_LAYOUT_BY_CHECKPOINT_ID: Dict[str, str] = {
    "tomato": "ttt",
    "onion": "ooo",
    "ttt": "ttt",
    "tto": "tto",
    "too": "too",
    "ooo": "ooo",
    "tto_mep1": "tto",
    "tto_mep2": "tto",
    "tto_mep3": "tto",
    "tto_mep4": "tto",
    "too_mep1": "too",
    "too_mep2": "too",
    "too_mep3": "too",
    "too_mep4": "too",
}
_DEFAULT_RELATIVE_RUN = Path("shared") / "rmappo" / "hsp-S1" / "seed1"
_PERIODIC_MODEL_RE = re.compile(r"^(actor|critic)_periodic_(\d+)\.pt$")
_ADAPTIVE_CONFIG_OVERRIDES = (
    "use_agent_policy_id",
    "num_v_out",
    "use_task_v_out",
    "use_policy_vhead",
    "use_proper_time_limits",
    "entropy_coefs",
    "entropy_coef_horizons",
    "use_peb",
    "data_parallel",
)


@dataclass(frozen=True)
class _ExplicitCheckpointSpec:
    layout_name: str
    run_dir: Path
    config_path: Path
    actor_path: Path
    critic_path: Optional[Path]
    step: int
    runtime_config_path: Optional[Path] = None


def _mep_spec(layout: str, variant: int) -> _ExplicitCheckpointSpec:
    run_dir = Path(f"results/Overcooked/{layout}/shared/mep/mep-S1-s5/seed1")
    return _ExplicitCheckpointSpec(
        layout_name=layout,
        run_dir=run_dir,
        config_path=run_dir / "policy_config.pkl",
        actor_path=run_dir / "models" / f"mep{variant}" / "actor_periodic_10000000.pt",
        critic_path=None,
        step=10_000_000,
    )


_EXPLICIT_CHECKPOINT_SPECS = {
    f"{layout}_mep{variant}": _mep_spec(layout, variant)
    for layout in ("tto", "too")
    for variant in (1, 2, 3, 4)
}


@dataclass(frozen=True)
class _CheckpointPaths:
    layout_name: str
    run_dir: Path
    config_path: Path
    actor_path: Path
    critic_path: Optional[Path]
    step: int
    runtime_config_path: Optional[Path] = None


class HSPPolicy:
    """
    Loads a trained HSP/rMAPPO checkpoint and returns deterministic argmax actions.

    The current engine policy contract has no episode lifecycle hook, so inference
    uses a zero recurrent state on each call. That keeps act(state) a pure,
    deterministic function of the current observation.
    """

    def __init__(self, checkpoint_id: CheckpointId, device: str = "cpu") -> None:
        self.checkpoint_id = checkpoint_id
        self.device = torch.device(device)
        self.agent_index = 1  # AI plays as player index 1
        self.paths = _resolve_checkpoint_paths(checkpoint_id)
        self.layout_name = self.paths.layout_name

        policy_config = _load_policy_config(self.paths)
        if not isinstance(policy_config, tuple) or len(policy_config) != 4:
            raise ValueError(
                f"Expected policy_config.pkl to contain "
                f"(args, obs_space, share_obs_space, act_space), got "
                f"{type(policy_config)!r}"
            )

        args, obs_space, share_obs_space, act_space = policy_config
        algorithm_name = getattr(args, "algorithm_name", None)
        # MEP / HSP training pipelines reuse R_MAPPOPolicy as the actor backbone.
        if algorithm_name not in {"rmappo", "mappo", "mep", "hsp"}:
            raise ValueError(
                f"Unsupported policy algorithm for HSPPolicy: {algorithm_name!r}"
            )

        self.args = args
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space
        self.recurrent_n = int(getattr(args, "recurrent_N"))
        self.hidden_size = int(getattr(args, "hidden_size"))
        self.episode_length = int(getattr(args, "episode_length", 400))
        self.old_dynamics = bool(getattr(args, "old_dynamics", False))
        self.use_available_actions = bool(getattr(args, "use_available_actions", True))

        self._mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self._policy = R_MAPPOPolicy(
            args,
            obs_space,
            share_obs_space,
            act_space,
            device=self.device,
        )
        self._policy.actor.load_state_dict(
            torch.load(self.paths.actor_path, map_location=self.device)
        )
        if self.paths.critic_path is not None:
            self._policy.critic.load_state_dict(
                torch.load(self.paths.critic_path, map_location=self.device)
            )
        self._policy.prep_rollout()

    @torch.no_grad()
    def act(self, state: OvercookedState) -> Any:
        obs, available_actions = self._preprocess_obs(state)
        rnn_states = np.zeros(
            (1, self.recurrent_n, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((1, 1), dtype=np.float32)
        actions, _ = self._policy.act(
            obs,
            rnn_states,
            masks,
            available_actions=available_actions,
            deterministic=True,
        )
        action_index = int(actions.detach().cpu().item())
        return Action.INDEX_TO_ACTION[action_index]

    def _preprocess_obs(
        self,
        state: OvercookedState,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if state is None:
            raise ValueError("HSPPolicy.act() requires a non-null OvercookedState")
        if len(state.players) <= self.agent_index:
            raise ValueError(
                f"AI agent index {self.agent_index} is unavailable for "
                f"{len(state.players)} players"
            )

        all_obs = self._mdp.lossless_state_encoding(
            state,
            horizon=self.episode_length,
            old_dynamics=self.old_dynamics,
        )
        obs = np.asarray(all_obs[self.agent_index] * 255, dtype=np.float32)
        expected_shape = tuple(cast(Tuple[int, ...], self.obs_space.shape))
        if obs.shape != expected_shape:
            raise ValueError(
                f"Observation shape mismatch for checkpoint {self.checkpoint_id!r}: "
                f"got {obs.shape}, expected {expected_shape}. "
                f"Use layout {self.layout_name!r} with this checkpoint."
            )

        available_actions = None
        if self.use_available_actions:
            available_actions = self._get_available_actions(state)[
                self.agent_index : self.agent_index + 1
            ]
        return np.expand_dims(obs, axis=0), available_actions

    def _get_available_actions(self, state: OvercookedState) -> np.ndarray:
        available_actions = np.ones(
            (len(state.players), len(Action.ALL_ACTIONS)),
            dtype=np.uint8,
        )
        interact_index = Action.ACTION_TO_INDEX[Action.INTERACT]
        valid_positions = self._mdp.get_valid_player_positions()

        for agent_idx, player in enumerate(state.players):
            position = player.position
            orientation = player.orientation
            for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                new_position = Action.move_in_direction(position, move)
                if new_position not in valid_positions and orientation == move:
                    available_actions[agent_idx, move_i] = 0

            interact_position = Action.move_in_direction(position, orientation)
            terrain_type = self._mdp.get_terrain_type_at_pos(interact_position)
            if (
                terrain_type == " "
                or (
                    terrain_type == "X"
                    and (
                        (not player.has_object() and not state.has_object(interact_position))
                        or (player.has_object() and state.has_object(interact_position))
                    )
                )
                or (terrain_type in ["O", "T", "D"] and player.has_object())
                or (
                    terrain_type == "S"
                    and (not player.has_object() or player.get_object().name != "soup")
                )
            ):
                available_actions[agent_idx, interact_index] = 0

            if terrain_type == "P":
                if player.has_object():
                    obj = player.get_object()
                    if obj.name not in ["dish", "onion", "tomato"]:
                        available_actions[agent_idx, interact_index] = 0
                elif (
                    self.old_dynamics
                    or not state.has_object(interact_position)
                    or state.get_object(interact_position).is_ready
                ):
                    available_actions[agent_idx, interact_index] = 0

        return available_actions


def load_policy(checkpoint_id: str, device: str = "cpu") -> HSPPolicy:
    """
    Factory for the engine/tick loop. Loads only trained HSP/rMAPPO policies.

    "tomato" and "onion" are experiment aliases for the ttt and ooo HSP
    checkpoints. The concrete layout ids are also accepted for inspection and
    tests.
    """
    if checkpoint_id not in _LAYOUT_BY_CHECKPOINT_ID:
        expected = ", ".join(sorted(_LAYOUT_BY_CHECKPOINT_ID))
        raise ValueError(
            f"Unknown AI checkpoint id {checkpoint_id!r}; expected one of {expected}"
        )
    return HSPPolicy(cast(CheckpointId, checkpoint_id), device=device)


def checkpoint_layout(checkpoint_id: str) -> str:
    if checkpoint_id not in _LAYOUT_BY_CHECKPOINT_ID:
        expected = ", ".join(sorted(_LAYOUT_BY_CHECKPOINT_ID))
        raise ValueError(
            f"Unknown AI checkpoint id {checkpoint_id!r}; expected one of {expected}"
        )
    return _LAYOUT_BY_CHECKPOINT_ID[checkpoint_id]


def _resolve_checkpoint_paths(checkpoint_id: str) -> _CheckpointPaths:
    repo_root = Path(__file__).resolve().parents[4]
    explicit_spec = _EXPLICIT_CHECKPOINT_SPECS.get(checkpoint_id)
    if explicit_spec is not None:
        paths = _CheckpointPaths(
            layout_name=explicit_spec.layout_name,
            run_dir=repo_root / explicit_spec.run_dir,
            config_path=repo_root / explicit_spec.config_path,
            actor_path=repo_root / explicit_spec.actor_path,
            critic_path=(
                repo_root / explicit_spec.critic_path
                if explicit_spec.critic_path is not None
                else None
            ),
            step=explicit_spec.step,
            runtime_config_path=(
                repo_root / explicit_spec.runtime_config_path
                if explicit_spec.runtime_config_path is not None
                else None
            ),
        )
        _validate_checkpoint_files(paths)
        return paths

    layout_name = checkpoint_layout(checkpoint_id)
    run_dir = repo_root / "results" / "Overcooked" / layout_name / _DEFAULT_RELATIVE_RUN
    config_path = run_dir / "policy_config.pkl"
    models_dir = run_dir / "models"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing HSP policy config: {config_path}")
    actor_path, critic_path, step = _find_latest_model_pair(models_dir)
    return _CheckpointPaths(
        layout_name=layout_name,
        run_dir=run_dir,
        config_path=config_path,
        actor_path=actor_path,
        critic_path=critic_path,
        step=step,
    )


def _load_policy_config(paths: _CheckpointPaths) -> Tuple[Any, Any, Any, Any]:
    with paths.config_path.open("rb") as config_file:
        policy_config = pickle.load(config_file)
    if paths.runtime_config_path is None:
        return cast(Tuple[Any, Any, Any, Any], policy_config)

    base_args, _, _, _ = policy_config
    with paths.runtime_config_path.open("rb") as runtime_config_file:
        runtime_args, obs_space, share_obs_space, act_space = pickle.load(
            runtime_config_file
        )
    for attribute in _ADAPTIVE_CONFIG_OVERRIDES:
        setattr(base_args, attribute, getattr(runtime_args, attribute))
    return base_args, obs_space, share_obs_space, act_space


def _validate_checkpoint_files(paths: _CheckpointPaths) -> None:
    for path in (
        paths.config_path,
        paths.actor_path,
        paths.critic_path,
        paths.runtime_config_path,
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Missing HSP checkpoint file: {path}")


def _find_latest_model_pair(models_dir: Path) -> Tuple[Path, Path, int]:
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing HSP models directory: {models_dir}")

    actor_steps: Dict[int, Path] = {}
    critic_steps: Dict[int, Path] = {}
    for path in models_dir.iterdir():
        match = _PERIODIC_MODEL_RE.match(path.name)
        if match is None:
            continue
        kind, step_raw = match.groups()
        step = int(step_raw)
        if kind == "actor":
            actor_steps[step] = path
        else:
            critic_steps[step] = path

    common_steps = sorted(set(actor_steps).intersection(critic_steps))
    if not common_steps:
        raise FileNotFoundError(
            f"No paired actor/critic periodic checkpoints in {models_dir}"
        )

    latest_step = common_steps[-1]
    return actor_steps[latest_step], critic_steps[latest_step], latest_step
