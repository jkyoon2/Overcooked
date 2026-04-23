import os
import pickle
import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import tqdm
import yaml
from loguru import logger

from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from zsceval.runner.shared.base_runner import make_trainer_policy_cls


def _resolve_path(base_path: str, path_value):
    if isinstance(path_value, dict):
        return {k: _resolve_path(base_path, v) for k, v in path_value.items()}
    if isinstance(path_value, str):
        if os.path.isabs(path_value):
            return path_value
        if base_path:
            return os.path.normpath(os.path.join(base_path, path_value))
        return os.path.normpath(path_value)
    raise RuntimeError(f"Unsupported path value type: {type(path_value)}")


def _load_policy_from_config(
    policy_key: str,
    layout_name: str,
    policy_config_path: str,
    model_path,
    featurize_type: str,
):
    policy_config = list(pickle.load(open(policy_config_path, "rb")))
    policy_args = policy_config[0]
    _, policy_cls = make_trainer_policy_cls(
        policy_args.algorithm_name,
        use_single_network=getattr(policy_args, "use_single_network", False),
    )
    policy = policy_cls(*policy_config, device=torch.device("cpu"))
    if model_path:
        policy.load_checkpoint(model_path)

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    return {
        "policy_key": policy_key,
        "policy_args": policy_args,
        "policy": policy,
        "featurize_type": featurize_type,
        "mdp": mdp,
    }


class AgentPool:
    def get_agent(self) -> Callable[[Dict, int], int]:
        raise NotImplementedError

    def _get_action(self, policy: EvalPolicy, state: np.ndarray, available_actions: np.ndarray = None) -> int:
        raise NotImplementedError

    def _process_state(self, state: Dict, featurize_type: str, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BaseOvercookedAgentPool(AgentPool):
    def __init__(self, layout_name: str, deterministic: bool = True, epsilon: float = 0.5):
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.deterministic = deterministic
        self.epsilon = epsilon

    def _build_eval_agent(self, policy_entry: Dict) -> Callable[[Dict, int], int]:
        policy = EvalPolicy(policy_entry["policy_args"], policy_entry["policy"])
        policy.reset(1, 1)
        policy.register_control_agent(0, 0)

        def _agent_call(state: Dict, pos: int) -> int:
            state_arr, available_actions = self._process_state(state, policy_entry["featurize_type"], pos)
            return self._get_action(policy, state_arr, available_actions)

        return _agent_call

    def _process_state(self, state: Dict, featurize_type: str, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        if featurize_type != "ppo":
            raise NotImplementedError(f"Unsupported featurize type: {featurize_type}")

        state_obj = state if isinstance(state, OvercookedState) else OvercookedState.from_dict(state)
        return self.mdp.lossless_state_encoding(state_obj)[pos] * 255, self._get_available_actions(state_obj)[pos]

    def _get_available_actions(self, state: OvercookedState) -> np.ndarray:
        num_agents = len(state.players)
        available_actions = np.ones((num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        interact_index = Action.ACTION_TO_INDEX["interact"]
        for agent_idx in range(num_agents):
            player = state.players[agent_idx]
            pos = player.position
            orientation = player.orientation
            for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                new_pos = Action.move_in_direction(pos, move)
                if new_pos not in self.mdp.get_valid_player_positions() and orientation == move:
                    available_actions[agent_idx, move_i] = 0

            interact_pos = Action.move_in_direction(pos, orientation)
            terrain_type = self.mdp.get_terrain_type_at_pos(interact_pos)

            if (
                terrain_type == " "
                or (
                    terrain_type == "X"
                    and (
                        (not player.has_object() and not state.has_object(interact_pos))
                        or (player.has_object() and state.has_object(interact_pos))
                    )
                )
                or (terrain_type in ["O", "T", "D"] and player.has_object())
                or (
                    terrain_type == "P"
                    and (not player.has_object() or player.get_object().name not in ["dish", "onion", "tomato"])
                )
                or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
            ):
                available_actions[agent_idx, interact_index] = 0
        return available_actions

    @torch.no_grad()
    def _get_action(self, policy: EvalPolicy, state: np.ndarray, available_actions: np.ndarray = None) -> int:
        policy.prep_rollout()
        epsilon = random.random()
        if not self.deterministic or epsilon < self.epsilon:
            return policy.step(
                np.array([state]),
                [(0, 0)],
                deterministic=False,
                available_actions=np.array([available_actions]),
            )[0]
        return policy.step(
            np.array([state]),
            [(0, 0)],
            deterministic=True,
            available_actions=np.array([available_actions]),
        )[0]


class ZSCEvalAgentPool(BaseOvercookedAgentPool):
    POLICY_POOL_PATH = os.environ.get("POLICY_POOL", "")

    def __init__(self, population_yaml_path: str, layout_name: str, deterministic: bool = True, epsilon: float = 0.5):
        super().__init__(layout_name, deterministic=deterministic, epsilon=epsilon)
        population_config = yaml.load(open(population_yaml_path, encoding="utf-8"), yaml.Loader)
        population_base_path = os.path.dirname(os.path.abspath(population_yaml_path))
        self.n_agents = len(population_config)
        self.policy_pool: Dict[str, List[Tuple]] = defaultdict(list)

        for policy_name in tqdm.tqdm(population_config, desc="Loading models..."):
            try:
                policy_config_path = _resolve_path(
                    self.POLICY_POOL_PATH or population_base_path,
                    population_config[policy_name]["policy_config_path"],
                )
                model_path = population_config[policy_name].get("model_path", None)
                if model_path:
                    model_path = _resolve_path(self.POLICY_POOL_PATH or population_base_path, model_path)
                policy_entry = _load_policy_from_config(
                    policy_key=policy_name,
                    layout_name=layout_name,
                    policy_config_path=policy_config_path,
                    model_path=model_path,
                    featurize_type=population_config[policy_name]["featurize_type"],
                )
                self.policy_pool[population_config[policy_name]["algo"]].append(
                    (
                        policy_entry["policy_key"],
                        policy_entry["policy_args"],
                        policy_entry["policy"],
                        policy_entry["featurize_type"],
                    )
                )
            except Exception as exc:
                logger.error(f"Error loading policy {policy_name}: {exc}")
                raise

    @property
    def agent_names(self) -> Dict[str, List[str]]:
        return {
            algo: [policy_tuple[0] for policy_tuple in self.policy_pool[algo]] for algo in self.policy_pool
        }

    def get_agent(self, algo: str) -> Callable[[Dict, int], int]:
        policy_tuple = random.choice(self.policy_pool[algo])
        return self._build_eval_agent(
            {
                "policy_key": policy_tuple[0],
                "policy_args": policy_tuple[1],
                "policy": policy_tuple[2],
                "featurize_type": policy_tuple[3],
            }
        )


class CheckpointAgentPool(BaseOvercookedAgentPool):
    def __init__(self, layout_name: str, agent_specs: List[Dict], deterministic: bool = True, epsilon: float = 0.0):
        super().__init__(layout_name, deterministic=deterministic, epsilon=epsilon)
        self.policy_specs: Dict[str, Dict] = {}
        self.policy_pool: Dict[str, Dict] = {}

        for agent_spec in tqdm.tqdm(agent_specs, desc="Loading EEG checkpoints..."):
            agent_id = agent_spec["id"]
            base_path = agent_spec.get("base_path", "")
            policy_config_path = _resolve_path(base_path, agent_spec["policy_config_path"])
            model_path = _resolve_path(base_path, agent_spec["model_path"])
            featurize_type = agent_spec.get("featurize_type", "ppo")
            try:
                policy_entry = _load_policy_from_config(
                    policy_key=agent_id,
                    layout_name=layout_name,
                    policy_config_path=policy_config_path,
                    model_path=model_path,
                    featurize_type=featurize_type,
                )
                self.policy_pool[agent_id] = policy_entry
                self.policy_specs[agent_id] = {
                    **agent_spec,
                    "policy_config_path": policy_config_path,
                    "model_path": model_path,
                }
            except Exception as exc:
                logger.error(f"Error loading checkpoint {agent_id}: {exc}")
                raise

    @property
    def agent_names(self) -> Dict[str, List[str]]:
        return {"fixed": list(self.policy_pool.keys())}

    def get_agent(self, agent_id: str) -> Callable[[Dict, int], int]:
        if agent_id not in self.policy_pool:
            raise KeyError(f"Unknown checkpoint agent: {agent_id}")
        return self._build_eval_agent(self.policy_pool[agent_id])
