from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    Recipe,
)

LAYOUTS = [
    "1_forced_easy",
    "1_forced_hard",
    "1_incentivized_easy",
    "1_incentivized_hard",
]

RECIPES: Dict[str, Dict[str, int]] = {
    "TTT": {"tomato": 3, "onion": 0},
    "TTO": {"tomato": 2, "onion": 1},
    "TOO": {"tomato": 1, "onion": 2},
    "OOO": {"tomato": 0, "onion": 3},
}

ACTION_MAP: Dict[str, Any] = {
    "NORTH":    Direction.NORTH,
    "SOUTH":    Direction.SOUTH,
    "EAST":     Direction.EAST,
    "WEST":     Direction.WEST,
    "STAY":     Action.STAY,
    "INTERACT": Action.INTERACT,
}

HORIZON = 400


def _recipe_ingredients(code: str) -> List[str]:
    r = RECIPES[code]
    return ["tomato"] * r["tomato"] + ["onion"] * r["onion"]


def _apply_recipe(code: str, mdp: OvercookedGridworld) -> None:
    ingredients = _recipe_ingredients(code)
    all_orders = [{"ingredients": ingredients}]
    cfg = {
        "all_orders": all_orders,
        "recipe_values": [20],
        "recipe_times": [20],
    }
    Recipe.configure(cfg)
    mdp.recipe_config = cfg
    mdp.start_all_orders = all_orders


@dataclass
class StepResult:
    state_dict: Dict[str, Any]       # serialisable game state for frontend
    raw_state: OvercookedState        # for lossless encoding in recorder
    joint_action: Tuple[int, int]     # (action_idx_p0, action_idx_p1)
    reward: int                       # total reward this step
    done: bool
    step_index: int


class TwoPlayerEngine:
    """2-player Overcooked engine. No AI — both players are human."""

    def __init__(self, layout_name: str, recipe_code: str) -> None:
        self.layout_name = layout_name
        self.recipe_code = recipe_code
        self.mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(
            layout_name, old_dynamics=True
        )
        _apply_recipe(recipe_code, self.mdp)
        self._state: Optional[OvercookedState] = None
        self._step_index = 0
        self._score = 0

    def reset(self) -> Dict[str, Any]:
        _apply_recipe(self.recipe_code, self.mdp)   # guard cross-trial leakage
        self._state = self.mdp.get_standard_start_state()
        self._step_index = 0
        self._score = 0
        return self._serialize()

    def step(self, action_p0: Any, action_p1: Any) -> StepResult:
        assert self._state is not None
        a0 = _norm(action_p0)
        a1 = _norm(action_p1)

        prev_state = self._state
        new_state, infos = self.mdp.get_state_transition(self._state, (a0, a1))

        reward = int(sum(infos["sparse_reward_by_agent"]))
        self._score += reward
        self._step_index += 1
        self._state = new_state

        ai0 = _action_index(a0)
        ai1 = _action_index(a1)

        return StepResult(
            state_dict=self._serialize(),
            raw_state=new_state,
            joint_action=(ai0, ai1),
            reward=reward,
            done=self._step_index >= HORIZON,
            step_index=self._step_index,
        )

    def lossless_encoding(self) -> Tuple[Any, Any]:
        assert self._state is not None
        return self.mdp.lossless_state_encoding(self._state, horizon=HORIZON)

    def _serialize(self) -> Dict[str, Any]:
        s = self._state
        players = []
        for p in s.players:
            held = None
            held_ings: List[str] = []
            if p.held_object is not None:
                if hasattr(p.held_object, "ingredients"):
                    held = "soup"
                    held_ings = list(p.held_object.ingredients)
                else:
                    held = p.held_object.name
            players.append({
                "position": list(p.position),
                "orientation": list(p.orientation),
                "held_object": held,
                "held_object_ingredients": held_ings,
            })

        objects = []
        for pos, obj in s.objects.items():
            if hasattr(obj, "ingredients"):
                st = "ready" if obj.is_ready else ("cooking" if obj.is_cooking else "idle")
                objects.append({
                    "name": "soup",
                    "position": list(pos),
                    "soup_state": st,
                    "ingredients": list(obj.ingredients),
                })
            else:
                objects.append({"name": obj.name, "position": list(pos)})

        return {
            "players": players,
            "objects": objects,
            "score": self._score,
            "step_index": self._step_index,
            "time_remaining": HORIZON - self._step_index,
            "layout_name": self.layout_name,
        }


def _norm(action: Any) -> Any:
    if isinstance(action, str):
        return ACTION_MAP.get(action, Action.STAY)
    return action


def _action_index(action: Any) -> int:
    for i, a in enumerate(Action.ALL_ACTIONS):
        if a == action:
            return i
    return 4  # STAY fallback
