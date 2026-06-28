from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    Recipe,
)

from ..data.schema import GameState, ObjectSnapshot, PlayerSnapshot
from .ai_loader import HSPPolicy, load_policy
from .events import GameEvent

# Frontend sends action names; we map to the tuple/string values the MDP expects (PI-6).
ACTION_MAP: Dict[str, Any] = {
    "NORTH": Direction.NORTH,   # (0, -1)
    "SOUTH": Direction.SOUTH,   # (0, 1)
    "EAST": Direction.EAST,     # (1, 0)
    "WEST": Direction.WEST,     # (-1, 0)
    "STAY": Action.STAY,        # (0, 0)
    "INTERACT": Action.INTERACT,  # "interact"
}

# Pickup event keys present in event_infos (subset of EVENT_TYPES relevant to our GameEvent).
_PICKUP_EVENT_KEYS = ("onion_pickup", "tomato_pickup", "dish_pickup", "soup_pickup")

# Maps per-soup reward → recipe type string.
_REWARD_TO_RECIPE: Dict[int, str] = {5: "ttt", 10: "tto", 15: "too", 20: "ooo"}

# The checkpoint layouts share one kitchen geometry. Their layout files contain
# different training orders, but the experiment must allow every recipe in every
# trial so only the AI policy changes between conditions.
_EXPERIMENT_RECIPES = (
    (["tomato", "tomato", "tomato"], 5),
    (["tomato", "tomato", "onion"], 10),
    (["tomato", "onion", "onion"], 15),
    (["onion", "onion", "onion"], 20),
)


@dataclass
class StepResult:
    next_state: GameState
    rewards: Tuple[float, float]
    done: bool
    events: List[GameEvent]
    step_index: int


class OvercookedEngine:
    """
    Thin synchronous wrapper around OvercookedGridworld.

    Constraints (CLAUDE.md Section 4):
    - C1: step() is fully synchronous — no async, no I/O, no network.
    - C2: no condition info leaked through this class.
    - C5: events emitted here in a form EEGMarkerBridge can later consume.
    """

    def __init__(
        self,
        layout_name: str,
        tomato_reward: int = 5,
        onion_reward: int = 20,
        seed: Optional[int] = None,
        ai_checkpoint_id: Optional[str] = None,
    ) -> None:
        self.layout_name = layout_name
        self.tomato_reward = tomato_reward
        self.onion_reward = onion_reward
        self.seed = seed
        self.ai_checkpoint_id = ai_checkpoint_id
        self._ai_policy: Optional[HSPPolicy] = (
            load_policy(ai_checkpoint_id) if ai_checkpoint_id is not None else None
        )

        # old_dynamics=True: pot auto-starts cooking when 3 ingredients are added
        # (matches HSP training convention; removes the extra empty-hand SPACE interact).
        self.mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(
            layout_name, old_dynamics=True
        )

        # B-cat-A: Recipe.configure() sets class-level state. Must be called AFTER
        # from_layout_name() because that method's _configure_recipes() would otherwise
        # override our per-ingredient values with the layout file's recipe_values.
        self._apply_recipe_config()
        self._state: Optional[OvercookedState] = None
        self._step_index: int = 0
        self._score: float = 0.0
        self.max_steps: int = 400  # 40 s at 10 Hz

        if seed is not None:
            random.seed(seed)

    def _apply_recipe_config(self) -> None:
        # Recipe.configure() is global, while start_all_orders is stored on each
        # MDP instance. Keep both in sync on every reset to prevent checkpoint
        # layout metadata from leaking into experiment scoring.
        all_orders = [
            {"ingredients": list(ingredients)}
            for ingredients, _ in _EXPERIMENT_RECIPES
        ]
        recipe_config = {
            "all_orders": all_orders,
            "recipe_values": [value for _, value in _EXPERIMENT_RECIPES],
            "recipe_times": [20] * len(_EXPERIMENT_RECIPES),
        }
        Recipe.configure(recipe_config)
        self.mdp.recipe_config = recipe_config
        self.mdp.start_all_orders = all_orders

    def reset(self) -> GameState:
        """Start a fresh episode. Returns the initial serialized state."""
        self._apply_recipe_config()  # guard against cross-trial Recipe leakage
        self._state = self.mdp.get_standard_start_state()
        self._step_index = 0
        self._score = 0.0
        return self._serialize_state()

    def step(self, player_action: Any, ai_action: Optional[Any] = None) -> StepResult:
        """
        Advance one game tick. Synchronous only — no async, no I/O (C1).

        player_action: frontend action name or ZSC-Eval Action value.
        ai_action: optional explicit AI action. If omitted, the engine-owned
        policy selects the AI action synchronously.

        Backward compatibility: passing a legacy `(player_action, ai_action)`
        tuple as the first argument is still supported for Task 2 tests.
        """
        assert self._state is not None, "Call reset() before step()"

        if ai_action is None and _is_legacy_joint_action(player_action):
            player_action, ai_action = player_action

        player_action = _normalize_action(player_action)
        if ai_action is None:
            if self._ai_policy is None:
                raise RuntimeError(
                    "Engine.step() requires ai_action when no ai_checkpoint_id "
                    "is configured"
                )
            ai_action = self._ai_policy.act(self._state)
        else:
            ai_action = _normalize_action(ai_action)

        prev_soup_states = _snapshot_soups(self._state)

        new_state, infos = self.mdp.get_state_transition(
            self._state,
            (player_action, ai_action),
        )

        rewards: Tuple[float, float] = (
            float(infos["sparse_reward_by_agent"][0]),
            float(infos["sparse_reward_by_agent"][1]),
        )
        self._score += rewards[0] + rewards[1]
        self._step_index += 1

        events = _detect_events(
            infos,
            prev_soup_states,
            new_state,
            self._step_index,
            self.tomato_reward,
            self.onion_reward,
        )

        self._state = new_state
        done = self._step_index >= self.max_steps

        return StepResult(
            next_state=self._serialize_state(),
            rewards=rewards,
            done=done,
            events=events,
            step_index=self._step_index,
        )

    @property
    def current_state(self) -> Optional[OvercookedState]:
        return self._state

    @property
    def current_game_state(self) -> Optional[GameState]:
        if self._state is None:
            return None
        return self._serialize_state()

    def _serialize_state(self) -> GameState:
        state = self._state
        players = []
        for p in state.players:
            held: Optional[str] = None
            held_object_ingredients: List[str] = []
            if p.held_object is not None:
                if hasattr(p.held_object, "ingredients"):
                    held = "soup"
                    held_object_ingredients = list(p.held_object.ingredients)
                else:
                    held = p.held_object.name
            players.append(
                PlayerSnapshot(
                    position=list(p.position),
                    orientation=list(p.orientation),
                    held_object=held,
                    held_object_ingredients=held_object_ingredients,
                )
            )

        objects = []
        for pos, obj in state.objects.items():
            if hasattr(obj, "ingredients"):
                soup_st = "ready" if obj.is_ready else ("cooking" if obj.is_cooking else "idle")
                objects.append(
                    ObjectSnapshot(
                        name="soup",
                        position=list(pos),
                        soup_state=soup_st,
                        ingredients=list(obj.ingredients),
                    )
                )
            else:
                objects.append(ObjectSnapshot(name=obj.name, position=list(pos)))

        return GameState(
            players=players,
            objects=objects,
            score=self._score,
            step_index=self._step_index,
            time_remaining=self.max_steps - self._step_index,
            layout_name=self.layout_name,
        )


# ---------------------------------------------------------------------------
# Pure helper functions (no self / no I/O — safe to call from step())
# ---------------------------------------------------------------------------


def _snapshot_soups(state: OvercookedState) -> Dict[tuple, Tuple[bool, bool]]:
    """pos → (is_cooking, is_ready) for soup_cooked transition detection."""
    return {
        pos: (obj.is_cooking, obj.is_ready)
        for pos, obj in state.objects.items()
        if hasattr(obj, "ingredients")
    }


def _normalize_action(action: Any) -> Any:
    if isinstance(action, str):
        if action in Action.ALL_ACTIONS:
            return action
        if action not in ACTION_MAP:
            raise ValueError(f"Unknown action name: {action!r}")
        return ACTION_MAP[action]
    return action


def _is_legacy_joint_action(action: Any) -> bool:
    if not isinstance(action, tuple) or len(action) != 2:
        return False
    return action[1] is None or action[1] in Action.ALL_ACTIONS


def _detect_events(
    infos: Dict,
    prev_soups: Dict,
    new_state: OvercookedState,
    step_index: int,
    tomato_reward: float,
    onion_reward: float = 20.0,
) -> List[GameEvent]:
    """
    Detect game events from a single step's infos. Fully synchronous (C1).
    Events are structured so EEGMarkerBridge can consume them later (C5).

    Role mapping: agent_id=0 → human player, agent_id=1 → AI.
    Recipe type mapped from reward: 5→ttt, 10→tto, 15→too, 20→ooo.
    """
    events: List[GameEvent] = []
    event_infos: Dict = infos.get("event_infos", {})
    rewards: List[float] = infos["sparse_reward_by_agent"]

    # Pickup events (from event_infos)
    for key in _PICKUP_EVENT_KEYS:
        for agent_id, triggered in enumerate(event_infos.get(key, [False, False])):
            if triggered:
                role = "player" if agent_id == 0 else "ai"
                item = key.replace("_pickup", "")
                events.append(
                    GameEvent(
                        event_type=f"{role}_pickup",
                        step_index=step_index,
                        agent_id=agent_id,
                        payload={"item": item},
                    )
                )

    # Delivery events (from sparse_reward_by_agent; recipe type inferred from reward)
    for agent_id, reward in enumerate(rewards):
        if reward > 0:
            role = "player" if agent_id == 0 else "ai"
            recipe = _REWARD_TO_RECIPE.get(int(round(reward)), "ooo")
            events.append(
                GameEvent(
                    event_type=f"{role}_deliver",
                    step_index=step_index,
                    agent_id=agent_id,
                    payload={"recipe": recipe, "reward": reward},
                )
            )

    # Soup drop events
    for agent_id, triggered in enumerate(event_infos.get("soup_drop", [False, False])):
        if triggered:
            events.append(
                GameEvent(
                    event_type="soup_drop",
                    step_index=step_index,
                    agent_id=agent_id,
                    payload={},
                )
            )

    # Soup cooked (cooking → ready transition)
    new_soups = _snapshot_soups(new_state)
    for pos, (was_cooking, was_ready) in prev_soups.items():
        if pos in new_soups:
            is_cooking_now, is_ready_now = new_soups[pos]
            if was_cooking and is_ready_now and not was_ready:
                events.append(
                    GameEvent(
                        event_type="soup_cooked",
                        step_index=step_index,
                        agent_id=0,  # cooking has no specific agent
                        payload={"position": list(pos)},
                    )
                )

    return events
