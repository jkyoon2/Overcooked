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

from backend.data.schema import GameState, ObjectSnapshot, PlayerSnapshot
from backend.game.events import GameEvent

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
        ai_checkpoint_id: Optional[str] = None,  # reserved for Task 3 (HSP loader)
    ) -> None:
        self.layout_name = layout_name
        self.tomato_reward = tomato_reward
        self.onion_reward = onion_reward
        self.seed = seed
        # ai_checkpoint_id: plugged in by Task 3. Ignored here; RandomPolicy is used instead.
        self.ai_checkpoint_id = ai_checkpoint_id

        # Configure Recipe singleton BEFORE from_layout_name() per PI-5 / B-cat-A.
        # Called on every __init__ so future multi-trial resets stay safe.
        self._apply_recipe_config()

        self.mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout_name)
        self._state: Optional[OvercookedState] = None
        self._step_index: int = 0
        self._score: float = 0.0
        self.max_steps: int = 600  # 60 s at 10 Hz

        if seed is not None:
            random.seed(seed)

    def _apply_recipe_config(self) -> None:
        # B-cat-A: Recipe.configure() sets class-level state. Re-apply on every
        # reset/init so cross-trial leakage cannot happen when Task 4 adds trials.
        Recipe.configure(
            {
                "onion_value": self.onion_reward,
                "tomato_value": self.tomato_reward,
                "onion_time": 20,
                "tomato_time": 20,
            }
        )

    def reset(self) -> GameState:
        """Start a fresh episode. Returns the initial serialized state."""
        self._apply_recipe_config()  # guard against cross-trial Recipe leakage
        self._state = self.mdp.get_standard_start_state()
        self._step_index = 0
        self._score = 0.0
        return self._serialize_state()

    def step(self, joint_action: Tuple[Any, Any]) -> StepResult:
        """
        Advance one game tick. Synchronous only — no async, no I/O (C1).

        joint_action: (player_action, ai_action) using Action values from actions.py (PI-6).
        """
        assert self._state is not None, "Call reset() before step()"

        prev_soup_states = _snapshot_soups(self._state)

        new_state, infos = self.mdp.get_state_transition(self._state, joint_action)

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
            if p.held_object is not None:
                held = "soup" if hasattr(p.held_object, "ingredients") else p.held_object.name
            players.append(
                PlayerSnapshot(
                    position=list(p.position),
                    orientation=list(p.orientation),
                    held_object=held,
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


def _detect_events(
    infos: Dict,
    prev_soups: Dict,
    new_state: OvercookedState,
    step_index: int,
    tomato_reward: float,
) -> List[GameEvent]:
    """
    Detect game events from a single step's infos. Fully synchronous (C1).
    Events are structured so EEGMarkerBridge can consume them later (C5).
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

    # Delivery events (from sparse_reward_by_agent; recipe inferred from reward value)
    for agent_id, reward in enumerate(rewards):
        if reward > 0:
            role = "player" if agent_id == 0 else "ai"
            recipe = "tomato" if abs(reward - tomato_reward) < 0.5 else "onion"
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
