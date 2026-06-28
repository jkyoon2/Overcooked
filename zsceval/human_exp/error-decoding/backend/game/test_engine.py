"""Tests for OvercookedEngine (Task 2).

PI-6: actions are tuple/string values (not integers).
PI-5: Recipe.configure() sets onion_value=20, tomato_value=5.
PI-2: get_state_transition() returns (new_state, infos) with sparse_reward_by_agent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.data.schema import GameState
from backend.game.engine import ACTION_MAP, OvercookedEngine, StepResult
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    PlayerState,
    Recipe,
    SoupState,
)


LAYOUT = "corner_onion_tomato"


@pytest.fixture
def engine() -> OvercookedEngine:
    return OvercookedEngine(layout_name=LAYOUT, seed=42)


def test_reset_returns_valid_state(engine: OvercookedEngine) -> None:
    state = engine.reset()
    assert isinstance(state, GameState)
    assert len(state.players) == 2
    assert state.step_index == 0
    assert state.score == 0.0
    assert state.layout_name == LAYOUT
    assert state.time_remaining == engine.max_steps


def test_step_increments_step_index(engine: OvercookedEngine) -> None:
    engine.reset()
    result = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))
    assert isinstance(result, StepResult)
    assert result.step_index == 1
    assert result.next_state.step_index == 1
    assert result.next_state.time_remaining == engine.max_steps - 1


def test_step_returns_two_rewards(engine: OvercookedEngine) -> None:
    engine.reset()
    result = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))
    assert len(result.rewards) == 2
    assert result.rewards[0] >= 0
    assert result.rewards[1] >= 0


def test_done_after_max_steps(engine: OvercookedEngine) -> None:
    engine.reset()
    engine.max_steps = 3
    for _ in range(2):
        r = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))
        assert not r.done
    r = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))
    assert r.done


def test_score_accumulates(engine: OvercookedEngine) -> None:
    engine.reset()
    # Patch get_state_transition to simulate a delivery for agent 0
    original_transition = engine.mdp.get_state_transition
    call_count = [0]

    def fake_transition(state, joint_action):
        new_state, infos = original_transition(state, joint_action)
        call_count[0] += 1
        if call_count[0] == 1:
            infos["sparse_reward_by_agent"] = [5.0, 0.0]
        return new_state, infos

    with patch.object(engine.mdp, "get_state_transition", side_effect=fake_transition):
        result = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))

    assert result.next_state.score == 5.0
    assert result.rewards[0] == 5.0


def test_tomato_delivery_yields_5(engine: OvercookedEngine) -> None:
    """Human (agent 0) delivers ttt soup → player_deliver, reward 5, recipe ttt. (PI-5)"""
    engine.reset()

    def fake_transition(state, joint_action):
        new_state, infos = engine.mdp.__class__.get_state_transition(
            engine.mdp, state, joint_action
        )
        infos["sparse_reward_by_agent"] = [5.0, 0.0]  # agent 0 = human
        infos["event_infos"]["soup_delivery"] = [True, False]
        return new_state, infos

    with patch.object(engine.mdp, "get_state_transition", side_effect=fake_transition):
        result = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))

    assert result.rewards[0] == 5.0
    deliver_events = [e for e in result.events if e.event_type == "player_deliver"]
    assert len(deliver_events) == 1
    assert deliver_events[0].payload["recipe"] == "ttt"
    assert deliver_events[0].payload["reward"] == 5.0


def test_onion_delivery_yields_20(engine: OvercookedEngine) -> None:
    """Human (agent 0) delivers ooo soup → player_deliver, reward 20, recipe ooo. (PI-5)"""
    engine.reset()

    def fake_transition(state, joint_action):
        new_state, infos = engine.mdp.__class__.get_state_transition(
            engine.mdp, state, joint_action
        )
        infos["sparse_reward_by_agent"] = [20.0, 0.0]  # agent 0 = human
        infos["event_infos"]["soup_delivery"] = [True, False]
        return new_state, infos

    with patch.object(engine.mdp, "get_state_transition", side_effect=fake_transition):
        result = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))

    assert result.rewards[0] == 20.0
    deliver_events = [e for e in result.events if e.event_type == "player_deliver"]
    assert len(deliver_events) == 1
    assert deliver_events[0].payload["recipe"] == "ooo"
    assert deliver_events[0].payload["reward"] == 20.0


@pytest.mark.parametrize(
    ("ingredients", "expected_reward"),
    [
        (["tomato", "tomato", "tomato"], 5),
        (["tomato", "tomato", "onion"], 10),
        (["tomato", "onion", "onion"], 15),
        (["onion", "onion", "onion"], 20),
    ],
)
def test_onion_checkpoint_layout_accepts_every_experiment_recipe(
    ingredients: list[str],
    expected_reward: int,
) -> None:
    engine = OvercookedEngine(layout_name="ooo")
    engine.reset()
    state = engine.current_state
    assert state is not None

    assert engine.mdp.get_recipe_value(state, Recipe(ingredients)) == expected_reward


def test_tomato_delivery_in_onion_checkpoint_layout_yields_five_points() -> None:
    engine = OvercookedEngine(layout_name="ooo")
    engine.reset()
    state = engine.current_state
    assert state is not None

    serving_position = (1, 3)
    tomato_soup = SoupState(
        serving_position,
        ingredients=[
            ObjectState("tomato", serving_position)
            for _ in range(3)
        ],
        cooking_tick=20,
        cook_time=20,
    )
    human = PlayerState(serving_position, Direction.WEST, tomato_soup)
    state.players = (human, state.players[1])

    result = engine.step((ACTION_MAP["INTERACT"], ACTION_MAP["STAY"]))

    assert result.rewards == (5.0, 0.0)
    assert result.next_state.score == 5.0
    assert [
        event.payload["recipe"]
        for event in result.events
        if event.event_type == "player_deliver"
    ] == ["ttt"]


def test_serialized_player_state_preserves_held_soup_ingredients() -> None:
    engine = OvercookedEngine(layout_name="ooo")
    engine.reset()
    state = engine.current_state
    assert state is not None

    player_position = state.players[0].position
    tomato_soup = SoupState(
        player_position,
        ingredients=[
            ObjectState("tomato", player_position)
            for _ in range(3)
        ],
        cooking_tick=20,
        cook_time=20,
    )
    state.players[0].set_object(tomato_soup)

    snapshot = engine.current_game_state

    assert snapshot is not None
    assert snapshot.players[0].held_object == "soup"
    assert snapshot.players[0].held_object_ingredients == [
        "tomato",
        "tomato",
        "tomato",
    ]


def test_step_is_synchronous(engine: OvercookedEngine) -> None:
    """step() must return a plain value, not a coroutine (C1)."""
    import inspect
    engine.reset()
    result = engine.step((ACTION_MAP["STAY"], ACTION_MAP["STAY"]))
    assert not inspect.iscoroutine(result), "step() must be synchronous (C1)"


def test_action_map_covers_all_frontend_actions() -> None:
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction

    for name in ("NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"):
        assert name in ACTION_MAP
    assert ACTION_MAP["INTERACT"] == Action.INTERACT
    assert ACTION_MAP["STAY"] == Action.STAY
    assert ACTION_MAP["NORTH"] == Direction.NORTH
    assert ACTION_MAP["SOUTH"] == Direction.SOUTH
