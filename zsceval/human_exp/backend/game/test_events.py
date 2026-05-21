from __future__ import annotations

"""Tests for event detection helpers (Task 2).

Events are detected inside engine.step() using _detect_events().
Tests here verify the detection logic in isolation.
"""

from unittest.mock import MagicMock

import pytest

from backend.game.engine import OvercookedEngine, _detect_events, ACTION_MAP
from backend.game.events import GameEvent


LAYOUT = "corner_onion_tomato"


def _make_fake_infos(
    pickup_agent: int | None = None,
    pickup_item: str = "onion",
    delivery_agent: int | None = None,
    delivery_reward: float = 0.0,
    soup_drop_agent: int | None = None,
) -> dict:
    """Build a minimal infos dict for _detect_events testing."""
    event_infos: dict = {key: [False, False] for key in (
        "onion_pickup", "tomato_pickup", "dish_pickup", "soup_pickup", "soup_drop"
    )}
    sparse_reward = [0.0, 0.0]

    if pickup_agent is not None:
        event_infos[f"{pickup_item}_pickup"][pickup_agent] = True

    if delivery_agent is not None:
        sparse_reward[delivery_agent] = delivery_reward

    if soup_drop_agent is not None:
        event_infos["soup_drop"][soup_drop_agent] = True

    return {"event_infos": event_infos, "sparse_reward_by_agent": sparse_reward}


def test_pickup_event_emitted_on_holding_change() -> None:
    """Picking up an onion fires a player_pickup event with item=onion."""
    infos = _make_fake_infos(pickup_agent=0, pickup_item="onion")
    new_state = MagicMock()
    new_state.objects = {}

    events = _detect_events(infos, {}, new_state, step_index=1, tomato_reward=5)

    pickup = [e for e in events if e.event_type == "player_pickup"]
    assert len(pickup) == 1
    assert pickup[0].agent_id == 0
    assert pickup[0].payload["item"] == "onion"


def test_ai_pickup_event_emitted() -> None:
    """Agent 1 picking up a tomato fires an ai_pickup event."""
    infos = _make_fake_infos(pickup_agent=1, pickup_item="tomato")
    new_state = MagicMock()
    new_state.objects = {}

    events = _detect_events(infos, {}, new_state, step_index=5, tomato_reward=5)

    pickup = [e for e in events if e.event_type == "ai_pickup"]
    assert len(pickup) == 1
    assert pickup[0].agent_id == 1


def test_player_deliver_event_emitted() -> None:
    """Reward for agent 0 → player_deliver event with correct recipe."""
    infos = _make_fake_infos(delivery_agent=0, delivery_reward=5.0)
    new_state = MagicMock()
    new_state.objects = {}

    events = _detect_events(infos, {}, new_state, step_index=10, tomato_reward=5)

    deliver = [e for e in events if e.event_type == "player_deliver"]
    assert len(deliver) == 1
    assert deliver[0].payload["recipe"] == "tomato"
    assert deliver[0].payload["reward"] == 5.0


def test_ai_deliver_event_emitted() -> None:
    """Reward for agent 1 → ai_deliver event."""
    infos = _make_fake_infos(delivery_agent=1, delivery_reward=20.0)
    new_state = MagicMock()
    new_state.objects = {}

    events = _detect_events(infos, {}, new_state, step_index=20, tomato_reward=5)

    deliver = [e for e in events if e.event_type == "ai_deliver"]
    assert len(deliver) == 1
    assert deliver[0].payload["recipe"] == "onion"
    assert deliver[0].payload["reward"] == 20.0


def test_soup_drop_event_emitted() -> None:
    """soup_drop event fires when event_infos["soup_drop"] is True."""
    infos = _make_fake_infos(soup_drop_agent=0)
    new_state = MagicMock()
    new_state.objects = {}

    events = _detect_events(infos, {}, new_state, step_index=3, tomato_reward=5)

    drop = [e for e in events if e.event_type == "soup_drop"]
    assert len(drop) == 1
    assert drop[0].agent_id == 0


def test_soup_cooked_event_on_cooking_to_ready_transition() -> None:
    """soup_cooked fires when a soup transitions from cooking to ready."""
    pos = (2, 1)
    prev_soups = {pos: (True, False)}  # was_cooking=True, was_ready=False

    new_state = MagicMock()
    soup_obj = MagicMock()
    soup_obj.is_cooking = False
    soup_obj.is_ready = True
    new_state.objects = {pos: soup_obj}
    # Make hasattr(soup_obj, "ingredients") return True
    type(soup_obj).ingredients = MagicMock()

    infos = {"event_infos": {}, "sparse_reward_by_agent": [0.0, 0.0]}

    events = _detect_events(infos, prev_soups, new_state, step_index=25, tomato_reward=5)

    cooked = [e for e in events if e.event_type == "soup_cooked"]
    assert len(cooked) == 1
    assert cooked[0].payload["position"] == list(pos)


def test_no_events_on_idle_step() -> None:
    """A STAY step with no deliveries/pickups produces no events."""
    infos = _make_fake_infos()
    new_state = MagicMock()
    new_state.objects = {}

    events = _detect_events(infos, {}, new_state, step_index=1, tomato_reward=5)
    assert events == []


def test_game_events_have_correct_structure() -> None:
    """GameEvent fields match the spec (C5 EEG bridge compatibility)."""
    e = GameEvent(
        event_type="player_pickup",
        step_index=42,
        agent_id=0,
        payload={"item": "onion"},
    )
    assert e.event_type == "player_pickup"
    assert e.step_index == 42
    assert e.agent_id == 0
    d = e.model_dump()
    assert "event_type" in d
    assert "step_index" in d
    assert "agent_id" in d
    assert "payload" in d


def test_engine_emits_pickup_during_real_step() -> None:
    """Integration: a real game step with a real pickup action emits a pickup event when held object changes."""
    engine = OvercookedEngine(layout_name=LAYOUT, seed=0)
    engine.reset()

    from unittest.mock import patch

    original = engine.mdp.get_state_transition

    def fake_transition(state, joint_action):
        new_state, infos = original(state, joint_action)
        infos["event_infos"]["onion_pickup"] = [True, False]
        return new_state, infos

    with patch.object(engine.mdp, "get_state_transition", side_effect=fake_transition):
        result = engine.step((ACTION_MAP["INTERACT"], ACTION_MAP["STAY"]))

    pickup_events = [e for e in result.events if e.event_type == "player_pickup"]
    assert len(pickup_events) == 1
    assert pickup_events[0].agent_id == 0
