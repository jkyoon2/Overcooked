from __future__ import annotations

import time
from typing import Iterator

import pytest

from backend.game.ai_loader import HSPPolicy, load_policy
from backend.game.engine import ACTION_MAP, OvercookedEngine


@pytest.fixture(scope="module")
def tomato_policy() -> Iterator[HSPPolicy]:
    policy = load_policy("tomato")
    assert isinstance(policy, HSPPolicy)
    yield policy


@pytest.mark.parametrize(
    ("checkpoint_id", "layout_name"),
    [
        ("tomato", "ttt"),
        ("onion", "ooo"),
    ],
)
def test_hsp_policy_returns_valid_action(checkpoint_id: str, layout_name: str) -> None:
    policy = load_policy(checkpoint_id)
    engine = OvercookedEngine(layout_name=layout_name, ai_checkpoint_id=None)
    engine.reset()
    state = engine.current_state
    assert state is not None

    action = policy.act(state)

    assert action in ACTION_MAP.values()


def test_hsp_policy_is_deterministic(tomato_policy: HSPPolicy) -> None:
    engine = OvercookedEngine(layout_name="ttt", ai_checkpoint_id=None)
    engine.reset()
    state = engine.current_state
    assert state is not None

    action_1 = tomato_policy.act(state)
    action_2 = tomato_policy.act(state)

    assert action_1 == action_2, "HSP policy must be deterministic (argmax)"


def test_engine_accepts_both_checkpoints() -> None:
    engine_t = OvercookedEngine(layout_name="ttt", ai_checkpoint_id="tomato")
    engine_o = OvercookedEngine(layout_name="ooo", ai_checkpoint_id="onion")
    engine_t.reset()
    engine_o.reset()

    result_t = engine_t.step(("STAY", None))
    result_o = engine_o.step(("STAY", None))

    assert result_t.step_index == 1
    assert result_o.step_index == 1


def test_engine_requires_policy_for_implicit_ai_action() -> None:
    engine = OvercookedEngine(layout_name="corner_onion_tomato")
    engine.reset()

    with pytest.raises(RuntimeError, match="requires ai_action"):
        engine.step("STAY")


def test_hsp_act_latency(tomato_policy: HSPPolicy) -> None:
    engine = OvercookedEngine(layout_name="ttt", ai_checkpoint_id=None)
    engine.reset()
    state = engine.current_state
    assert state is not None

    # Exclude one warmup call from the latency sample.
    tomato_policy.act(state)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        tomato_policy.act(state)
        times.append((time.perf_counter() - start) * 1000)

    assert max(times) < 5.0, f"act() exceeded 5ms budget: max={max(times):.1f}ms"
