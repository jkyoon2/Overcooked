from __future__ import annotations

import time
from typing import Iterator

import pytest

from backend.game.ai_loader import (
    HSPPolicy,
    _resolve_checkpoint_paths,
    load_policy,
)
from backend.game.engine import ACTION_MAP, OvercookedEngine


_MEP_CHECKPOINT_PARAMS = [
    (f"{layout}_mep{variant}", layout, 10_000_000)
    for layout in ("tto", "too")
    for variant in (1, 2, 3, 4)
]


@pytest.fixture(scope="module")
def primary_policy() -> Iterator[HSPPolicy]:
    policy = load_policy("tto_mep1")
    assert isinstance(policy, HSPPolicy)
    yield policy


@pytest.mark.parametrize(
    ("checkpoint_id", "layout_name", "step"),
    _MEP_CHECKPOINT_PARAMS,
)
def test_requested_policy_returns_valid_action(
    checkpoint_id: str,
    layout_name: str,
    step: int,
) -> None:
    policy = load_policy(checkpoint_id)
    engine = OvercookedEngine(layout_name=layout_name, ai_checkpoint_id=None)
    engine.reset()
    state = engine.current_state
    assert state is not None

    action = policy.act(state)

    assert action in ACTION_MAP.values()
    assert policy.paths.layout_name == layout_name
    assert policy.paths.step == step


def test_hsp_policy_is_deterministic(primary_policy: HSPPolicy) -> None:
    engine = OvercookedEngine(layout_name="tto", ai_checkpoint_id=None)
    engine.reset()
    state = engine.current_state
    assert state is not None

    action_1 = primary_policy.act(state)
    action_2 = primary_policy.act(state)

    assert action_1 == action_2, "HSP policy must be deterministic (argmax)"


def test_requested_checkpoint_paths_are_exact() -> None:
    for checkpoint_id, layout, _ in _MEP_CHECKPOINT_PARAMS:
        variant = checkpoint_id.rsplit("mep", 1)[-1]
        expected_suffix = (
            f"results/Overcooked/{layout}/shared/mep/mep-S1-s5/seed1/"
            f"models/mep{variant}/actor_periodic_10000000.pt"
        )
        paths = _resolve_checkpoint_paths(checkpoint_id)
        assert str(paths.actor_path).endswith(expected_suffix)
        assert paths.critic_path is None


def test_engine_requires_policy_for_implicit_ai_action() -> None:
    engine = OvercookedEngine(layout_name="corner_onion_tomato")
    engine.reset()

    with pytest.raises(RuntimeError, match="requires ai_action"):
        engine.step("STAY")


def test_hsp_act_latency(primary_policy: HSPPolicy) -> None:
    engine = OvercookedEngine(layout_name="tto", ai_checkpoint_id=None)
    engine.reset()
    state = engine.current_state
    assert state is not None

    # Exclude one warmup call from the latency sample.
    primary_policy.act(state)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        primary_policy.act(state)
        times.append((time.perf_counter() - start) * 1000)

    assert max(times) < 5.0, f"act() exceeded 5ms budget: max={max(times):.1f}ms"
