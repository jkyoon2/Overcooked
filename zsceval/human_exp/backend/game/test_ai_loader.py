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


@pytest.fixture(scope="module")
def primary_policy() -> Iterator[HSPPolicy]:
    policy = load_policy("tto_sp_seed4")
    assert isinstance(policy, HSPPolicy)
    yield policy


@pytest.mark.parametrize(
    ("checkpoint_id", "layout_name", "step"),
    [
        ("tto_sp_seed4", "tto", 10_000_000),
        ("tto_sp_seed5", "tto", 10_000_000),
        ("ttt_adaptive_seed1", "ttt", 50_000_000),
        ("ttt_adaptive_seed2", "ttt", 50_000_000),
    ],
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
    expected_actor_paths = {
        "tto_sp_seed4": (
            "results/Overcooked/tto/shared/rmappo/agent_pool_sp/seed4/"
            "models/actor_periodic_10000000.pt"
        ),
        "tto_sp_seed5": (
            "results/Overcooked/tto/shared/rmappo/agent_pool_sp/seed5/"
            "models/actor_periodic_10000000.pt"
        ),
        "ttt_adaptive_seed1": (
            "results/Overcooked/ttt/shared/adaptive/hsp-S2-s12/seed1/"
            "models/hsp_adaptive/actor_periodic_50000000.pt"
        ),
        "ttt_adaptive_seed2": (
            "results/Overcooked/ttt/shared/adaptive/hsp-S2-s12/seed2/"
            "models/hsp_adaptive/actor_periodic_50000000.pt"
        ),
    }

    for checkpoint_id, expected_suffix in expected_actor_paths.items():
        paths = _resolve_checkpoint_paths(checkpoint_id)
        assert str(paths.actor_path).endswith(expected_suffix)


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
