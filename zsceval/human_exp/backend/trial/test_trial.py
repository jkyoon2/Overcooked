from __future__ import annotations

from typing import Any, Dict, Optional

from starlette.testclient import TestClient, WebSocketTestSession

from backend.main import app
from backend.trial.condition import TrialCondition
from backend.trial.manager import TrialManager, TrialPhase
from backend.trial.session import SessionManager


def test_trial_condition_alignment() -> None:
    assert TrialCondition(ai_checkpoint="tomato", player_intent="tomato").is_aligned
    assert not TrialCondition(ai_checkpoint="onion", player_intent="tomato").is_aligned


def test_trial_manager_full_cycle() -> None:
    manager = TrialManager()
    condition = TrialCondition(ai_checkpoint="tomato", player_intent="tomato")

    manager.start_instruction(trial_id=1, condition=condition)
    assert manager.phase == TrialPhase.INSTRUCTION
    manager.start_play()
    assert manager.phase == TrialPhase.PLAY
    manager.end_play()
    assert manager.phase == TrialPhase.RATING
    result = manager.submit_rating(quality=4, intent_alignment="yes_clearly")

    assert manager.phase == TrialPhase.BREAK
    assert result.excluded is False


def test_manipulation_check_excludes_misaligned_rating() -> None:
    manager = TrialManager()
    condition = TrialCondition(ai_checkpoint="tomato", player_intent="tomato")
    manager.start_instruction(1, condition)
    manager.start_play()
    manager.end_play()

    result = manager.submit_rating(quality=3, intent_alignment="no_clearly")

    assert result.excluded is True
    assert result.exclusion_reason is not None


def test_manipulation_check_scale_definition() -> None:
    aligned = TrialManager()
    aligned.start_instruction(
        1,
        TrialCondition(ai_checkpoint="tomato", player_intent="tomato"),
    )
    aligned.start_play()
    aligned.end_play()
    aligned_result = aligned.submit_rating(
        quality=4,
        intent_alignment="yes_somewhat",
    )
    assert aligned_result.excluded is False

    misaligned = TrialManager()
    misaligned.start_instruction(
        2,
        TrialCondition(ai_checkpoint="onion", player_intent="tomato"),
    )
    misaligned.start_play()
    misaligned.end_play()
    misaligned_result = misaligned.submit_rating(
        quality=4,
        intent_alignment="yes_clearly",
    )
    assert misaligned_result.excluded is True


def test_session_manager_two_trial_sequence() -> None:
    session = SessionManager("TEST_S01")

    assert not session.is_complete()
    assert session.current_condition() == TrialCondition(
        ai_checkpoint="tomato",
        player_intent="tomato",
    )
    assert session.advance_trial()
    assert not session.is_complete()
    assert session.current_condition() == TrialCondition(
        ai_checkpoint="onion",
        player_intent="tomato",
    )
    assert not session.advance_trial()
    assert session.is_complete()


def test_ws_session_flow() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session", "payload": {"session_id": "TEST_S01"}})
        trial_start = ws.receive_json()
        assert trial_start["type"] == "trial_start"
        assert trial_start["payload"]["trial_id"] == 1
        assert trial_start["payload"]["phase"] == "instruction"
        assert trial_start["payload"]["player_hat"] == "blue"
        assert trial_start["payload"]["ai_hat"] == "red"

        ws.send_json({"type": "phase_ready", "payload": {}})
        play_change = _receive_until(ws, "phase_change", phase="play")
        assert play_change["payload"]["duration_ms"] == 75_000

        ws.send_json({"type": "phase_ready", "payload": {}})
        rating_change = _receive_until(ws, "phase_change", phase="rating")
        assert rating_change["payload"]["duration_ms"] == 20_000

        ws.send_json(
            {
                "type": "submit_rating",
                "payload": {"quality": 4, "intent_alignment": "yes_clearly"},
            }
        )
        rating_ack = _receive_until(ws, "rating_ack")
        assert rating_ack["payload"]["trial_id"] == 1
        assert rating_ack["payload"]["excluded"] is False

        break_change = _receive_until(ws, "phase_change", phase="break")
        assert break_change["payload"]["duration_ms"] == 5_000

        ws.send_json({"type": "phase_ready", "payload": {}})
        next_trial = _receive_until(ws, "trial_start")
        assert next_trial["payload"]["trial_id"] == 2


def _receive_until(
    ws: WebSocketTestSession,
    message_type: str,
    phase: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    for _ in range(limit):
        message = ws.receive_json()
        if message.get("type") != message_type:
            continue
        if phase is not None and message.get("payload", {}).get("phase") != phase:
            continue
        return message
    raise AssertionError(f"Did not receive {message_type!r} message")
