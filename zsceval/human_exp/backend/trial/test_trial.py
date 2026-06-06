from __future__ import annotations

from pathlib import Path
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


def test_session_manager_phase_two_sequence() -> None:
    session = SessionManager("TEST_S01")

    assert session.total_trials == 4
    assert not session.is_complete()
    assert session.trials == [
        TrialCondition(
            ai_checkpoint="tomato",
            player_intent="tomato",
            model_checkpoint="tto_sp_seed4",
        ),
        TrialCondition(
            ai_checkpoint="onion",
            player_intent="tomato",
            model_checkpoint="tto_sp_seed5",
        ),
        TrialCondition(
            ai_checkpoint="tomato",
            player_intent="onion",
            model_checkpoint="ttt_adaptive_seed1",
        ),
        TrialCondition(
            ai_checkpoint="onion",
            player_intent="onion",
            model_checkpoint="ttt_adaptive_seed2",
        ),
    ]

    for expected_trial_id in range(1, session.total_trials + 1):
        assert session.current_trial_id() == expected_trial_id
        has_next = session.advance_trial()
        assert has_next is (expected_trial_id < session.total_trials)

    assert session.is_complete()


def test_ws_session_flow(monkeypatch: Any) -> None:
    saved: Dict[str, Any] = {}

    def fake_save_phase_three_record(
        session_id: str,
        replay_trials: Any,
        selections: Any,
    ) -> Path:
        saved["session_id"] = session_id
        saved["replay_trials"] = replay_trials
        saved["selections"] = selections
        return Path("/tmp/phase3-test.json")

    monkeypatch.setattr(
        "backend.api.websocket.save_phase_three_record",
        fake_save_phase_three_record,
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_session", "payload": {"session_id": "TEST_S01"}})
        for trial_id in range(1, 5):
            trial_start = _receive_until(ws, "trial_start")
            assert trial_start["payload"]["trial_id"] == trial_id
            assert trial_start["payload"]["total_trials"] == 4
            assert trial_start["payload"]["phase"] == "instruction"
            assert trial_start["payload"]["player_hat"] == "blue"
            assert trial_start["payload"]["ai_hat"] == "red"

            ws.send_json({"type": "phase_ready", "payload": {}})
            play_change = _receive_until(ws, "phase_change", phase="play")
            assert play_change["payload"]["duration_ms"] == 75_000

            ws.send_json({"type": "phase_ready", "payload": {}})
            rating_change = _receive_until(ws, "phase_change", phase="rating")
            assert rating_change["payload"]["duration_ms"] == 20_000

            intent_alignment = (
                "yes_clearly" if trial_id in {1, 4} else "no_clearly"
            )
            ws.send_json(
                {
                    "type": "submit_rating",
                    "payload": {
                        "quality": 4,
                        "intent_alignment": intent_alignment,
                    },
                }
            )
            rating_ack = _receive_until(ws, "rating_ack")
            assert rating_ack["payload"]["trial_id"] == trial_id
            assert rating_ack["payload"]["excluded"] is False

            break_change = _receive_until(ws, "phase_change", phase="break")
            assert break_change["payload"]["duration_ms"] == 5_000
            ws.send_json({"type": "phase_ready", "payload": {}})

        phase_three = _receive_until(ws, "phase3_start")
        assert phase_three["payload"]["session_id"] == "TEST_S01"
        assert phase_three["payload"]["frame_duration_ms"] == 100
        assert [trial["trial_id"] for trial in phase_three["payload"]["trials"]] == [
            1,
            2,
            3,
            4,
        ]
        assert all(
            len(trial["frames"]) >= 1
            for trial in phase_three["payload"]["trials"]
        )

        ws.send_json(
            {
                "type": "submit_phase3",
                "payload": {
                    "trials": [
                        {
                            "trial_id": trial_id,
                            "segments": [
                                {
                                    "segment_id": f"{trial_id}-segment-1",
                                    "start_frame": 0,
                                    "end_frame": 0,
                                    "created_at_ms": 1_700_000_000_000,
                                }
                            ],
                        }
                        for trial_id in range(1, 5)
                    ]
                },
            }
        )
        complete = _receive_until(ws, "phase3_complete")
        assert complete["payload"] == {
            "session_id": "TEST_S01",
            "saved": True,
        }

    assert saved["session_id"] == "TEST_S01"
    assert len(saved["replay_trials"]) == 4
    assert len(saved["selections"]) == 4


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
