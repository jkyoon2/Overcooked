from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action

from backend.data.schema import (
    HelloAckMessage,
    HelloAckPayload,
    HelloMessage,
    PhaseChangeMessage,
    PhaseChangePayload,
    PhaseReadyMessage,
    RatingAckMessage,
    RatingAckPayload,
    SessionCompleteMessage,
    SessionCompletePayload,
    StartSessionMessage,
    SubmitRatingMessage,
    TrialStartMessage,
    TrialStartPayload,
)
from backend.game.engine import ACTION_MAP, OvercookedEngine
from backend.trial.condition import TrialCondition
from backend.trial.manager import TrialPhase
from backend.trial.session import SessionManager


INSTRUCTION_DURATION_MS = 10_000
PLAY_DURATION_MS = 75_000
RATING_DURATION_MS = 20_000
BREAK_DURATION_MS = 5_000
PLAYER_HAT = "blue"
AI_HAT = "red"
_LAYOUT_BY_CHECKPOINT = {
    "tomato": "ttt",
    "onion": "ooo",
    "ttt": "ttt",
    "tto": "tto",
    "too": "too",
    "ooo": "ooo",
}

# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------


@dataclass
class ConnectionState:
    engine: Optional[OvercookedEngine] = None
    session: Optional[SessionManager] = None
    latest_player_action: Any = None  # overwrite slot; None → STAY
    tick_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
    tick_timestamps: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tick driver (10 Hz)
# ---------------------------------------------------------------------------


async def _tick_loop(websocket: WebSocket, conn: ConnectionState) -> None:
    """
    Drives the game at 10 Hz and broadcasts state to the frontend.

    Synchronous path (C1):
        read player_action → engine.step() selects AI action → serialize
        ─── only then ──► await websocket.send_json()
    """
    while True:
        tick_start = time.monotonic()

        if conn.session and conn.session.trial_manager.phase != TrialPhase.PLAY:
            # Sleep required: cooperative pause while waiting for phase_ready.
            await asyncio.sleep(0.1)
            continue
        if conn.engine is None:
            # Sleep required: cooperative pause until a game engine exists.
            await asyncio.sleep(0.1)
            continue

        # Read latest player action (overwrite semantics — no queue backlog)
        player_action = conn.latest_player_action if conn.latest_player_action is not None else Action.STAY
        conn.latest_player_action = None  # reset slot

        # Step the engine — synchronous; AI action is selected inside engine.step().
        result = conn.engine.step(player_action)

        # Serialize state
        msg = {
            "type": "game_step",
            "payload": {
                "step_index": result.step_index,
                "state": result.next_state.model_dump(),
                "events": [e.model_dump() for e in result.events],
                "rewards": list(result.rewards),
                "server_timestamp_ms": int(time.time() * 1000),
            },
        }

        # Track tick rate for AC5 measurement
        conn.tick_timestamps.append(tick_start)
        if len(conn.tick_timestamps) > 150:
            conn.tick_timestamps.pop(0)
        if len(conn.tick_timestamps) >= 20 and result.step_index % 50 == 0:
            span = conn.tick_timestamps[-1] - conn.tick_timestamps[0]
            n = len(conn.tick_timestamps) - 1
            rate = n / span if span > 0 else 0.0
            print(f"[tick] step={result.step_index:4d}  rate={rate:.2f} Hz  "
                  f"score={result.next_state.score:.0f}")

        # Notify pygame window of new engine state (throwaway tool — no-op if inactive)
        try:
            from backend.game.pygame_window import set_pygame_engine
            set_pygame_engine(conn.engine)
        except Exception:
            pass

        # Send game_step to frontend — await required: WebSocket.send_json() is async
        await websocket.send_json(msg)

        if result.done:
            end_msg = {
                "type": "game_end",
                "payload": {
                    "step_index": result.step_index,
                    "final_score": result.next_state.score,
                },
            }
            # Send game_end — await required: WebSocket.send_json() is async
            await websocket.send_json(end_msg)
            break

        # Sleep remainder of 100 ms tick period — await required: asyncio cooperative scheduling
        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0.0, 0.1 - elapsed))


# ---------------------------------------------------------------------------
# Main WebSocket handler
# ---------------------------------------------------------------------------


async def websocket_handler(websocket: WebSocket) -> None:
    # Accept connection — await required: WebSocket.accept() is async
    await websocket.accept()
    conn = ConnectionState()

    try:
        while True:
            # Receive next client message — await required: WebSocket.receive_json() is async
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            # --- Task 1 hello/hello_ack (kept for backward compat) ---
            if msg_type == "hello":
                parsed = HelloMessage.model_validate(data)
                timestamp_ms = int(time.time() * 1000)
                ack = HelloAckMessage(
                    type="hello_ack",
                    payload=HelloAckPayload(
                        echo=parsed.payload.message,
                        server_timestamp_ms=timestamp_ms,
                    ),
                )
                # Send ack — await required: WebSocket.send_json() is async
                await websocket.send_json(ack.model_dump())

            # --- Task 2 game protocol ---
            elif msg_type == "start_game":
                payload = data.get("payload", {})
                ai_checkpoint_id = payload.get("ai_checkpoint", "tomato")
                if ai_checkpoint_id not in _LAYOUT_BY_CHECKPOINT:
                    expected = ", ".join(sorted(_LAYOUT_BY_CHECKPOINT))
                    raise ValueError(
                        f"Unknown AI checkpoint id {ai_checkpoint_id!r}; "
                        f"expected one of {expected}"
                    )
                layout_name = (
                    payload.get("layout")
                    if "ai_checkpoint" in payload
                    else None
                ) or _LAYOUT_BY_CHECKPOINT[ai_checkpoint_id]

                await _cancel_tick_task(conn)

                conn.engine = OvercookedEngine(
                    layout_name=layout_name,
                    ai_checkpoint_id=ai_checkpoint_id,
                )
                conn.session = None
                conn.tick_timestamps = []
                initial_state = conn.engine.reset()

                start_msg = {
                    "type": "game_start",
                    "payload": {
                        "layout": layout_name,
                        "initial_state": initial_state.model_dump(),
                    },
                }
                # Send game_start — await required: WebSocket.send_json() is async
                await websocket.send_json(start_msg)

                conn.tick_task = asyncio.create_task(_tick_loop(websocket, conn))

            elif msg_type == "player_action":
                action_name = data.get("payload", {}).get("action", "STAY")
                conn.latest_player_action = ACTION_MAP.get(action_name, Action.STAY)

            elif msg_type == "end_game":
                await _cancel_tick_task(conn)

            # --- Task 4 trial/session protocol ---
            elif msg_type == "start_session":
                parsed = StartSessionMessage.model_validate(data)
                await _cancel_tick_task(conn)
                conn.engine = None
                conn.session = SessionManager(parsed.payload.session_id)
                conn.session.trial_manager.start_instruction(
                    trial_id=conn.session.current_trial_id(),
                    condition=conn.session.current_condition(),
                )
                # Send trial_start — await required: WebSocket.send_json() is async
                await websocket.send_json(_trial_start_message(conn.session).model_dump())

            elif msg_type == "phase_ready":
                PhaseReadyMessage.model_validate(data)
                if conn.session is None:
                    continue
                phase = conn.session.trial_manager.phase
                if phase == TrialPhase.INSTRUCTION:
                    await _start_play_phase(websocket, conn)
                elif phase == TrialPhase.PLAY:
                    await _end_play_phase(websocket, conn)
                elif phase == TrialPhase.BREAK:
                    await _complete_break_or_session(websocket, conn)

            elif msg_type == "submit_rating":
                parsed = SubmitRatingMessage.model_validate(data)
                if conn.session is None:
                    continue
                result = conn.session.trial_manager.submit_rating(
                    quality=parsed.payload.quality,
                    intent_alignment=parsed.payload.intent_alignment,
                )
                rating_ack = RatingAckMessage(
                    type="rating_ack",
                    payload=RatingAckPayload(
                        trial_id=result.trial_id,
                        excluded=result.excluded,
                        exclusion_reason=result.exclusion_reason,
                    ),
                )
                # Send rating_ack — await required: WebSocket.send_json() is async
                await websocket.send_json(rating_ack.model_dump())
                # Send phase_change — await required: WebSocket.send_json() is async
                await websocket.send_json(
                    _phase_change_message(TrialPhase.BREAK).model_dump()
                )

    except WebSocketDisconnect:
        await _cancel_tick_task(conn)
    except asyncio.CancelledError:
        pass


async def _cancel_tick_task(conn: ConnectionState) -> None:
    if conn.tick_task and not conn.tick_task.done():
        conn.tick_task.cancel()
        try:
            # Await required: wait for cooperative task cancellation to finish.
            await conn.tick_task
        except asyncio.CancelledError:
            pass
    conn.tick_task = None


async def _start_play_phase(websocket: WebSocket, conn: ConnectionState) -> None:
    assert conn.session is not None
    condition = conn.session.current_condition()
    conn.engine = OvercookedEngine(
        layout_name=_layout_for_condition(condition),
        ai_checkpoint_id=condition.ai_checkpoint,
    )
    initial_state = conn.engine.reset()
    conn.session.trial_manager.start_play()
    conn.tick_timestamps = []

    # Send phase_change — await required: WebSocket.send_json() is async
    await websocket.send_json(_phase_change_message(TrialPhase.PLAY).model_dump())
    game_start = {
        "type": "game_start",
        "payload": {
            "layout": conn.engine.layout_name,
            "initial_state": initial_state.model_dump(),
        },
    }
    # Send game_start — await required: WebSocket.send_json() is async
    await websocket.send_json(game_start)
    conn.tick_task = asyncio.create_task(_tick_loop(websocket, conn))


async def _end_play_phase(websocket: WebSocket, conn: ConnectionState) -> None:
    assert conn.session is not None
    await _cancel_tick_task(conn)
    conn.session.trial_manager.end_play()
    # Send phase_change — await required: WebSocket.send_json() is async
    await websocket.send_json(_phase_change_message(TrialPhase.RATING).model_dump())


async def _complete_break_or_session(websocket: WebSocket, conn: ConnectionState) -> None:
    assert conn.session is not None
    conn.session.trial_manager.start_break()
    if conn.session.advance_trial():
        conn.session.trial_manager.start_instruction(
            trial_id=conn.session.current_trial_id(),
            condition=conn.session.current_condition(),
        )
        # Send trial_start — await required: WebSocket.send_json() is async
        await websocket.send_json(_trial_start_message(conn.session).model_dump())
        return

    complete = SessionCompleteMessage(
        type="session_complete",
        payload=SessionCompletePayload(session_id=conn.session.session_id),
    )
    # Send session_complete — await required: WebSocket.send_json() is async
    await websocket.send_json(complete.model_dump())


def _trial_start_message(session: SessionManager) -> TrialStartMessage:
    return TrialStartMessage(
        type="trial_start",
        payload=TrialStartPayload(
            trial_id=session.current_trial_id(),
            phase="instruction",
            duration_ms=INSTRUCTION_DURATION_MS,
            player_hat=PLAYER_HAT,
            ai_hat=AI_HAT,
        ),
    )


def _phase_change_message(phase: TrialPhase) -> PhaseChangeMessage:
    return PhaseChangeMessage(
        type="phase_change",
        payload=PhaseChangePayload(
            phase=phase.value,
            duration_ms=_duration_for_phase(phase),
        ),
    )


def _duration_for_phase(phase: TrialPhase) -> int:
    if phase == TrialPhase.INSTRUCTION:
        return INSTRUCTION_DURATION_MS
    if phase == TrialPhase.PLAY:
        return PLAY_DURATION_MS
    if phase == TrialPhase.RATING:
        return RATING_DURATION_MS
    if phase == TrialPhase.BREAK:
        return BREAK_DURATION_MS
    return 0


def _layout_for_condition(condition: TrialCondition) -> str:
    return _LAYOUT_BY_CHECKPOINT[condition.ai_checkpoint]
