from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction

from backend.data.schema import HelloAckMessage, HelloAckPayload, HelloMessage
from backend.game.ai_loader import RandomPolicy
from backend.game.engine import ACTION_MAP, OvercookedEngine

# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------


@dataclass
class ConnectionState:
    engine: Optional[OvercookedEngine] = None
    policy: Optional[RandomPolicy] = None
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
        read player_action → policy.act() → engine.step() → serialize
        ─── only then ──► await websocket.send_json()
    """
    while True:
        tick_start = time.monotonic()

        # Read latest player action (overwrite semantics — no queue backlog)
        player_action = conn.latest_player_action if conn.latest_player_action is not None else Action.STAY
        conn.latest_player_action = None  # reset slot

        # AI action — synchronous (C1: no await between here and send)
        ai_action = conn.policy.act(conn.engine.current_state)

        # Step the engine — synchronous (C1)
        result = conn.engine.step((player_action, ai_action))

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
                layout_name = payload.get("layout", "corner_onion_tomato")

                if conn.tick_task and not conn.tick_task.done():
                    conn.tick_task.cancel()
                    try:
                        await conn.tick_task
                    except asyncio.CancelledError:
                        pass

                conn.engine = OvercookedEngine(layout_name=layout_name)
                conn.policy = RandomPolicy()
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
                if conn.tick_task and not conn.tick_task.done():
                    conn.tick_task.cancel()
                    try:
                        await conn.tick_task
                    except asyncio.CancelledError:
                        pass
                conn.tick_task = None

    except WebSocketDisconnect:
        if conn.tick_task and not conn.tick_task.done():
            conn.tick_task.cancel()
    except asyncio.CancelledError:
        pass
