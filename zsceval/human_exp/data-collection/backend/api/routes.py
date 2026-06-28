from __future__ import annotations

import threading
import time
from typing import Any, Optional

from flask import jsonify
from flask_socketio import SocketIO, emit

from game.engine import Action, LAYOUTS, TwoPlayerEngine
from game.recorder import Recorder

TICK_HZ = 10
TICK_INTERVAL = 1.0 / TICK_HZ


class GameSession:
    def __init__(self) -> None:
        self.engine: Optional[TwoPlayerEngine] = None
        self.recorder: Optional[Recorder] = None
        self.participant_id: str = ""
        self.trial_index: int = 0       # 0-3
        self.running: bool = False
        self.action_p0: Any = Action.STAY
        self.action_p1: Any = Action.STAY
        self._lock = threading.Lock()


_session = GameSession()


def register(app: Any, socketio: SocketIO) -> None:
    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @socketio.on("set_participant")
    def on_set_participant(data: dict) -> None:
        _session.participant_id = data.get("participant_id", "").strip()
        _session.trial_index = 0
        emit("participant_ack", {"participant_id": _session.participant_id})

    @socketio.on("start_trial")
    def on_start_trial(data: dict) -> None:
        idx = _session.trial_index
        if idx >= len(LAYOUTS):
            emit("all_done", {})
            return

        layout = LAYOUTS[idx]
        recipe = data.get("recipe", "OOO")

        with _session._lock:
            _session.running = False   # stop previous loop if any
            _session.engine = TwoPlayerEngine(layout, recipe)
            initial = _session.engine.reset()
            _session.recorder = Recorder(
                participant_id=_session.participant_id,
                layout=layout,
                recipe=recipe,
                engine=_session.engine,
            )
            _session.action_p0 = Action.STAY
            _session.action_p1 = Action.STAY
            _session.running = True

        emit("trial_start", {
            "trial_index": idx,
            "total_trials": len(LAYOUTS),
            "layout": layout,
            "recipe": recipe,
            "initial_state": initial,
        })

        socketio.start_background_task(_game_loop, socketio)

    @socketio.on("player_action")
    def on_player_action(data: dict) -> None:
        player = data.get("player", 0)
        action = data.get("action", "STAY")
        with _session._lock:
            if player == 0:
                _session.action_p0 = action
            else:
                _session.action_p1 = action


def _game_loop(socketio: SocketIO) -> None:
    while True:
        with _session._lock:
            if not _session.running or _session.engine is None:
                return
            a0 = _session.action_p0
            a1 = _session.action_p1
            _session.action_p0 = Action.STAY
            _session.action_p1 = Action.STAY

        tick_start = time.monotonic()
        result = _session.engine.step(a0, a1)

        if _session.recorder:
            _session.recorder.record(result.joint_action, result.reward)

        socketio.emit("game_step", {
            "state": result.state_dict,
            "step_index": result.step_index,
            "reward": result.reward,
            "done": result.done,
        })

        if result.done:
            with _session._lock:
                _session.running = False
                score = result.state_dict["score"]
                saved_path = _session.recorder.save(int(score)) if _session.recorder else None
                _session.trial_index += 1

            socketio.emit("trial_end", {
                "trial_index": _session.trial_index - 1,
                "score": score,
                "saved": str(saved_path) if saved_path else None,
                "next_trial_index": _session.trial_index,
                "all_done": _session.trial_index >= len(LAYOUTS),
            })
            return

        elapsed = time.monotonic() - tick_start
        socketio.sleep(max(0.0, TICK_INTERVAL - elapsed))
