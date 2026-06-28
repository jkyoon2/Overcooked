# Task 4 Pre-Investigation

Date: 2026-05-31

Scope: inspection only. No Task 4 implementation changes have been made.

## PI-1: Current `backend/trial/` Stubs

Current files:

- `backend/trial/__init__.py`: empty package marker.
- `backend/trial/condition.py`: only module docstring and
  `from __future__ import annotations`.
- `backend/trial/manager.py`: only module docstring and
  `from __future__ import annotations`.
- `backend/trial/session.py`: only module docstring and
  `from __future__ import annotations`.

No classes, methods, enums, Pydantic models, or tests are currently defined in
`backend/trial/`.

## PI-2: WebSocket Integration Point

Incoming client messages are handled in:

```python
backend/api/websocket.py::websocket_handler()
```

Current message branches:

- `hello`: Task 1 echo/ack compatibility path.
- `start_game`: Task 2 path; constructs `OvercookedEngine`, constructs a
  separate `RandomPolicy`, sends `game_start`, and starts `_tick_loop()`.
- `player_action`: stores latest action in `ConnectionState.latest_player_action`.
- `end_game`: cancels the tick loop.

Current `ConnectionState` fields:

```python
engine: Optional[OvercookedEngine]
policy: Optional[RandomPolicy]
latest_player_action: Any
tick_task: Optional[asyncio.Task]
tick_timestamps: List[float]
```

There is no session/trial state in `ConnectionState` yet.

Task 4 lifecycle messages should slot into `websocket_handler()` as additional
branches:

- `start_session`
- `phase_ready`
- `submit_rating`

The tick loop currently starts immediately after `start_game` and runs
continuously until done/cancelled. Task 4 must guard or start it only during the
`PLAY` phase, per the spec's B-cat-B note.

## Additional WebSocket Coupling Observation

Task 3 added `OvercookedEngine(ai_checkpoint_id=...)` and engine-side policy
selection, but the current WebSocket path still bypasses that by creating
`conn.policy = RandomPolicy()` and passing an explicit `ai_action` into
`engine.step((player_action, ai_action))`.

For trial conditions to select HSP checkpoints through WebSocket play, Task 4
will need to either:

- construct `OvercookedEngine(layout_name=..., ai_checkpoint_id=condition.ai_checkpoint)`
  and call `engine.step((player_action, None))`, or
- replace `conn.policy` with the loaded condition policy.

The first option better matches the current Task 3 engine integration and avoids
duplicating policy ownership in `websocket.py`.

## Current Schema/Test State

`backend/data/schema.py` currently has Task 1 and Task 2 schemas only:

- `HelloMessage`, `HelloAckMessage`
- `GameState`, `PlayerSnapshot`, `ObjectSnapshot`, `JointAction`
- `StartGameMessage`, `PlayerActionMessage`, `EndGameMessage`

There are no trial lifecycle message models yet.

Existing backend tests:

- `backend/api/test_websocket.py`: health + hello/ack only.
- `backend/game/test_engine.py`
- `backend/game/test_events.py`
- `backend/game/test_ai_loader.py`

There are no trial tests yet.

## PI-3: Critical Constraints Relevant To Task 4

From `CLAUDE.md` Section 4:

1. Event timing precision ±5ms. Task 4 must keep `engine.step()` synchronous and
   must not add async/network/I/O inside game/trial hot paths. Trial state
   transitions should be synchronous; WebSocket awaits remain in `api/`.
2. AI character visual is fixed. Task 4 may route backend conditions, but must
   not expose AI checkpoint identity in participant-visible payloads.
3. Manipulation check enforced. Task 4 must implement the 4-point
   `intent_alignment` exclusion rule in `TrialManager.submit_rating()`.
4. Trial sequence pre-generated. Task 4 spec explicitly allows a hardcoded
   deterministic 2-trial sequence until Task 10. Do not randomize in app.
5. EEG/LSL out of scope. Task 4 should not call or implement LSL; logging/EEG
   hooks remain future tasks.

## Implementation Risks To Watch

- Python 3.9 compatibility: avoid runtime `X | Y` unions and other 3.10+ syntax
  despite pseudocode in the task spec.
- Existing `websocket.py` imports `Direction` but does not use it; full ruff is
  already known to fail on pre-existing lint outside Task 4.
- Existing Task 2 tests expect old `start_game` and hello behavior. Task 4 should
  preserve backward compatibility unless the spec explicitly says otherwise.
- `BREAK` auto-advance is optional/ambiguous in the spec. Client-driven timing is
  the safest default unless we choose to implement a narrowly scoped break timer.
