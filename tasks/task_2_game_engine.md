> NOTE (2026-05-21): The paths in this spec refer to the OLD layout. Task 2.5 moved everything under zsceval/human_exp/. See updated CLAUDE.md Section 5 for current paths.

# Task 2: Game Engine Integration (ZSC-Eval / overcooked-ai wrap)

> Read `CLAUDE.md` first. Confirm constraints C1-C5 before writing code.
> Status: Not started
> Estimated effort: 4-6 hours for Claude Code agent
> Created: 2026-05-21

---

## Goal

Wrap the existing **ZSC-Eval / overcooked-ai** game engine in a thin Python layer (`backend/game/engine.py`) so the experiment backend can:

1. Drive a kitchen episode forward at a fixed tick rate (10 Hz).
2. Override recipe scores to the experiment's values (tomato soup = 5, onion soup = 20).
3. Emit structured game events (`player_pickup`, `player_deliver`, `ai_pickup`, `ai_deliver`) from inside the synchronous step loop — these will later be timestamped to ±5 ms for EEG sync (Task 9).
4. Broadcast game state to the frontend over WebSocket at the tick rate.
5. Receive player keyboard actions from the frontend over WebSocket and feed them into the step loop.
6. Visually verify the engine is running by launching the **ZSC-Eval pygame renderer** in a separate window when `--pygame` flag is set.

Task 2 is **headless engine + plumbing + throwaway pygame verification**. No web-based game rendering yet (Task 6).

---

## Pre-Investigation (FIRST STEP — do this before writing any wrapper code)

The Claude Code agent **must** complete this investigation phase and report findings before implementation. ZSC-Eval is the base of this repository; its API shape determines the wrapper.

### Pre-Investigation Checklist

- [ ] **PI-1:** Locate the ZSC-Eval root in the repo. Identify how it imports `overcooked-ai` (is it a git submodule, a pip dependency, a vendored copy, or fully fused?).
- [ ] **PI-2:** Identify the `OvercookedEnv` / `MDP` / equivalent class that ZSC-Eval uses to step the game. Report the import path.
- [ ] **PI-3:** List the available layout names (e.g. `ttt`, `tto`, `ooo`). Report which layouts ZSC-Eval has actually been validated on.
- [ ] **PI-4:** Locate the pygame renderer. Is it ZSC-Eval's own or the upstream `overcooked-ai` visualizer (`overcooked_ai_py.visualization.state_visualizer`)? Report the import path and how it's normally invoked.
- [ ] **PI-5:** Identify how recipes and scores are configured. Is there a `recipe_config` dict, a `score_dict`, or hardcoded values? Report the modification point for the (5, 20) override.
- [ ] **PI-6:** Identify the action space — confirm it is the 6-action discrete set `{NORTH, SOUTH, EAST, WEST, STAY, INTERACT}` and report the exact encoding (string vs int).
- [ ] **PI-7:** Confirm whether ZSC-Eval already loads HSP checkpoints (relevant for Task 3) — note the loader entry point but do NOT use it in Task 2.

Write findings in `backend/game/INVESTIGATION.md` (delete this file before declaring Task 2 done — it's a scratch note, not a deliverable). Then proceed.

**If PI-1 finds no overcooked-ai dependency at all (i.e. ZSC-Eval is something else than what we think), STOP and surface to Julie.**

---

## Scope

### Backend

1. **`backend/game/engine.py` — `OvercookedEngine` class:**
   - `__init__(layout_name: str, tomato_reward: int = 5, onion_reward: int = 20, seed: int | None = None)`
   - `reset() -> GameState` — start a fresh episode, return initial state.
   - `step(joint_action: tuple[Action, Action]) -> StepResult` — advance one tick.
     - `StepResult` includes: `next_state`, `rewards: tuple[float, float]`, `done: bool`, `events: list[GameEvent]`, `step_index: int`.
   - `render_frame(state) -> bytes | None` — optional, returns PNG bytes for pygame window. Only used in pygame verification mode.
   - **Synchronous only.** No `async`, no `await`, no I/O inside `step()` (constraint C1).

2. **`backend/game/events.py` — Pydantic models for game events:**
   ```python
   class GameEvent(BaseModel):
       event_type: Literal["player_pickup", "player_deliver", "ai_pickup", "ai_deliver", "soup_drop", "soup_cooked"]
       step_index: int
       agent_id: Literal[0, 1]  # 0 = player, 1 = AI (mapping fixed here, hat color in frontend)
       payload: dict  # event-specific fields (recipe type, item type, position)
   ```
   Event detection happens inside `engine.step()` by diffing `prev_state` vs `next_state`. Do NOT emit events from async code paths.

3. **`backend/game/ai_loader.py` — Random policy placeholder:**
   - `class RandomPolicy: def act(self, state) -> Action: ...` returns a uniformly random action from the 6 actions.
   - This is a stub for Task 3 (HSP checkpoint loading). Do NOT touch real checkpoints.

4. **`backend/data/schema.py` — Extend with:**
   - `GameState` Pydantic model (serializable kitchen state — positions, holding objects, pot contents, score, time remaining).
   - `JointAction` model.
   - WebSocket message types: `game_start`, `game_step`, `player_action`, `game_end`.

5. **`backend/api/websocket.py` — Extended protocol (see WS Protocol section below).**
   - One additional async loop per connection: **tick driver** that calls `engine.step()` every 100 ms (10 Hz) and broadcasts state.
   - Player action receive path: parse `player_action` message, queue into a `latest_player_action: Action` slot (overwrite, never queue — keep latency low).
   - Step loop must read `latest_player_action`, call `RandomPolicy.act()` for AI agent, then `engine.step((player_action, ai_action))` **synchronously**.

6. **`backend/main.py` — Add `--pygame` CLI flag:**
   - When set, spawn a pygame window in a separate thread (or process) that polls the engine's current state and renders it. This is the **verification deliverable** for AC1.
   - When not set (default), backend runs headless.

7. **`pyproject.toml` — Add `pygame` to dev-deps** (not main deps — it's throwaway).

### Frontend

1. **`frontend/src/lib/websocket.ts` — Extend WS client:**
   - Type-safe message handlers for `game_start`, `game_step`, `game_end`.
   - Send `player_action` on keyboard events (W/A/S/D = NORTH/WEST/SOUTH/EAST, Space = INTERACT, otherwise STAY).

2. **`frontend/src/screens/PlayScreen.tsx` — Temporary raw display:**
   - Show received `game_state` as pretty-printed JSON in a `<pre>` block.
   - Show running score, step index, time remaining as plain numbers.
   - **No grid/sprite rendering.** That's Task 6.
   - Capture keyboard events on this screen and send them via the WS client.

3. **`frontend/src/App.tsx` — Wire PlayScreen as the default route** (replacing the Task 1 hello/ack demo). Keep a "Connect" button to open the WS session.

---

## WebSocket Protocol (Task 2 additions)

### Client → Server

```json
{ "type": "start_game", "payload": { "layout": "tto" } }
```
Server starts an episode and begins broadcasting state at 10 Hz.

```json
{ "type": "player_action", "payload": { "action": "NORTH" } }
```
Server overwrites the latest_player_action slot.

```json
{ "type": "end_game", "payload": {} }
```
Server stops the tick loop.

### Server → Client

```json
{ "type": "game_start", "payload": { "layout": "too", "initial_state": { ... } } }
```

```json
{
  "type": "game_step",
  "payload": {
    "step_index": 42,
    "state": { ... },
    "events": [ {"event_type": "player_pickup", "agent_id": 0, "payload": {...}} ],
    "rewards": [0, 5],
    "server_timestamp_ms": 1716285600123
  }
}
```

```json
{ "type": "game_end", "payload": { "step_index": 600, "final_score": 35 } }
```

**Synchronous path requirement (C1):** Inside the tick coroutine in `websocket.py`, the sequence MUST be:
```
read latest_player_action  →  random_policy.act()  →  engine.step()  →  serialize state  →  await websocket.send_json()
```
Only `send_json` is allowed to `await`. No other I/O between step and serialize.

---

## Out of Scope (do NOT build in Task 2)

- ❌ Real HSP checkpoint loader → Task 3
- ❌ Trial state machine (INSTRUCTION/PLAY/RATING/BREAK) → Task 4
- ❌ Instruction screen, Hat legend, Rating form → Tasks 5-7
- ❌ **Web-based game rendering** (sprites, grid) → Task 6. Frontend in Task 2 shows raw JSON only.
- ❌ Data logger (JSONL) → Task 8
- ❌ EEG marker bridge integration → Task 9
- ❌ Session order loading → Task 10
- ❌ Counterbalancing logic → Task 10

Pygame renderer in Task 2 is a **throwaway verification tool**. Do not invest in making it pretty. Do not let it leak into other tasks.

---

## Acceptance Criteria

1. **Pre-investigation report exists** at `backend/game/INVESTIGATION.md` answering PI-1 through PI-7. (Deleted before final commit, but committed at least once on a feature branch so Julie can review.)

2. **Engine starts and runs:**
   ```bash
   cd backend && uv run uvicorn main:app --reload -- --pygame
   ```
   - Pygame window opens within 3 seconds, shows kitchen layout with two agents (same visual; condition swap is Task 3).
   - Random AI agent visibly moves around.

3. **Frontend connects and drives:**
   ```bash
   cd frontend && npm run dev
   ```
   - Browser opens PlayScreen. Press "Connect" → WS opens.
   - Press "Start Game" → pygame window starts ticking, PlayScreen JSON updates ~10 times per second.
   - Press W/A/S/D in browser → player character in pygame moves correspondingly within 1 step (≤100 ms).
   - Press Space near a counter with an ingredient → INTERACT event fires; PlayScreen `events` array shows `player_pickup`.

4. **Recipe scores correct:**
   - Deliver a tomato soup → reward = 5, score increments by 5.
   - Deliver an onion soup → reward = 20, score increments by 20.
   - Verify by inspecting `events` and `rewards` in the WS messages.

5. **Tick rate within ±5%:**
   - Backend logs (or pygame window overlay) shows measured tick rate. Run for 30 seconds, mean tick rate ∈ [9.5, 10.5] Hz.
   - Implement this measurement explicitly — don't skip.

6. **Synchronous step path verified:**
   - `grep -n "await" backend/game/` returns NOTHING.
   - `grep -n "await" backend/api/websocket.py` — only `await websocket.send_json(...)` and `await websocket.receive_json(...)` allowed. Document each remaining `await` with an inline comment justifying it.

7. **Tests pass:**
   - `cd backend && uv run pytest` — at least:
     - `test_engine.py::test_reset_returns_valid_state`
     - `test_engine.py::test_step_increments_step_index`
     - `test_engine.py::test_tomato_delivery_yields_5`
     - `test_engine.py::test_onion_delivery_yields_20` (use a scripted deterministic action sequence to deliver an onion soup, OR mock the state to simulate a delivery event)
     - `test_events.py::test_pickup_event_emitted_on_holding_change`
   - `cd frontend && npm run typecheck && npm run build` — clean.

8. **No constraint violations:**
   - C1 (timing): synchronous step path enforced (see AC6).
   - C2 (visual blinding): both agents use the same sprite/character. AI's visual MUST NOT differ from player's by anything that could leak the AI condition (in Task 2 there's no condition swap yet, but the wrapper interface should already accept a `ai_checkpoint_id` parameter on `__init__` so Task 3 plugs in cleanly — leave it as `None` for now and add a comment).
   - C5 (EEG bridge): No `EEGMarkerBridge` calls yet — Task 9. But events must already be emitted in a form the bridge can consume later (i.e., `GameEvent` Pydantic model with `step_index` and structured payload).

---

## Files You Will Create / Modify (checklist)

Backend (new):
- [ ] `backend/game/INVESTIGATION.md` (delete before final commit; commit once for Julie review)
- [ ] `backend/game/engine.py` (real implementation, replacing Task 1 stub)
- [ ] `backend/game/events.py` (real implementation, replacing stub)
- [ ] `backend/game/ai_loader.py` (RandomPolicy only — stub for Task 3)
- [ ] `backend/game/test_engine.py`
- [ ] `backend/game/test_events.py`
- [ ] `backend/game/pygame_window.py` (throwaway visualization launcher)

Backend (modify):
- [ ] `backend/main.py` (add `--pygame` flag handling)
- [ ] `backend/api/websocket.py` (extend protocol, add tick driver)
- [ ] `backend/api/test_websocket.py` (extend tests for new message types)
- [ ] `backend/data/schema.py` (add GameState, JointAction, new WS message models)
- [ ] `backend/pyproject.toml` (add pygame to dev-deps; add ZSC-Eval / overcooked-ai dependency entry confirmed from PI-1)

Frontend (modify):
- [ ] `frontend/src/lib/websocket.ts` (handlers for game_start / game_step / game_end / player_action)
- [ ] `frontend/src/screens/PlayScreen.tsx` (raw JSON + keyboard capture; replaces stub)
- [ ] `frontend/src/App.tsx` (default route → PlayScreen)

---

## Code Style Reminders

- **Python:** `from __future__ import annotations`; full type hints; Pydantic for ALL game state and event schemas; ruff clean.
- **TypeScript:** strict mode; no `any`; discriminated union for WS messages (`type` field).
- **Tests co-located:** `test_*.py` next to source.
- **Inline comments at every `await`** in the tick path explaining why it's necessary.

---

## When You Finish

1. Run all acceptance checks above.
2. Update this file: `Status: Not started` → `Status: Done (YYYY-MM-DD)`.
3. Add `## Completion Notes` at the bottom with:
   - Pre-investigation findings summary (one paragraph).
   - Any deviations from this spec and why.
   - Which layout(s) you successfully ran.
   - Measured tick rate over 30s.
   - Any design questions for Julie.
   - Confirmation that out-of-scope items were NOT touched.
4. Surface to Julie: "Task 2 complete. Ready to start Task 3 (AI checkpoint loader + HSP integration)?"

---

## If You Get Blocked

Stop and write the blocker under `## Blockers` at the bottom of this file. Categories most likely to block:

- **B-cat-A (engine API mismatch):** ZSC-Eval / overcooked-ai API doesn't expose what we need (e.g. recipe override hook missing). → Document the API gap; do not fork the library — surface to Julie for adapt vs wrap decision revisit.
- **B-cat-B (tick rate unstable):** 10 Hz drifts beyond ±5%. → Document measurements; check if synchronous step itself is too slow vs scheduler issue.
- **B-cat-C (pygame won't launch headlessly in dev):** May need `SDL_VIDEODRIVER` env var on the dev machine. Document and continue with no-pygame mode if blocked.
- **B-cat-D (HSP coupling):** ZSC-Eval forces an AI checkpoint at engine init. → Use a no-op AI wrapper; Task 3 will plug in real checkpoint.

Do not silently work around. Ask Julie.
