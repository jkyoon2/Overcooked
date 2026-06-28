# Architecture Reference — Brain-in-the-Loop Overcooked Interface

> Generated from architecture walkthrough (Sections 1–2 complete; 3–7 in progress).
> Keep in sync as new tasks land.

---

## Section 1 — Directory Responsibilities

### Backend (`backend/`)

| Directory | Responsibility |
|---|---|
| `backend/game/` | Wraps ZSC-Eval's `OvercookedGridworld` into a synchronous, deterministic step loop and emits typed `GameEvent` objects — no network, no async, no I/O. |
| `backend/api/` | Owns the network surface: one WebSocket route that drives the game tick loop, one REST route for health/config. |
| `backend/data/` | Defines the stable wire contract (Pydantic schemas) and the JSONL event writer that logs to disk. |
| `backend/trial/` | Manages the within-session state machine (INSTRUCTION → PLAY → RATING → BREAK) — currently all stubs, Task 4 scope. |
| `backend/eeg/` | Provides a single `emit(event)` stub interface so the coworker can plug in LSL without touching any game or API code. |

### Frontend (`frontend/src/`)

| Directory | Responsibility |
|---|---|
| `frontend/src/screens/` | One top-level component per trial phase (instruction, play, rating, break). |
| `frontend/src/components/` | Reusable sub-components shared across screens — currently stubs, Tasks 5–7 scope. |
| `frontend/src/state/` | Client-side trial state store (Zustand stub). |
| `frontend/src/lib/` | Low-level WebSocket client wrapper with typed discriminated-union message protocol — no React, no business logic, pure transport. |

**Smell check:** `trial/` has 3 modules for one concern — deferred clarity (Task 4 will fill them), not a structural problem.

---

## Section 2 — File-by-File Purpose Table

Covers every non-test, non-stub file with real code. Stubs (`logger.py`, `trial/`, `eeg/marker_bridge.py`) are omitted.

### Backend

| File | Purpose | Primary exports | Path |
|---|---|---|---|
| `backend/main.py` | FastAPI app factory: registers `/health` and `/ws` routes; optionally spawns the pygame verification thread on `--pygame` flag. | `app` | cold — executes once at process start |
| `backend/game/engine.py` | Wraps `OvercookedGridworld` in a synchronous step loop; owns recipe config, score accumulation, soup snapshot diffing, and event detection. | `OvercookedEngine`, `StepResult`, `ACTION_MAP` | **hot** — `engine.step()` called every tick; `_detect_events()` + `_snapshot_soups()` run inside it |
| `backend/game/events.py` | Defines the stable `GameEvent` Pydantic model — the single shared type between engine, WS serialiser, and future EEG bridge. | `GameEvent` | **hot** — instances constructed inside `_detect_events()` every tick |
| `backend/game/ai_loader.py` | Loads trained HSP/rMAPPO checkpoints from `results/Overcooked/{ttt,tto,too,ooo}/shared/rmappo/hsp-S1/seed1/`; `act(state) → Action` is deterministic and synchronous. | `HSPPolicy`, `load_policy` | **hot** — `HSPPolicy.act()` called every tick during PLAY |
| `backend/game/pygame_window.py` | Throwaway Task 2 verification window: spawns a daemon thread that renders `StateVisualizer` frames at 10 Hz; SDL `offscreen` fallback when no display exists. | `start_pygame_thread`, `set_pygame_engine` | throwaway hot — delete-safe after Task 6 |
| `backend/api/websocket.py` | Owns the entire network surface for one game session: receives client messages, drives `_tick_loop` at 10 Hz via `asyncio.create_task`, manages `ConnectionState`. | `websocket_handler`, `ConnectionState`, `_tick_loop` | **hot** — `_tick_loop` fires every 100 ms |
| `backend/data/schema.py` | Defines the full wire contract as Pydantic v2 models: game state snapshots, WS message envelopes for Task 1 (hello) and Task 2 (game) protocols. | `PlayerSnapshot`, `ObjectSnapshot`, `GameState`, `HelloMessage`, `HelloAckMessage`, `StartGameMessage`, `PlayerActionMessage`, `EndGameMessage` | **hot** — `GameState.model_dump()` called every tick |

### Frontend

| File | Purpose | Primary exports | Path |
|---|---|---|---|
| `frontend/src/lib/websocket.ts` | Pure transport layer: wraps the browser `WebSocket` API into a typed client with a discriminated-union message protocol; no React, no business logic. | `WebSocketClient`, `createWebSocketClient`, `ServerMessage`, `GameState`, `GameEvent` | **hot** — `onmessage` fires every tick; `sendPlayerAction` fires on every keydown |
| `frontend/src/screens/PlayScreen.tsx` | Renders the active game session: score HUD, recent-events list, raw JSON state dump, WASD/arrow keyboard handler; no sprite graphics (Task 6 scope). | `PlayScreen` (default) | **hot** — `onGameStep` handler updates React state every tick |
| `frontend/src/App.tsx` | Root component: manages WS connection lifecycle (connect/disconnect), polls `isOpen()` with `setInterval`, renders `<PlayScreen>` once connected. | `App` (default) | cold — connect/disconnect run once per session |

### Cross-cutting notes

- `GameState` (`backend/data/schema.py`) is the only type that crosses every layer — constructed in `engine.py`, serialised in `websocket.py`, deserialized in `websocket.ts`. Shape changes break all three simultaneously.
- `ACTION_MAP` (`engine.py`) is the sole translation point between frontend string names (`"NORTH"`) and ZSC-Eval values (`Direction.NORTH`). Only call site: `websocket.py:167`.

---

## Section 3 — Sequence Diagrams

*To be added.*

---

## Section 4 — Async Boundary Audit

*To be added.*

---

## Section 5 — Design Decisions

*To be added.*
