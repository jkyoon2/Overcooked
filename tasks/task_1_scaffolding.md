# Task 1: Project Scaffolding + WebSocket Hello World

> Read `CLAUDE.md` first. Confirm constraints C1-C5 before writing code.
> Status: Done (2026-05-20)
> Estimated effort: 2-3 hours for Claude Code agent

---

## Goal

Set up the complete project skeleton (backend + frontend) per `CLAUDE.md` Section 5, with a minimal **WebSocket "hello world" loop** verifying end-to-end communication. No game logic, no UI screens yet — just plumbing.

---

## Scope (what to build)

### Backend
1. Python 3.11+ project initialized with `uv`.
2. Folder structure per `CLAUDE.md` Section 5 (`backend/game/`, `backend/trial/`, `backend/data/`, `backend/api/`, `backend/eeg/`).
3. `backend/main.py`: FastAPI app entry point with:
   - `GET /health` → `{"status": "ok"}`
   - `WebSocket /ws` → echo handler (see protocol below)
4. `backend/api/websocket.py`: WebSocket route handler.
5. `backend/data/schema.py`: Pydantic models for the two messages defined below.
6. Empty stub files (with single-line docstring) at each module location listed in `CLAUDE.md` Section 5 — these mark intended structure for future tasks.
7. `pyproject.toml` with dependencies pinned: `fastapi`, `uvicorn[standard]`, `pydantic`, `pytest`, `pytest-asyncio`, `ruff`.

### Frontend
1. Vite + React + TypeScript project in `frontend/`, strict mode.
2. Folder structure per `CLAUDE.md` Section 5.
3. `src/App.tsx`: minimal UI with one button "Send hello", one text area showing received messages.
4. `src/lib/websocket.ts`: WebSocket client wrapper.
5. Empty stub files at each module location.
6. `package.json` with `react`, `react-dom`, `typescript`, `vite`. No state library yet (Zustand comes in Task 4).

### Repository root
1. `README.md`: how to install dependencies, how to run backend + frontend together (two commands).
2. `.gitignore`: Python, Node, IDE artifacts, `data/logs/`, `data/checkpoints/`.
3. `data/` directory with empty `session_orders/`, `checkpoints/`, `logs/` subdirs (with `.gitkeep`).

---

## WebSocket Protocol (Task 1 only — minimal)

This is the **only** protocol for Task 1. Full game protocol comes in Task 4.

### Client → Server
```json
{ "type": "hello", "payload": { "message": "hello from frontend" } }
```

### Server → Client
```json
{ "type": "hello_ack", "payload": { "echo": "hello from frontend", "server_timestamp_ms": 1715842934123 } }
```

Behavior: server receives `hello`, immediately sends back `hello_ack` with the original message echoed and server-side timestamp.

Implementation note: **NO async I/O between receiving the message and sending the ack** — keep this synchronous path tight (foreshadowing the future game-step constraint C2).

---

## Out of Scope (do NOT build in Task 1)

- ❌ Game engine integration → Task 2
- ❌ AI checkpoint loading → Task 3
- ❌ Trial state machine → Task 4
- ❌ Any UI screen (Instruction, Play, Rating) → Tasks 5-7
- ❌ Data logger → Task 8
- ❌ EEG marker bridge → Task 9
- ❌ Counterbalanced session order generator → Task 10

If you find yourself wanting to add anything from this list, **stop**. Note the impulse in a comment or `BACKLOG.md` and stay scoped.

---

## Acceptance Criteria

1. **Backend runs cleanly:**
   ```bash
   cd backend && uv run uvicorn main:app --reload
   ```
   - `GET http://localhost:8000/health` returns `{"status": "ok"}`.
   - WebSocket connection to `ws://localhost:8000/ws` succeeds.

2. **Frontend runs cleanly:**
   ```bash
   cd frontend && npm install && npm run dev
   ```
   - Opens browser. Button "Send hello" is visible.
   - Clicking the button sends WebSocket message, receives ack within 50ms (manual check via DevTools).
   - Text area shows received `hello_ack` payload.

3. **Tests pass:**
   - Backend: `cd backend && uv run pytest` — at least one test verifying `/health` and one test for WS echo using `pytest-asyncio` and `httpx`/`websockets` test client.
   - Frontend: `cd frontend && npm run typecheck && npm run build` — both succeed with no errors.

4. **Folder structure matches `CLAUDE.md` Section 5.** Run `tree -I 'node_modules|__pycache__|.venv|dist'` from repo root; verify visually.

5. **README.md** has two clear sections: "Install" and "Run". A new researcher could clone the repo and have both servers running in under 5 minutes.

6. **`.gitignore`** properly excludes: `__pycache__/`, `.venv/`, `node_modules/`, `dist/`, `data/logs/*.jsonl`, `data/checkpoints/*.pt`, `.DS_Store`, IDE configs.

7. **No constraint violations** from `CLAUDE.md` Section 8. In particular:
   - Echo path is synchronous (no `await` other than `websocket.send_json()` which is required by FastAPI).
   - No async/network/I/O in any hot path (only one WS message, so this is trivially satisfied — but build the habit now).

---

## Code Style Reminders

- **Python:** `from __future__ import annotations`; full type hints; Pydantic models for ALL message schemas.
- **TypeScript:** strict mode; functional components only; no `any`.
- **Tests co-located:** `backend/api/test_websocket.py` next to `backend/api/websocket.py`.

---

## Files You Will Create (checklist)

Backend:
- [ ] `backend/pyproject.toml`
- [ ] `backend/main.py`
- [ ] `backend/api/websocket.py`
- [ ] `backend/api/test_websocket.py`
- [ ] `backend/api/rest.py` (stub)
- [ ] `backend/data/schema.py` (Pydantic models for hello/hello_ack)
- [ ] `backend/data/logger.py` (stub)
- [ ] `backend/game/engine.py` (stub)
- [ ] `backend/game/ai_loader.py` (stub)
- [ ] `backend/game/events.py` (stub)
- [ ] `backend/trial/manager.py` (stub)
- [ ] `backend/trial/session.py` (stub)
- [ ] `backend/trial/condition.py` (stub)
- [ ] `backend/eeg/marker_bridge.py` (stub with `EEGMarkerBridge` Protocol class signature only)

Frontend:
- [ ] `frontend/package.json`
- [ ] `frontend/vite.config.ts`
- [ ] `frontend/tsconfig.json`
- [ ] `frontend/index.html`
- [ ] `frontend/src/main.tsx`
- [ ] `frontend/src/App.tsx`
- [ ] `frontend/src/lib/websocket.ts`
- [ ] `frontend/src/screens/InstructionScreen.tsx` (stub)
- [ ] `frontend/src/screens/PlayScreen.tsx` (stub)
- [ ] `frontend/src/screens/RatingScreen.tsx` (stub)
- [ ] `frontend/src/screens/BreakScreen.tsx` (stub)
- [ ] `frontend/src/components/GameView.tsx` (stub)
- [ ] `frontend/src/components/HatLegend.tsx` (stub)
- [ ] `frontend/src/components/RatingForm.tsx` (stub)
- [ ] `frontend/src/state/trialState.ts` (stub)

Root:
- [ ] `README.md`
- [ ] `.gitignore`
- [ ] `data/session_orders/.gitkeep`
- [ ] `data/checkpoints/.gitkeep`
- [ ] `data/logs/.gitkeep`

---

## When You Finish

1. Run all acceptance checks above and confirm pass.
2. Update this file: change "Status: Not started" → "Status: Done (YYYY-MM-DD)".
3. Add a `## Completion Notes` section at the bottom with:
   - Any deviations from this spec and why.
   - Any design questions surfaced for Julie.
   - Confirmation that you did NOT touch out-of-scope items.
4. Surface to Julie: "Task 1 complete. Ready to start Task 2 (Game engine integration)?"

---

## Completion Notes

**All 7 Acceptance Criteria passed on 2026-05-20.**

- `uv` was not pre-installed; installed it via the official installer script before running `uv sync`.
- `pytest-asyncio` is wired up (`asyncio_mode = "auto"`) but the Task 1 tests use `starlette.testclient.TestClient` (synchronous) rather than async — this is simpler and reliable for WS echo tests. `pytest-asyncio` stays in dev-deps for future tasks.
- Root `README.md` already existed (ZSC-Eval docs); the experiment interface Install/Run section was prepended rather than replacing it, so ZSC-Eval docs are preserved.
- `node_modules/` was missing from `.gitignore`; added it along with `data/logs/*.jsonl`, `data/checkpoints/*.pt`, `.ruff_cache/`, `.DS_Store`.
- No out-of-scope items touched (no game engine, no AI loader, no trial state machine, no data logger, no EEG bridge implementation).

**Design questions for Julie:** None — spec was unambiguous.

---

## If You Get Blocked

Stop and write the blocker as a comment in the relevant file, plus a summary at the bottom of this task file under `## Blockers`. Do not silently work around — ask Julie.
