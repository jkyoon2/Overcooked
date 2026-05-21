# Task 1 — Project Scaffolding & Hello WebSocket

> Read `CLAUDE.md` first. This task assumes you've internalized Sections 3 (tech stack), 5 (module structure), and 8 (do-not list).

---

## Goal

Stand up a minimal but working project skeleton: backend (Python/FastAPI) + frontend (React/TS/Vite) that can run locally and communicate via WebSocket. **No business logic in this task** — just the plumbing and a "hello world" round-trip.

---

## Acceptance Criteria

A reviewer should be able to:

1. Clone the repo into a fresh machine, follow `README.md` setup steps, and reach a working dev environment in <5 minutes.
2. Run `make dev` (or two parallel commands documented in README) to start backend on `localhost:8000` and frontend on `localhost:5173`.
3. Open `localhost:5173` in a browser and see:
   - Heading: **"Brain-in-the-Loop Experiment"**
   - Subheading: **"Connecting to backend..."** initially, then **"Connected ✓"** once WebSocket is up.
   - A simple **"Send Ping"** button.
4. Click **"Send Ping"** — frontend sends `{"type": "ping"}` over WebSocket. Backend logs `[INFO] Received ping from client {client_id}`. Backend replies `{"type": "pong", "timestamp_ms": ...}`. Frontend displays the latest pong timestamp under the button.
5. Close the browser tab — backend logs `[INFO] Client {client_id} disconnected`.
6. Run `pytest` in `backend/` — at least one passing smoke test exists (e.g., test the ping/pong handler).
7. Run `npm run typecheck && npm run build` in `frontend/` — no errors, no warnings.

---

## Scope (Files to Create)

### Root
- `README.md` — setup instructions: prereqs (Python 3.11+, Node 20+, uv, npm), install steps, run commands, ports.
- `Makefile` — targets: `install`, `dev`, `dev-backend`, `dev-frontend`, `test`, `lint`.
- `.gitignore` — Python (`__pycache__`, `.venv`, etc.) + Node (`node_modules`, `dist`) + IDE (`.vscode`, `.idea`).
- `BACKLOG.md` — empty file with header; future-task surface area.

### Backend (`backend/`)
- `pyproject.toml` — managed by `uv`. Deps: `fastapi`, `uvicorn[standard]`, `websockets`, `pydantic`, `pytest`. Dev deps: `ruff`.
- `main.py` — FastAPI app. WebSocket route at `/ws`. Handles `ping` → `pong`. Logs client connect/disconnect with a generated `client_id` (use `uuid4`).
- `api/__init__.py` — empty
- `api/websocket.py` — WebSocket handler logic (factored out of main.py, importable for tests).
- `tests/__init__.py` — empty
- `tests/test_ping.py` — pytest smoke test using FastAPI's `TestClient` to verify ping/pong over WS.

### Frontend (`frontend/`)
- `package.json` — `react`, `react-dom`, `vite`, `typescript`, `@types/react`, `@types/react-dom`. Scripts: `dev`, `build`, `typecheck`, `lint`.
- `vite.config.ts` — minimal Vite config, port 5173.
- `tsconfig.json` — `"strict": true`.
- `index.html` — root mount point.
- `src/main.tsx` — React root.
- `src/App.tsx` — UI: heading + subheading + ping button + last pong timestamp display.
- `src/lib/websocket.ts` — WebSocket client wrapper with `onMessage`, `send`, `connect`. Auto-reconnect optional but document the choice.

---

## Implementation Notes

- WebSocket URL: hardcode `ws://localhost:8000/ws` in `frontend/src/lib/websocket.ts` for now. Externalize to env var in a later task.
- Use `useEffect` in `App.tsx` to connect on mount and disconnect on unmount.
- Backend timestamp: `int(time.time() * 1000)` for `timestamp_ms`. Document this convention in a comment — same precision target as event logs (Section 4 of CLAUDE.md).
- README must include exact commands for first-time setup:
  ```bash
  # backend
  cd backend && uv sync
  # frontend
  cd frontend && npm install
  # run both
  make dev
  ```

---

## Constraints (re-emphasize from CLAUDE.md)

- ❌ Do NOT add game logic, AI loading, trial state, ratings, or any business module yet.
- ❌ Do NOT add styling beyond inline minimums (a styled UI is Task 5+).
- ❌ Do NOT install state libraries (Zustand etc.) — `useState` suffices for Task 1.
- ❌ Do NOT create files outside the Scope list. If you feel the need, write the idea in `BACKLOG.md`.
- ❌ Do NOT add EEG/LSL code (out of scope per CLAUDE.md Section 9).
- ✅ DO keep the WebSocket handler factored so it can be tested in isolation.
- ✅ DO write the README assuming the reader has never seen this project.

---

## Verification (run these in order at the end)

```bash
# 1. From repo root
make install      # should succeed cleanly
make dev          # should start both servers

# 2. In browser
open http://localhost:5173
# Expected:
#  - Heading visible
#  - Subheading toggles to "Connected ✓"
#  - Click "Send Ping" → pong timestamp appears

# 3. Tests
cd backend && pytest             # at least 1 pass
cd ../frontend && npm run typecheck && npm run build   # no errors

# 4. Cleanup
# Close browser tab → check backend logs for "Client ... disconnected"
```

---

## What to Output on Completion

In your final summary message, include:
1. The exact commands a fresh user would run to reach a working dev environment.
2. Any non-obvious design decisions you made (with rationale).
3. Any questions or ambiguities you ran into — surface them, don't paper over.
4. A list of `BACKLOG.md` items you added (if any).

---

## Out of Scope for Task 1 (will be handled in later tasks)

- Trial state machine (Task 4)
- Game rendering (Task 6)
- Rating UI (Task 7)
- Data logging beyond stdout (Task 8)
- EEG marker bridge (Task 9 — stub only)
- AI checkpoint loading (Task 3)
- Authentication / participant identity (later task, separate spec)
