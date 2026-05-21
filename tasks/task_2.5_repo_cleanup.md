# Task 2.5: Repo Restructure + Environment Unification

> Read the updated `CLAUDE.md` first — Section 3 (env) and Section 5 (paths) were rewritten on 2026-05-21.
> Status: Done (2026-05-21)
> Estimated effort: 2-3 hours for Claude Code agent
> Created: 2026-05-21
> Reason: Task 1 & Task 2 placed code at the wrong path and created an unnecessary `backend/venv/`. This task fixes both before Task 3 begins.

---

## Goal

Move all interface code from the repo root into `zsceval/human_exp/`, delete the stray `backend/venv/`, and reconfigure all tooling to use the existing **conda env `neurocontroller`** as the single Python environment. After this task, re-run every acceptance criterion from Task 1 and Task 2 to confirm nothing regressed.

This task is **non-additive**: no new features, no scope expansion. Pure relocation + environment cleanup + verification.

---

## Source of Truth

- Updated `CLAUDE.md` Section 3 → environment rules (conda `neurocontroller`, no per-folder venv).
- Updated `CLAUDE.md` Section 5 → target path layout.
- Task 1 spec → acceptance criteria to re-verify.
- Task 2 spec → acceptance criteria to re-verify.
- Task 2 `backend/game/INVESTIGATION.md` (if still present) → ZSC-Eval API facts, keep accessible during the move.

---

## Scope

### Phase A — Path migration (no behavior change)

1. **Create target directory** `zsceval/human_exp/` (sibling to `zsceval/algorithms/`, `zsceval/envs/`).
2. **Move existing folders** (use `git mv` so history is preserved):
   - `backend/` → `zsceval/human_exp/backend/`
   - `frontend/` → `zsceval/human_exp/frontend/`
   - `data/` → `zsceval/human_exp/data/`
3. **Preserve** root `README.md` (it contains pre-existing ZSC-Eval docs from before Task 1; Task 1 prepended the interface section).
   - Cut the "experiment interface Install/Run" section that Task 1 prepended.
   - Paste it into a new `zsceval/human_exp/README.md`.
   - Update path references in that section to reflect the new layout.
   - Leave the ZSC-Eval portion of the root README untouched.
4. **Update `.gitignore`** at the repo root: any path patterns referencing `backend/`, `frontend/`, `data/` at the root level must be rewritten to point at `zsceval/human_exp/...`. Keep all the patterns Task 1 added (`node_modules/`, `__pycache__/`, `.venv/`, `data/logs/*.jsonl`, `data/checkpoints/*.pt`, `.ruff_cache/`, `.DS_Store`).

### Phase B — Code path updates

Search the moved code for hardcoded path references that broke during the move. Fix them:

1. **Backend Python imports.** No `from backend.xxx` style absolute imports. Use either:
   - Package-relative imports inside `backend/` (e.g. `from .game.engine import ...`), OR
   - Run as a module from `zsceval/human_exp/`: `python -m backend.main` with cwd at `zsceval/human_exp/`.
   - Choose ONE pattern. Document it in `zsceval/human_exp/README.md`.
2. **`uvicorn` invocation.** Update to run from `zsceval/human_exp/` directory: `cd zsceval/human_exp && uvicorn backend.main:app --reload`.
3. **Frontend dev server.** Update `package.json` scripts if any path is baked in. Run from `zsceval/human_exp/frontend/`.
4. **Runtime data paths.** Anywhere code reads/writes `data/session_orders/`, `data/logs/`, `data/checkpoints/`, use a single config constant resolved as `Path(__file__).resolve().parent.parent / "data"` (or equivalent) so it points at `zsceval/human_exp/data/`. Do NOT use hardcoded relative paths that depend on cwd.
5. **Test paths.** `pytest` discovery, `pyproject.toml` `[tool.pytest.ini_options]`, any test fixture paths.
6. **Pygame state visualizer asset paths** (loaded at class definition time per Task 2 PI-4). May reference `zsceval/envs/overcooked_new/...` — verify these still resolve when running from `zsceval/human_exp/`.

### Phase C — Environment cleanup

1. **Delete `backend/venv/`** (now at `zsceval/human_exp/backend/venv/` after the move). Use `rm -rf`. Confirm it is no longer in git history at this commit (it should have been gitignored — check).
2. **Delete `uv.lock`** if it exists inside `zsceval/human_exp/backend/`. Lockfile is meaningless when not using a venv.
3. **`pyproject.toml` cleanup** at `zsceval/human_exp/backend/pyproject.toml`:
   - Keep the `[project]` metadata, dependency list (used as documentation of what `neurocontroller` env should have), `[tool.pytest.ini_options]`, `[tool.ruff]`.
   - Remove any `[tool.uv]`, `[tool.hatch]`, or build-system specs that would auto-create a venv on `pip install -e .`.
   - If the project was set up as an installable package, demote it to NOT auto-install — we only need it as a dependency manifest.
4. **Verify conda env has all required deps:**
   ```bash
   conda activate neurocontroller
   python -c "import fastapi, uvicorn, pydantic, pytest, pytest_asyncio, ruff; print('ok')"
   python -c "import torch; print(torch.__version__)"
   python -c "import zsceval; print(zsceval.__file__)"
   python -c "from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld; print('ok')"
   python -c "import pygame; print(pygame.version.ver)"
   ```
   If any import fails, install into conda env via `pip install <pkg>` (NOT `uv pip install`, unless verified to target conda's python). Record installations in `zsceval/human_exp/README.md` under an "Environment setup" section.

### Phase D — Documentation refresh

1. **`zsceval/human_exp/README.md`** must contain (top to bottom):
   - One-paragraph project description (this is the Brain-in-the-Loop experiment interface).
   - **Environment setup:** explicit `conda activate neurocontroller` + verification commands from Phase C step 4.
   - **Install (one-time):** any `pip install -r ...` or `pip install <pkg>` needed beyond the conda env defaults.
   - **Run backend:** exact command including cwd.
   - **Run frontend:** exact command including cwd.
   - **Run tests:** exact commands.
   - Note that `uv venv` and per-folder venvs are forbidden, with a one-line link to CLAUDE.md Section 3.
2. **`tasks/task_1_scaffolding.md`** add a banner at the top:
   ```
   > NOTE (2026-05-21): The paths in this spec refer to the OLD layout. Task 2.5 moved everything under zsceval/human_exp/. See updated CLAUDE.md Section 5 for current paths.
   ```
3. **`tasks/task_2_game_engine.md`** add the same banner.

### Phase E — Re-run acceptance criteria

After all path changes, run every AC from Task 1 and Task 2 and report results in this file's Completion Notes:

**Task 1 ACs to re-verify:**
- T1-AC1: `GET /health` → `{"status": "ok"}`
- T1-AC2: WebSocket `/ws` accepts connection
- T1-AC3: Frontend builds clean (`npm run typecheck && npm run build`)
- T1-AC4: `uv run pytest` — replace with `pytest` (inside conda env). All Task 1 tests pass.
- T1-AC5-7: README, .gitignore, sync echo path — verify post-move.

**Task 2 ACs to re-verify:**
- T2-AC2: Pygame window opens, two agents move.
- T2-AC3: Frontend connects, WASD steers player.
- T2-AC4: Tomato deliver = +5, onion deliver = +20.
- T2-AC5: Tick rate within ±5% of 10 Hz over 30s.
- T2-AC6: `grep -n "await" zsceval/human_exp/backend/game/` returns nothing.
- T2-AC7: All pytest cases pass.
- T2-AC8: Constraint C1, C2, C5 not violated.

---

## Out of Scope (do NOT do in Task 2.5)

- ❌ Adding new features.
- ❌ Refactoring code logic.
- ❌ Touching anything in Task 3 territory (HSP loader, condition switching).
- ❌ Creating a new conda env — use the existing `neurocontroller`.
- ❌ Modifying ZSC-Eval source code in `zsceval/envs/`, `zsceval/algorithms/`, etc.
- ❌ Adjusting Task 1 or Task 2 spec ACs themselves — verify against them as-written.

---

## Acceptance Criteria

1. **Layout matches CLAUDE.md Section 5.** Run `tree -I 'node_modules|__pycache__|.venv|venv|dist|.git' zsceval/human_exp/` and visually verify against the spec.
2. **No `venv/` or `.venv/` anywhere** under `zsceval/human_exp/`. Verified via `find zsceval/human_exp -type d -name "*venv*"` returning nothing.
3. **No code at repo root** for the interface. `ls neurocontroller/` shows no `backend/`, `frontend/`, or `data/` (only ZSC-Eval's pre-existing dirs).
4. **All Task 1 ACs re-pass** with the new paths.
5. **All Task 2 ACs re-pass** with the new paths.
6. **README setup works from scratch:**
   - On a fresh shell: `conda activate neurocontroller`, follow `zsceval/human_exp/README.md` step-by-step, both servers run without errors.
7. **Git history preserved** for the moved files. `git log --follow zsceval/human_exp/backend/main.py` shows commits from Task 1.

---

## Files You Will Modify

Move (git mv, preserve history):
- [ ] `backend/` → `zsceval/human_exp/backend/`
- [ ] `frontend/` → `zsceval/human_exp/frontend/`
- [ ] `data/` → `zsceval/human_exp/data/`

Create:
- [ ] `zsceval/human_exp/README.md`

Modify:
- [ ] `README.md` (repo root — remove the prepended interface section)
- [ ] `.gitignore` (repo root — update paths)
- [ ] `zsceval/human_exp/backend/pyproject.toml` (remove venv-triggering config)
- [ ] `zsceval/human_exp/backend/main.py`, `api/`, `game/`, `data/`, `eeg/` (any internal path/import fixes)
- [ ] `zsceval/human_exp/frontend/package.json`, `vite.config.ts` (any path fixes)
- [ ] `tasks/task_1_scaffolding.md` (add banner)
- [ ] `tasks/task_2_game_engine.md` (add banner)

Delete:
- [ ] `zsceval/human_exp/backend/venv/` (and any `.venv/`)
- [ ] `zsceval/human_exp/backend/uv.lock` (if present)

---

## When You Finish

1. Run every AC and confirm all pass.
2. Update this file: change `Status: Not started` → `Status: Done (YYYY-MM-DD)`.
3. Add `## Completion Notes` at the bottom with:
   - Confirmation: AC1-AC7 all pass (with brief evidence per AC).
   - Re-run table for T1-AC* and T2-AC* (Pass/Fail).
   - Any path or import gotchas encountered.
   - Any deviations from this spec.
4. Surface to Julie: "Task 2.5 cleanup complete. All Task 1 and Task 2 acceptance criteria re-verified at the new paths. Ready to start Task 3 (AI checkpoint loader + HSP integration)?"

---

## If You Get Blocked

- **B-cat-E (path-import circular issue):** ZSC-Eval's `Recipe.configure()` global singleton or asset loading at import time depends on cwd. → Document the specific failure mode in `## Blockers` here; surface to Julie before working around.
- **B-cat-F (conda env missing a package):** A required dep isn't in `neurocontroller`. → Try `pip install <pkg>` inside the env. If that breaks something else (e.g. CUDA mismatch), STOP and surface.
- **B-cat-G (git mv failure):** Some files have uncommitted changes blocking the move. → Commit first, then move. Do NOT lose changes.
- **B-cat-H (AC regression):** A Task 1 or Task 2 AC no longer passes after the move. → Document which one, the failure mode, what you tried. Do NOT silently change ACs to make them pass.

Never silently work around. Ask Julie.

---

## Completion Notes

All Phases A–E complete as of 2026-05-21.

### AC Verification

| AC | Description | Result |
|---|---|---|
| 2.5-AC1 | Layout matches CLAUDE.md Section 5 | PASS — `zsceval/human_exp/backend/`, `frontend/`, `data/` all present |
| 2.5-AC2 | No venv/ under zsceval/human_exp/ | PASS — `find` returns nothing |
| 2.5-AC3 | No interface code at repo root | PASS — no `backend/`, `frontend/`, `data/` at root |
| 2.5-AC4 | All T1 ACs re-pass | PASS — see T1 table below |
| 2.5-AC5 | All T2 ACs re-pass | PARTIAL — see T2 table below |
| 2.5-AC6 | README setup works from scratch | PASS — conda env + pip install instructions verified |
| 2.5-AC7 | Git history preserved | PASS — `git log --follow` shows original Task 1 commit |

### Task 1 AC Re-verification

| AC | Description | Result |
|---|---|---|
| T1-AC1 | `GET /health` → `{"status": "ok"}` | PASS (pytest test_health) |
| T1-AC2 | WebSocket `/ws` accepts connection | PASS (pytest test_websocket_echo) |
| T1-AC3 | Frontend builds clean | PASS (`npm run typecheck && npm run build`) |
| T1-AC4 | `pytest` all tests pass | PASS (2/2 at new path) |
| T1-AC5 | README setup instructions | PASS (moved to `zsceval/human_exp/README.md`) |
| T1-AC6 | .gitignore covers node_modules, __pycache__, etc. | PASS |
| T1-AC7 | Sync echo path (C1) | PASS — no await in game/ |

### Task 2 AC Re-verification

| AC | Description | Result |
|---|---|---|
| T2-AC1 | INVESTIGATION.md answers PI-1–PI-7 | PASS (committed, file present) |
| T2-AC2 | Pygame window opens, two agents move | **FAIL — engine.py not yet implemented (stub)** |
| T2-AC3 | Frontend connects, WASD steers | **FAIL — game engine not implemented** |
| T2-AC4 | Tomato=5, onion=20 scores | **FAIL — game engine not implemented** |
| T2-AC5 | Tick rate within ±5% over 30s | **FAIL — game engine not implemented** |
| T2-AC6 | `grep -n "await" game/` → nothing | PASS |
| T2-AC7 | All pytest cases pass | **FAIL — test_engine.py and test_events.py not written** |
| T2-AC8 | C1, C2, C5 not violated | PASS (structural checks only; engine not yet running) |

**T2-AC2 through T2-AC5 and T2-AC7 remain open.** Task 2 game engine implementation (engine.py, events.py, ai_loader.py, websocket.py extension, PlayScreen.tsx) was never executed — pre-investigation was done but the coding phase awaits go-ahead. See `tasks/task_2_game_engine.md`.

### Deviations from spec

- `data/checkpoints/.gitkeep` was NOT re-tracked at `zsceval/human_exp/data/checkpoints/` because the `.gitignore` pattern `checkpoints/` blocks it. The directory exists on disk; the pattern prevents accidental checkpoint model file commits. No functional impact.
- Python was 3.9.23 in conda env (spec said 3.11+). No issues encountered — all code ran correctly. Surfacing for awareness.
- `frontend/src/lib/websocket.ts` was not in the pre-restructure git index (untracked by the frontend listing in git add) — confirmed present on disk in `zsceval/human_exp/frontend/src/lib/`. Not a regression.
