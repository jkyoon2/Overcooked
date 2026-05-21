# CLAUDE.md — Brain-in-the-Loop Overcooked Experiment Interface

> Source of truth for Claude Code agents working on this codebase.
> Last updated: 2026-05-18

---

## 1. Project Context

This is the **experimental interface** for an HCI study on brain-in-the-loop human-AI collaboration. Participants play Overcooked with a trained RL agent while EEG is recorded. The interface drives a 2×2 factorial experiment to collect event-locked EEG data for downstream decoder training and Bayesian alignment modeling.

**Research goal (not your scope to implement):** Decode player's prediction-error signal during AI misalignment events; integrate decoded signal into a Bayesian alignment model that improves AI performance.

**Your scope:** Build the interface that (a) presents trials to participants, (b) plays Overcooked with selected AI checkpoints, (c) emits precisely-timed events for EEG sync, (d) collects ratings, (e) logs everything for offline analysis.

---

## 2. Experimental Setup (Brief — for context only)

**Design:** 2×2 factorial, within-subjects, 6-7 sessions per participant.

| | AI = tomato-converged | AI = onion-converged |
|---|---|---|
| Player intent → tomato | ALIGNED | MISALIGNED |
| Player intent → onion | MISALIGNED | ALIGNED |

- **Hidden conditions:** Participant does NOT know which AI checkpoint they face. AI character visual is **fixed**; only the backend checkpoint swaps.
- **Hat colors:** Before each trial, UI shows "AI wears [hat_color_1]; you wear [hat_color_2]" to disambiguate which character the participant controls. Hat colors are FIXED per role (not per condition).
- **Recipe values:** Tomato soup = close, 5 points. Onion soup = far, 20 points.
- **Trial flow:** Instruction (10s) → Play (60-75s) → Rating (15-20s) → Inter-trial break (5-10s).
- **Per session:** 4-8 trials (counterbalanced order via Latin square).
- **Per participant:** 6-7 sessions, ~160-200 events total target.

---

## 3. Tech Stack (Decisions Already Made)

- **Backend:** Python 3.11+, FastAPI (for async WebSocket).
- **Python environment: `conda activate neurocontroller`.** This is the project-wide environment that already has PyTorch + CUDA + ZSC-Eval dependencies installed. All Python tooling (pytest, uvicorn, pip, etc.) MUST be invoked from inside this activated conda env.
  - **Do NOT create a per-folder `venv/` or `.venv/` or use `uv venv` to make a new venv.** A previous task accidentally created `backend/venv/` — delete it on sight.
  - To install new packages: `conda activate neurocontroller && pip install <package>` (or `uv pip install <package>` if `uv` is already pointing at the conda env's Python; verify with `uv pip --python $(which python) ...`).
  - `pyproject.toml` is allowed for declaring dependencies and dev-tooling config (ruff, pytest), but it must NOT trigger venv creation. If a tool insists on a venv, document the friction in the task spec and surface to Julie before working around it.
- **Frontend:** React 18 + TypeScript (strict mode) + Vite. Node-side environment is independent of conda; `npm` runs as usual.
- **Communication:** WebSocket (single persistent connection per session).
- **Data storage:** JSONL (one event per line) + JSON (per-trial summary).
- **Game engine:** Wrap ZSC-Eval's vendored `overcooked-ai` (multi-recipe fork at `zsceval/envs/overcooked_new/`). Use `OvercookedGridworld.get_state_transition()` directly; do NOT use the heavier `OvercookedEnv` wrapper. Recipe override via `Recipe.configure({'tomato_value': 5, 'onion_value': 20, ...})` is a global class-level singleton — Task 2 INVESTIGATION.md flagged this as B-cat-A; Task 3/4 must guard against cross-trial leakage.
- **AI checkpoints:** Loaded via PyTorch. Checkpoint format follows HSP algorithm output (zero-shot coordination SOTA). HSP loader entry point: `rMAPPOPolicy` + `torch.load()` in `zsceval/algorithms/r_mappo/`.

Do **not** introduce new languages, frameworks, or build systems without updating this file.

---

## 4. Critical Constraints

1. **Event timing precision:** Game events → data log target ±5ms. Achieve by emitting markers from within the synchronous game step loop. **NO async/await/network calls inside step loop.**
2. **AI character visual is FIXED.** Condition (tomato vs onion-converged) is backend-only. Visual must NOT leak condition info.
3. **Manipulation check enforced.** Rating Q2 ("Did AI follow your intent?") is the manipulation check. If participant answer disagrees with assigned condition by >1 step on a 4-point scale, mark trial `excluded=true` in the log.
4. **Trial sequence pre-generated.** Each session loads from `data/session_orders/{participant_id}_session_{n}.json`. No randomization in-app.
5. **EEG/LSL is out of scope** (coworker handles). Your job: emit events to a clean Python interface `EEGMarkerBridge.emit(event)` that the coworker will plug into LSL later.

---

## 5. Module Structure

**Root of the experiment interface code: `neurocontroller/zsceval/human_exp/`** — sibling to the existing `zsceval/algorithms/`, `zsceval/envs/`, etc. The repo root (`neurocontroller/`) is where ZSC-Eval already lives; do NOT pollute it with experiment-specific code.

```
neurocontroller/                                # repo root — ZSC-Eval lives here
  └── zsceval/
      ├── algorithms/                           # existing (HSP, rMAPPO, ...)
      ├── envs/                                 # existing (overcooked_new vendored fork)
      └── human_exp/                            # ← experiment interface root (this project)
          ├── README.md                         # how to install + run (conda env: neurocontroller)
          ├── backend/
          │   ├── pyproject.toml                # NO venv creation — see Section 3
          │   ├── main.py                       # FastAPI app entrypoint
          │   ├── game/
          │   │   ├── engine.py                 # Wraps OvercookedGridworld
          │   │   ├── ai_loader.py              # Load PyTorch checkpoints (Task 3)
          │   │   └── events.py                 # Event type definitions
          │   ├── trial/
          │   │   ├── manager.py                # State machine (INSTRUCTION→PLAY→RATING→BREAK)
          │   │   ├── session.py                # Session orchestration
          │   │   └── condition.py              # 2×2 condition representation
          │   ├── data/                         # Pydantic schemas + JSONL writers (CODE)
          │   │   ├── logger.py                 # JSONL event writer
          │   │   └── schema.py                 # Pydantic models for events, trials, ratings
          │   ├── api/
          │   │   ├── websocket.py              # WebSocket route
          │   │   └── rest.py                   # REST endpoints (config, health)
          │   └── eeg/
          │       └── marker_bridge.py          # Stub for coworker's LSL integration
          ├── frontend/
          │   ├── package.json
          │   ├── vite.config.ts
          │   ├── tsconfig.json
          │   ├── index.html
          │   └── src/
          │       ├── main.tsx
          │       ├── App.tsx
          │       ├── screens/
          │       │   ├── InstructionScreen.tsx
          │       │   ├── PlayScreen.tsx
          │       │   ├── RatingScreen.tsx
          │       │   └── BreakScreen.tsx
          │       ├── components/
          │       │   ├── GameView.tsx          # Renders game state from backend
          │       │   ├── HatLegend.tsx         # "You wear X, AI wears Y" pre-trial
          │       │   └── RatingForm.tsx
          │       ├── state/
          │       │   └── trialState.ts         # Zustand or similar, synced with backend
          │       └── lib/
          │           └── websocket.ts          # WS client wrapper
          └── data/                             # RUNTIME DATA (separate from backend/data/ which is code)
              ├── session_orders/               # Pre-generated trial sequences
              ├── checkpoints/                  # AI model weights (tomato-, onion-converged)
              └── logs/                         # Output JSONL files
```

**Path conventions:**
- All paths in code reference `zsceval/human_exp/...` (relative to repo root) or use `pathlib.Path(__file__).parent` to stay portable.
- Working directory for all `uvicorn`, `pytest`, `npm` commands: `zsceval/human_exp/backend/` or `zsceval/human_exp/frontend/`.
- Working directory for python `-m` imports from `zsceval.*`: repo root (`neurocontroller/`).
- Note the name collision between `backend/data/` (code, Pydantic + logger module) and `human_exp/data/` (runtime data, JSONL files). The code module is `backend.data.*`; the runtime directory is `zsceval/human_exp/data/`. Do NOT merge them.

Stick to this structure. Adding new top-level dirs anywhere requires updating this file.

---

## 6. Code Style

**Python:**
- Type hints everywhere (`from __future__ import annotations`).
- `ruff` for lint + format. Default config.
- `pytest` for tests. Co-locate `test_*.py` next to source.
- Pydantic for data schemas.
- No global state. Pass dependencies explicitly.

**TypeScript:**
- `strict: true` in tsconfig.
- Functional components only, no class components.
- `camelCase` for variables/functions, `PascalCase` for components/types.
- Prefer `type` over `interface` unless extending.
- No `any` without explicit `// eslint-disable` + comment.

**Naming:**
- Python: `snake_case` for everything except classes (`PascalCase`).
- File names match primary export.

---

## 7. Data Schema (Stable Contract — DO NOT BREAK)

### Event log entry (JSONL, one per line)
```json
{
  "timestamp_ms": 1715842934123,
  "session_id": "P001_S03",
  "trial_id": 7,
  "event_type": "ai_pickup" | "ai_deliver" | "player_pickup" | "player_deliver"
                | "trial_start" | "trial_end" | "instruction_shown" | "rating_submitted",
  "payload": { /* event-specific fields */ },
  "condition": { "ai_checkpoint": "tomato" | "onion",
                 "player_intent": "tomato" | "onion" }
}
```

### Trial summary (one JSON per trial)
```json
{
  "session_id": "P001_S03",
  "trial_id": 7,
  "condition": { ... },
  "duration_ms": 73500,
  "deliveries": [...],
  "rating": { "quality": 5, "intent_alignment": "yes_clearly" | "yes_somewhat" | "no_somewhat" | "no_clearly" },
  "excluded": false,
  "exclusion_reason": null
}
```

---

## 8. DO NOT (hard rules)

- ❌ Do not implement LSL or EEG hardware integration. Stop at `EEGMarkerBridge.emit()` stub.
- ❌ Do not vary AI character visual appearance based on condition.
- ❌ Do not introduce async/network/I/O inside the game step loop.
- ❌ Do not randomize trial order in-app — always read from pre-generated session_order file.
- ❌ Do not change the event log schema (Section 7) without explicit instruction.
- ❌ Do not add features beyond the current task's scope. Surface ideas in comments or a `BACKLOG.md`.

---

## 9. Out of Scope

- EEG hardware integration (coworker)
- LSL marker bridge (coworker — Task `eeg/marker_bridge.py` stub only)
- AI model training (separate repo)
- Decoder training pipeline (offline analysis, separate repo)
- Bayesian alignment model integration (research code, separate repo)

---

## 10. Task Roadmap (high-level)

- **Task 1:** Project scaffolding + WebSocket hello world — `task_1_scaffolding.md`
- **Task 2:** Game engine integration (overcooked-ai wrapping)
- **Task 3:** AI checkpoint loader + AI step integration
- **Task 4:** Trial state machine + WebSocket protocol
- **Task 5:** Instruction screen + Hat legend UI
- **Task 6:** Play screen (game rendering)
- **Task 7:** Rating screen + manipulation check exclusion logic
- **Task 8:** Data logger (JSONL + per-trial JSON)
- **Task 9:** EEGMarkerBridge stub interface (defined for coworker plug-in)
- **Task 10:** Session order generator (offline script, Latin square)
- **Task 11:** Integration test + dry run

Work tasks sequentially. Each task has its own `task_N_*.md` spec.

---

## 11. Testing & Verification

- Every task spec includes Acceptance Criteria — these are the verification steps.
- Run `pytest` on backend before declaring task done.
- Run `npm run typecheck && npm run build` on frontend.
- Manual smoke test as described in each task spec.

---

## 12. When in doubt

- If a design decision is ambiguous, **stop and ask** rather than guess. Write the question as a comment in the relevant file and surface it in your task completion summary.
- Do not silently expand scope. Stay within the current task's explicit Scope section.