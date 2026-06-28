# CLAUDE.md — Brain-in-the-Loop Overcooked Experiment Interface (Error Decoding)

> Source of truth for Claude Code agents working on this sub-project.
> Last updated: 2026-06-28
>
> **Code location:** `zsceval/human_exp/error-decoding/` (backend/, frontend/, data/ 모두 이 안에 있음)

---

## Scope boundary (절대 규칙)

- `zsceval/human_exp/` 안의 error-decoding 관련 파일만 수정한다.
- `zsceval/human_exp/data-collection/`는 별개 프로젝트다. 읽지도 수정하지도 않는다.
  2인 플레이 / Flask 스택 / trajectory 수집 로직과 혼동 금지.

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

- **Backend:** Python 3.9+ (conda env `neurocontroller` runs 3.9.23; cannot upgrade without breaking PyTorch + CUDA + ZSC-Eval deps), FastAPI (for async WebSocket). All code must remain 3.9-compatible: use `from __future__ import annotations` everywhere; do NOT use 3.10+ syntax (`match/case`, `X | Y` union types in runtime contexts — `X | Y` in annotations is OK with `from __future__ import annotations`).
- **Python environment: `conda activate neurocontroller`.** This is the project-wide environment that already has PyTorch + CUDA + ZSC-Eval dependencies installed. All Python tooling (pytest, uvicorn, pip, etc.) MUST be invoked from inside this activated conda env.
  - **Do NOT create a per-folder `venv/` or `.venv/` or use `uv venv` to make a new venv.** A previous task accidentally created `backend/venv/` — delete it on sight.
  - To install new packages: `conda activate neurocontroller && pip install <package>`.
  - `pyproject.toml` is allowed for declaring dependencies and dev-tooling config (ruff, pytest), but it must NOT trigger venv creation.
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

**Root of the experiment interface code: `zsceval/human_exp/`** (currently at root; will move to `human_exp/error-decoding/` in a future migration step).

```
zsceval/human_exp/
  ├── error-decoding/           ← this sub-project (CLAUDE.md lives here)
  │   └── CLAUDE.md
  ├── backend/
  │   ├── pyproject.toml        # NO venv creation
  │   ├── main.py               # FastAPI app entrypoint
  │   ├── game/
  │   │   ├── engine.py         # Wraps OvercookedGridworld
  │   │   ├── ai_loader.py      # Load PyTorch checkpoints
  │   │   └── events.py         # Event type definitions
  │   ├── trial/
  │   │   ├── manager.py        # State machine (INSTRUCTION→PLAY→RATING→BREAK)
  │   │   ├── session.py        # Session orchestration
  │   │   └── condition.py      # 2×2 condition representation
  │   ├── data/
  │   │   ├── logger.py         # JSONL event writer
  │   │   └── schema.py         # Pydantic models
  │   ├── api/
  │   │   ├── websocket.py
  │   │   └── rest.py
  │   └── eeg/
  │       └── marker_bridge.py  # Stub for coworker's LSL integration
  ├── frontend/src/
  │   ├── screens/              # InstructionScreen, PlayScreen, RatingScreen, BreakScreen
  │   └── components/           # GameView, HatLegend, RatingForm
  └── data/                     # RUNTIME: session_orders/, checkpoints/, logs/
```

**실행 경로:**
- `run-dev.sh`: `bash zsceval/human_exp/error-decoding/run-dev.sh` (repo root 또는 어디서든)
- uvicorn, pytest: `zsceval/human_exp/error-decoding/backend/`
- npm: `zsceval/human_exp/error-decoding/frontend/`
- `zsceval.*` import: conda env에 editable 설치됨 — 경로 무관하게 작동
- `backend/data/` = code (Pydantic + logger); `error-decoding/data/` = runtime data. Do NOT merge.

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

---

## 7. Data Schema (Stable Contract — DO NOT BREAK)

### Event log entry (JSONL, one per line)
```json
{
  "timestamp_ms": 1715842934123,
  "session_id": "P001_S03",
  "trial_id": 7,
  "event_type": "ai_pickup|ai_deliver|player_pickup|player_deliver|trial_start|trial_end|instruction_shown|rating_submitted",
  "payload": {},
  "condition": { "ai_checkpoint": "tomato|onion", "player_intent": "tomato|onion" }
}
```

### Trial summary (one JSON per trial)
```json
{
  "session_id": "P001_S03",
  "trial_id": 7,
  "condition": {},
  "duration_ms": 73500,
  "deliveries": [],
  "rating": { "quality": 5, "intent_alignment": "yes_clearly|yes_somewhat|no_somewhat|no_clearly" },
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
- ❌ Do not add features beyond the current task's scope.
- ❌ Do not touch `data-collection/` code.

---

## 9. Out of Scope

- EEG hardware integration (coworker)
- LSL marker bridge (coworker — stub only)
- AI model training (separate repo)
- Decoder training pipeline (offline analysis)
- Bayesian alignment model integration (research code)

---

## 10. Task Roadmap

- **Task 1:** Project scaffolding + WebSocket hello world
- **Task 2:** Game engine integration (overcooked-ai wrapping)
- **Task 3:** AI checkpoint loader + AI step integration
- **Task 4:** Trial state machine + WebSocket protocol
- **Task 5:** Instruction screen + Hat legend UI
- **Task 6:** Play screen (game rendering)
- **Task 7:** Rating screen + manipulation check exclusion logic
- **Task 8:** Data logger (JSONL + per-trial JSON)
- **Task 9:** EEGMarkerBridge stub interface
- **Task 10:** Session order generator (Latin square)
- **Task 11:** Integration test + dry run

---

## 11. Testing & Verification

- Run `pytest` on backend before declaring task done.
- Run `npm run typecheck && npm run build` on frontend.
- Manual smoke test as described in each task spec.

---

## 12. When in doubt

- If a design decision is ambiguous, **stop and ask** rather than guess.
- Do not silently expand scope.
