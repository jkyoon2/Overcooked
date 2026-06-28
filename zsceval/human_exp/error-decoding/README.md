# Brain-in-the-Loop Overcooked Experiment Interface

Experimental interface for an HCI study on brain-in-the-loop human-AI collaboration. Participants play Overcooked with a trained RL agent while EEG is recorded. The interface drives a 2×2 factorial experiment (tomato-converged vs onion-converged AI, forced vs incentivized conditions) to collect event-locked EEG data for downstream decoder training and Bayesian alignment modeling.

All experiment interface code lives under `zsceval/human_exp/backend/` and `zsceval/human_exp/frontend/`. The ZSC-Eval engine and overcooked-ai fork live at `zsceval/envs/overcooked_new/` (do not modify).

---

## Environment Setup

This project uses the existing **conda env `neurocontroller`**. Do not create a new env, a per-folder venv, or use `uv venv`. See [CLAUDE.md Section 3](../../CLAUDE.md) for the rule.

```bash
conda activate neurocontroller
```

Verify all required imports:

```bash
python -c "import fastapi, uvicorn, pydantic, pytest, pytest_asyncio, ruff; print('ok')"
python -c "import torch; print(torch.__version__)"
python -c "import zsceval; print(zsceval.__file__)"
python -c "from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld; print('ok')"
python -c "import pygame; print(pygame.version.ver)"
```

---

## Install (one-time)

From inside the activated conda env, install backend deps:

```bash
conda activate neurocontroller
pip install "fastapi>=0.111.0" "uvicorn[standard]>=0.30.0" "pydantic>=2.7.0" \
    "websockets>=12.0" "pytest>=8.2.0" "pytest-asyncio>=0.23.0" \
    "httpx>=0.27.0" "ruff>=0.4.0" "starlette"
```

Install `zsceval` as an editable package so `from zsceval.envs...` imports resolve (run once from repo root):

```bash
conda activate neurocontroller
cd /path/to/neurocontroller  # repo root
pip install -e . --no-deps
```

Frontend (one-time):

```bash
cd zsceval/human_exp/frontend && npm install
```

---

## Run (one shot)

```bash
bash zsceval/human_exp/run-dev.sh
# → backend  http://localhost:8001  (uvicorn --reload)
# → frontend http://localhost:5173  (Vite dev server)
# Ctrl+C tears both down together.
```

The script activates the `neurocontroller` conda env, starts uvicorn in the background, waits for `/health`, then runs `npm run dev` in the foreground.  `VITE_BACKEND_PORT` is passed to Vite automatically so the frontend talks to the right port.

Override ports via env vars:

```bash
BACKEND_PORT=8002 FRONTEND_PORT=5174 bash zsceval/human_exp/run-dev.sh
```

---

## Run backend or frontend on their own (optional)

```bash
# Backend only
conda activate neurocontroller
cd zsceval/human_exp
uvicorn backend.main:app --reload --port 8001

# Frontend only — point Vite at whatever backend port you used
cd zsceval/human_exp/frontend
cp .env.example .env       # one-time; gitignored
VITE_BACKEND_PORT=8001 npm run dev
```

---

## Open in browser

1. Open [http://localhost:5173](http://localhost:5173) in Chrome (or any modern browser).
2. The IdleScreen appears with `Press ENTER to start`.
3. Press **Enter** (or click *Start session*) to begin Phase 1 (8 trials).
4. Phase 1 cycle per trial: Instruction (10s) → Play (≤75s) → Rating (20s) → Break (5s).
5. After the 8th trial, Phase 2 (replay + misalignment annotation) starts automatically.
6. Phase 2: replay each trial, drag across the timeline to mark misaligned intervals.
   Drag the small handles at either edge of a saved segment to adjust its endpoints.
   Click *Finish Phase 2* to save annotations under `data/logs/phase2/{session_id}.json`.

### Troubleshooting

- **`Connection lost. Retrying automatically…`** — the frontend cannot reach the backend WebSocket. Check what is on the backend port:
  ```bash
  lsof -nP -iTCP:8001 -sTCP:LISTEN
  # or
  ss -tlnp | grep 8001
  ```
  If a non-uvicorn process owns the port (e.g. `python3 -m http.server 8000 --directory analysis/dashboard` already squats on 8000 on this machine), pick another port: `BACKEND_PORT=8002 bash run-dev.sh`.
- **`run-dev.sh` exits with `port … is already in use`** — same fix: rerun with a different `BACKEND_PORT` / `FRONTEND_PORT`.
- **Backend reload restarts session** — uvicorn `--reload` drops in-memory state. Refresh the browser tab to reset the IdleScreen.
- **Checkpoint load error** — confirm MEP weights exist at `results/Overcooked/{tto,too}/shared/mep/mep-S1-s5/seed1/models/mep{1..4}/actor_periodic_10000000.pt`.

---

## Run Tests

```bash
conda activate neurocontroller
# From repo root neurocontroller/
cd zsceval/human_exp
python -m pytest backend/ -v

# Frontend
cd zsceval/human_exp/frontend && npm run typecheck && npm run build
```

---

## Import Convention

All backend modules use `backend.*` absolute imports (e.g. `from backend.api.websocket import ...`). Always run uvicorn and pytest from `zsceval/human_exp/` so that `backend` resolves as a package. The repo root `neurocontroller/` must also be on `PYTHONPATH` for `zsceval.*` imports — the conda editable install handles this automatically.

**Forbidden:** `uv venv`, `python -m venv`, per-folder `.venv/`. See [CLAUDE.md Section 3](../../CLAUDE.md).

---

## Legacy flask experiment

The `overcooked-flask/` subdirectory is the prior flask-based implementation. It is kept for reference only and is not part of the current interface.
