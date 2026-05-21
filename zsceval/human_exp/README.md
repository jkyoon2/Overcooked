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

## Run Backend

```bash
conda activate neurocontroller
cd zsceval/human_exp
uvicorn backend.main:app --reload
# Backend runs on http://localhost:8000
```

---

## Run Frontend

```bash
cd zsceval/human_exp/frontend
npm run dev
# Dev server runs on http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173). Click **Send hello** to verify WebSocket round-trip.

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
