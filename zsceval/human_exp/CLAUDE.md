# CLAUDE.md — human_exp (공통 context)

> 이 파일은 `error-decoding/`과 `data-collection/` 양쪽에 공통으로 적용됩니다.
> 각 sub-project의 CLAUDE.md가 이 파일보다 우선합니다.
> Last updated: 2026-06-28

---

## Sub-project 구조

```
zsceval/human_exp/
  ├── error-decoding/    ← EEG 기반 1인+AI 플레이 (Brain-in-the-Loop)
  └── data-collection/   ← 2인 동시 플레이 데이터 수집 (AI 없음)
```

**두 sub-project는 완전히 독립적이다.** 어느 쪽을 작업하든 반대쪽 코드를 읽거나 수정하지 않는다.

---

## 공통 환경

**Python:** `conda activate neurocontroller` (3.9.23)
- PyTorch + CUDA + ZSC-Eval 의존성 포함.
- **절대 금지:** venv, .venv, `uv venv` 신규 생성. 이전 작업에서 `backend/venv/`가 실수로 생성된 적 있음 — 발견 즉시 삭제.
- 새 패키지: `conda activate neurocontroller && pip install <package>`
- Python 3.9-compatible 필수: `from __future__ import annotations`. `match/case`, `X | Y` runtime union 사용 금지.

**Node/Frontend:** conda와 무관. `npm` 그대로 사용.

---

## 공통 Game Engine

두 인터페이스 모두 같은 ZSC-Eval game engine을 사용한다.

- **사용법:** `OvercookedGridworld.get_state_transition()` 직접 호출. `OvercookedEnv` wrapper 사용 금지.
- **레이아웃 경로:** `zsceval/envs/overcooked_new/src/overcooked_ai_py/data/layouts/`
- **사용 레이아웃:** `1_forced_easy`, `1_forced_hard`, `1_incentivized_easy`, `1_incentivized_hard`
- **import 경로:** `from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld`
- **작업 디렉토리:** `zsceval.*` import 시 repo root(`neurocontroller/`)에서 실행.
- `zsceval/envs/`, `zsceval/algorithms/`는 **읽기 전용**. 수정 금지.

---

## 공통 Code Style

**Python:**
- Type hints everywhere (`from __future__ import annotations`).
- `ruff` for lint + format. Default config.
- `pytest` for tests. Co-locate `test_*.py` next to source.
- Pydantic for data schemas.
- No global state. Pass dependencies explicitly.
- `snake_case` for everything except classes (`PascalCase`).

**TypeScript:**
- `strict: true` in tsconfig.
- Functional components only.
- `camelCase` for variables/functions, `PascalCase` for components/types.
- Prefer `type` over `interface` unless extending.
- No `any` without explicit `// eslint-disable` + comment.

---

## 워크플로우

- 코드 수정 전 분석·계획을 먼저 제시하고 확인을 받는다. (Plan Mode 우선)
- 한 번에 한 단계씩 진행한다.
- 모호한 결정은 추측하지 않고 질문한다.
