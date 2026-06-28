# CLAUDE.md — Human Gameplay Data Collection Interface

> Last updated: 2026-06-29

---

## Scope boundary

- `zsceval/human_exp/data-collection/` 안의 파일만 수정한다.
- `zsceval/human_exp/error-decoding/`는 별개 프로젝트. 읽지도 수정하지도 않는다.

---

## 프로젝트 개요

사람 2명이 Overcooked를 동시에 플레이하며 human-AI collaboration 훈련 데이터를 수집하는 인터페이스. AI 에이전트 없음. EEG 없음.

**세션 흐름:**
1. Participant ID 입력 (e.g. `P001`)
2. 4개 레이아웃을 아래 순서로 순차 플레이
   ```
   1_forced_easy → 1_forced_hard → 1_incentivized_easy → 1_incentivized_hard
   ```
3. 각 trial 시작 전: 레시피 선택 (아래 참조)
4. 플레이 종료 후 즉시 자동 저장

---

## Trial 구성

### 레시피 선택 (trial마다)

| 코드 | 재료 | 설명 |
|---|---|---|
| `TTT` | 토마토 × 3 | 토토토 |
| `TTO` | 토마토 × 2 + 양파 × 1 | 토토양 |
| `TOO` | 토마토 × 1 + 양파 × 2 | 토양양 |
| `OOO` | 양파 × 3 | 양양양 |

레시피는 `Recipe.configure()`로 설정한다. **전역 싱글턴이므로 trial 간 반드시 초기화해야 한다** (cross-trial leakage 방지).

### 키 바인딩 (미결 — 구현 전 확인)
- Player 0 / Player 1 각자의 키 배치 결정 필요

---

## 구현 지침

**게임 플레이 구현은 `error-decoding/`의 Phase 1(PlayScreen + game engine 루프)을 그대로 참고한다.**
변경점은 단 하나: 1인 + AI → **사람 2인**. 구체적으로:
- AI 에이전트 step 제거, 대신 Player 1의 키 입력을 Player 0과 동일한 방식으로 수집
- 두 플레이어의 action을 joint action으로 묶어 `get_state_transition()` 호출
- 그 외 렌더링·타이밍·루프 구조는 Phase 1 그대로 사용

---

## 기술 스택

- **Backend:** Flask + Python 3.9 (`conda activate neurocontroller`)
- **Frontend:** React 18 + TypeScript (strict) + Vite
- **Game engine:** `OvercookedGridworld.get_state_transition()` 직접 사용
- **저장:** 로컬 NumPy `.npz` + `.json`. 외부 전송 없음.

---

## 코드 구조

```
zsceval/human_exp/data-collection/
├── CLAUDE.md
├── backend/
│   ├── app.py                   # Flask entrypoint
│   ├── game/
│   │   ├── engine.py            # OvercookedGridworld wrapper
│   │   └── recorder.py          # 매 step obs/action/reward 누적 → npz 저장
│   └── api/
│       └── routes.py            # REST: /session/start, /step, /session/end
├── frontend/
│   └── src/
│       ├── screens/
│       │   ├── IdScreen.tsx     # Participant ID 입력
│       │   ├── RecipeScreen.tsx # trial별 레시피 선택
│       │   └── PlayScreen.tsx   # 2인 플레이
│       └── components/
│           └── GameView.tsx
└── data/
    └── trajectories/            # {participant_id}_{layout}_{recipe}.npz + _meta.json
```

---

## Data Schema

### 파일명 규칙

```
{participant_id}_{layout}_{recipe}.npz
{participant_id}_{layout}_{recipe}_meta.json
```

예: `P001_1_forced_easy_TTT.npz` / `P001_1_forced_easy_TTT_meta.json`

### Trajectory: `*.npz`

```python
np.savez_compressed(
    path,
    obs_p0    = ...,  # (T, 13, 7, 25) int8  — P0 ego-centric lossless encoding
    obs_p1    = ...,  # (T, 13, 7, 25) int8  — P1 ego-centric lossless encoding
    actions   = ...,  # (T, 2)         int8  — [[a_p0, a_p1], ...], 0=N 1=S 2=E 3=W 4=Stay 5=Interact
    rewards   = ...,  # (T,)           int16 — step별 점수 (delivery 없으면 0)
    timestamps= ...,  # (T,)           int64 — 각 step unix timestamp (ms)
)
```

- `obs_p0[t]` = `lossless_state_encoding(state_t, horizon)[0]` (action 적용 전 상태)
- 채널 25개: `lossless_state_encoding()` 참조 (overcooked_mdp.py L2588). ⚠️ `get_lossless_state_encoding_shape()`은 26 반환하지만 실제는 **25** (코드 버그)

### Metadata: `*_meta.json`

```json
{
  "participant_id": "P001",
  "layout": "1_forced_easy",
  "recipe": "TTT",
  "recipe_ingredients": {"tomato": 3, "onion": 0},
  "horizon": 400,
  "total_steps": 387,
  "total_score": 35,
  "duration_ms": 61200,
  "recorded_at": "2026-06-29T14:32:00Z"
}
```

---

## DO NOT

- ❌ `error-decoding/` 코드 참조·수정
- ❌ AI 에이전트, EEG/LSL 관련 코드
- ❌ Game step loop 내부 async/네트워크/IO
- ❌ 새 venv 생성 (conda env 전용)
- ❌ 명시적 지시 없이 Data Schema 변경
- ❌ trial 간 Recipe.configure() 초기화 생략
