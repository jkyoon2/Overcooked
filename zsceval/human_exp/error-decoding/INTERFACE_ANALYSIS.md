# human_exp 인터페이스 구조 분석

> 이 문서는 `zsceval/human_exp/` 인터페이스의 backend/frontend 구조를 정리한 것이다.
> 본 개편 작업 (Phase 라벨 재정의 / 8-trial MEP 세션 / IdleScreen / PlayScreen·ReplayScreen UX) 을 위한 베이스라인 스냅샷이다.

작성: 2026-06-10 / 대상 브랜치: `main` / 마지막 코드 커밋: `aebcf2a feat: decoding interface v1 complete`

---

## 1. 디렉터리 구조 & 책임

```
zsceval/human_exp/
├── README.md                     # 설치 + 실행 안내
├── CLAUDE.md                     # 프로젝트 헌법 (수정 금지 영역, schema 계약 등)
├── ARCHITECTURE.md               # 모듈 책임 매트릭스 (미완성)
├── backend/
│   ├── main.py                   # FastAPI app (/health, /ws)
│   ├── api/
│   │   └── websocket.py          # WebSocket 라우터, 모든 메시지 핸들러
│   ├── game/
│   │   ├── engine.py             # OvercookedEngine — get_state_transition 직접 호출
│   │   ├── ai_loader.py          # PyTorch 체크포인트 → HSPPolicy (R_MAPPOPolicy 래퍼)
│   │   ├── pygame_window.py      # 디버그용 throwaway 윈도우
│   │   └── events.py             # 게임 이벤트 타입
│   ├── trial/
│   │   ├── condition.py          # TrialCondition (ai_checkpoint × player_intent × model_checkpoint)
│   │   ├── manager.py            # TrialPhase enum + TrialManager 상태머신
│   │   └── session.py            # SessionManager — 현재는 4-trial 하드코딩
│   ├── data/
│   │   ├── schema.py             # 모든 WebSocket 메시지/페이로드의 Pydantic 모델
│   │   ├── phase3_store.py       # 리플레이 phase 결과 JSON 저장기
│   │   └── logger.py             # JSONL 이벤트 라이터 (Task 8)
│   ├── eeg/
│   │   └── marker_bridge.py      # 동료 LSL 통합용 stub
│   └── pyproject.toml            # ruff/pytest 설정 (venv 생성 금지)
├── frontend/
│   ├── package.json              # React 18.3.1 + Vite 5.3.4
│   ├── vite.config.ts            # 기본 dev server (포트 5173)
│   ├── public/sprites/           # terrain/chefs/objects/soups png
│   └── src/
│       ├── main.tsx
│       ├── App.tsx               # 라우팅 + WebSocket 라이프사이클 + 모든 트라이얼 상태
│       ├── lib/websocket.ts      # WebSocket 클라이언트 + 메시지 타입 정의
│       ├── components/
│       │   ├── GameView.tsx      # 캔버스 렌더 (스프라이트 시트 기반)
│       │   ├── HatLegend.tsx     # 모자 색 안내 카드
│       │   └── RatingForm.tsx    # quality + intent_alignment 폼
│       └── screens/
│           ├── InstructionScreen.tsx   # 10초 카운트다운 + 모자 안내
│           ├── PlayScreen.tsx          # HUD + GameView (현재 좌측 정렬)
│           ├── RatingScreen.tsx        # 20초 카운트다운 + RatingForm
│           ├── BreakScreen.tsx         # 5초 휴식 카운트다운
│           └── PhaseThreeScreen.tsx    # 리플레이 + 타임라인 미스얼라인먼트 선택
└── data/                         # 런타임 데이터 (코드 아님)
    ├── session_orders/           # 비어 있음 (.gitkeep)
    ├── checkpoints/              # AI 가중치 (생성 시점에 결정)
    └── logs/
        └── phase3/               # 리플레이 phase 산출물
```

---

## 2. Backend 상태머신

### 2.1 TrialPhase enum (`backend/trial/manager.py:29`)

```python
class TrialPhase(str, Enum):
    IDLE = "idle"
    INSTRUCTION = "instruction"
    PLAY = "play"
    RATING = "rating"
    BREAK = "break"
```

전이는 `TrialManager` 내부 `_require_phase()` 가드로 강제. `IDLE → INSTRUCTION → PLAY → RATING → BREAK → IDLE` 한 주기가 1 trial.

### 2.2 리플레이 phase (현 코드명 `phase3`)

별도의 phase enum이 **없다**. `SessionManager.is_complete()` 가 True가 되는 시점에 `_complete_break_or_session()` 가 `PhaseThreeStartMessage` 를 보내 frontend가 리플레이 화면으로 전환. backend는 `submit_phase3` 메시지를 기다렸다가 `save_phase_three_record()` 로 저장.

→ 본 개편에서 **"phase3" 이름을 "phase2" 로 일관 리네임**.

### 2.3 SessionManager 현황 (`backend/trial/session.py:14`)

```python
self.trials = [
    TrialCondition(ai_checkpoint="tomato", player_intent="tomato", model_checkpoint="tto_sp_seed4"),
    TrialCondition(ai_checkpoint="onion",  player_intent="tomato", model_checkpoint="tto_sp_seed5"),
    TrialCondition(ai_checkpoint="tomato", player_intent="onion",  model_checkpoint="ttt_adaptive_seed1"),
    TrialCondition(ai_checkpoint="onion",  player_intent="onion",  model_checkpoint="ttt_adaptive_seed2"),
]
```

4 trial 하드코딩. 본 개편에서 **8-trial seeded generator** 로 교체.

---

## 3. AI checkpoint 로딩 (`backend/game/ai_loader.py`)

```
load_policy(checkpoint_id, device) → HSPPolicy
  └─ _resolve_checkpoint_paths(checkpoint_id)
      └─ _EXPLICIT_CHECKPOINT_SPECS[checkpoint_id]   # 명시적 등록된 체크포인트
  └─ _load_policy_config(paths)                       # policy_config.pkl 로드
  └─ R_MAPPOPolicy(args, obs_space, share_obs_space, act_space)
  └─ policy.actor.load_state_dict(torch.load(actor_path))
  └─ (optional) policy.critic.load_state_dict(torch.load(critic_path))
```

- `_LAYOUT_BY_CHECKPOINT_ID` 가 checkpoint id → 레이아웃명 매핑 (`tto_sp_seed4` → `tto`, `ttt_adaptive_seed1` → `ttt` 등).
- `act(state)` 는 deterministic argmax, recurrent state는 매 호출 zero. 동기 함수 — tick loop 안에서 호출 가능.
- MEP 체크포인트 (`results/Overcooked/{tto,too}/shared/mep/mep-S1-s5/seed1/`) 는 **아직 등록되지 않음**. 본 개편에서 8종 (`tto_mep1~4`, `too_mep1~4`) 추가.
  - 디렉터리 확인: `policy_config.pkl` 존재 ✓, `models/mep{1..5}/actor_periodic_10000000.pt` 존재 ✓, critic 없음 → `critic_path=None` 으로 설정.

---

## 4. WebSocket 메시지 카탈로그

### 4.1 Backend → Frontend (송신)

| Type | Schema | 언제 |
|---|---|---|
| `hello_ack` | `HelloAckMessage` | `hello` 수신 시 echo |
| `trial_start` | `TrialStartMessage` | 세션 시작 또는 break 종료 후 다음 trial 시작 |
| `phase_change` | `PhaseChangeMessage` (`phase: instruction|play|rating|break`) | 페이즈 전이 시 |
| `game_start` | `{ layout, initial_state: GameState }` | PLAY 페이즈 시작 시 |
| `game_step` | `{ step_index, state, events, rewards, server_timestamp_ms }` | 10 Hz (100ms tick) |
| `game_end` | `{ step_index, final_score }` | 엔진 max_steps 도달 |
| `rating_ack` | `RatingAckMessage` (`excluded`, `exclusion_reason`) | `submit_rating` 처리 후 |
| **`phase3_start`** | `PhaseThreeStartMessage` (모든 trial 의 frames + 메타) | 모든 trial 완료 후 |
| **`phase3_complete`** | `PhaseThreeCompleteMessage` (`saved: bool`) | `submit_phase3` 저장 완료 후 |

### 4.2 Frontend → Backend (수신)

| Type | Schema | 효과 |
|---|---|---|
| `hello` | `HelloMessage` | `hello_ack` echo |
| `start_game` | `{ ai_checkpoint?, layout? }` | 엔진 생성 + tick loop 시작 (테스트용 단발) |
| `player_action` | `{ action: NORTH/SOUTH/EAST/WEST/INTERACT/STAY }` | `latest_player_action` 슬롯 덮어쓰기 |
| `end_game` | `{}` | tick loop 취소 |
| `start_session` | `{ session_id }` | `SessionManager` 생성, 첫 trial_start 송신 |
| `phase_ready` | `{}` | 현 phase 종료 신호 → 다음 phase로 전이 |
| `submit_rating` | `{ quality, intent_alignment }` | manipulation check + rating_ack + phase_change(break) |
| **`submit_phase3`** | `SubmitPhaseThreeMessage` (trial별 segment 리스트) | 저장 + phase3_complete 송신 |

**굵게 표시된 phase3 메시지 3종은 본 개편에서 phase2로 리네임.**

### 4.3 페이즈 / 메시지 흐름 다이어그램

```
Frontend                        Backend
────────                        ───────
                start_session →
                                SessionManager(session_id) 생성
                                trial_manager.start_instruction()
              ← trial_start (phase=instruction)
[InstructionScreen 10초 카운트다운]
                phase_ready  →
                                _start_play_phase()
              ← phase_change (phase=play)
              ← game_start
              ← game_step ...(10 Hz)
[PlayScreen 키보드 입력]
                player_action →
              ← game_step ...
              ← game_end (엔진 max_steps)
                phase_ready  →
                                _end_play_phase()
              ← phase_change (phase=rating)
[RatingScreen]
                submit_rating →
                                trial_manager.submit_rating() (manipulation check)
              ← rating_ack
              ← phase_change (phase=break)
[BreakScreen 5초]
                phase_ready  →
                                advance_trial()
                                  ├── 다음 trial 남음 → trial_start (반복)
                                  └── is_complete() → phase3_start (8개 trial frames 전송)
[PhaseThreeScreen 리플레이 + 미스얼라인먼트 선택]
                submit_phase3 →
                                save_phase_three_record()
              ← phase3_complete
[완료 화면]
```

---

## 5. 데이터 모델

### 5.1 TrialCondition (`backend/trial/condition.py`)

```python
AICheckpoint = Literal["tomato", "onion"]
PlayerIntent = Literal["tomato", "onion"]
ModelCheckpoint = Literal[
    "tto_sp_seed4", "tto_sp_seed5",
    "ttt_adaptive_seed1", "ttt_adaptive_seed2",
]

class TrialCondition(BaseModel):
    ai_checkpoint: AICheckpoint
    player_intent: PlayerIntent
    model_checkpoint: ModelCheckpoint = "tto_sp_seed4"

    @property
    def is_aligned(self) -> bool:
        return self.ai_checkpoint == self.player_intent
```

- `ai_checkpoint` 는 실험적 의미 (tomato-converged vs onion-converged) — 참가자에게 숨김.
- `player_intent` 는 참가자에게 제시되는 목표.
- `model_checkpoint` 는 실제 로드할 가중치 식별자 (`ai_loader.py` 의 `_EXPLICIT_CHECKPOINT_SPECS` 키).

### 5.2 Phase 3 리플레이 페이로드 (`backend/data/schema.py:180`)

```python
class PhaseThreeReplayTrial(BaseModel):
    trial_id: int
    frames: List[GameState]

class PhaseThreeStartPayload(BaseModel):
    session_id: str
    frame_duration_ms: int
    player_hat: str
    ai_hat: str
    trials: List[PhaseThreeReplayTrial]

class PhaseThreeSegment(BaseModel):
    segment_id: str
    start_frame: int
    end_frame: int       # validator: start ≤ end
    created_at_ms: int

class PhaseThreeTrialSelection(BaseModel):
    trial_id: int
    segments: List[PhaseThreeSegment]
```

### 5.3 저장 출력 (`backend/data/phase3_store.py`)

`zsceval/human_exp/data/logs/phase3/{session_id}.json`:

```json
{
  "session_id": "...",
  "saved_at_ms": ...,
  "phase": 3,
  "trials": [
    {
      "trial_id": 1,
      "frame_duration_ms": 100,
      "frames": [...],
      "misalignment_segments": [
        { "segment_id", "start_frame", "end_frame", "created_at_ms",
          "start_step", "end_step", "start_ms", "end_ms" }
      ]
    }
  ]
}
```

→ 본 개편에서 디렉터리 `phase3` → `phase2`, `phase: 3` → `phase: 2`.

---

## 6. Frontend 상태 전이 (`App.tsx`)

```
ConnectionStatus: 'disconnected' | 'connecting' | 'connected'
AppPhase: 'idle' | 'waiting' | 'instruction' | 'play' | 'rating' | 'break' | 'phase3' | 'complete'
```

WebSocket 메시지 핸들러가 phase를 갱신:

- `onTrialStart` → `instruction`
- `onPhaseChange` → payload.phase (play/rating/break)
- `onRatingAck` → `waiting` (다음 phase_change 대기)
- `onPhaseThreeStart` → `phase3`
- `onPhaseThreeComplete` → `complete`

**자동 시작 로직** (`App.tsx:225`): `connected && phase==='idle' && !sessionStartedRef.current` 일 때 `startSession()` 자동 호출. → 본 개편에서 **사용자가 ENTER 키를 눌러야 시작하도록 IdleScreen 추가**.

---

## 7. UI 라벨 현황

현재 모든 사용자 노출 라벨이 "Phase 2 / Phase 3" 로 되어 있어 혼동:

| 위치 | 현재 표기 | 개편 후 |
|---|---|---|
| App.tsx 헤더 (`Phase 2, Trial N`) | Phase 2 | **Phase 1** |
| InstructionScreen.tsx (`Phase 2, Trial {id}`) | Phase 2 | **Phase 1** |
| RatingScreen.tsx (`Phase 2, Trial {id}`) | Phase 2 | **Phase 1** |
| BreakScreen.tsx (`Phase 2, Trial {id} complete`) | Phase 2 | **Phase 1** |
| PhaseThreeScreen.tsx (`Phase 3, Replay Trial`) | Phase 3 | **Phase 2** |
| App.tsx complete (`Phase 3 complete`) | Phase 3 | **Phase 2** |
| PhaseThreeScreen `Finish Phase 3` 버튼 | Phase 3 | **Phase 2** |

---

## 8. 변경 전후 비교 표

| 항목 | Before | After |
|---|---|---|
| 시작 흐름 | WebSocket 연결되면 자동 `startSession()` | IdleScreen → ENTER 키 → `startSession()` |
| Trial 수 / 세션 | 4 (하드코딩) | 8 (seed=hash(session_id) 결정적) |
| 체크포인트 풀 | `tto_sp_seed4/5`, `ttt_adaptive_seed1/2` | `tto_mep1~4`, `too_mep1~4` (MEP 전용) |
| Trial 디자인 | 2×2 cell 각 1 trial | 2×2 cell 각 2 trial = 8 |
| 체크포인트 할당 | cell마다 고정 | cell 안에서 mep1~4 무작위 추첨 (seeded) |
| 사용자 라벨: 인스트럭션·플레이·평가 | "Phase 2" | "Phase 1" |
| 사용자 라벨: 리플레이 | "Phase 3" | "Phase 2" |
| 백엔드 메시지 타입 | `phase3_start`, `phase3_complete`, `submit_phase3` | `phase2_start`, `phase2_complete`, `submit_phase2` |
| Pydantic 모델 prefix | `PhaseThree*` | `PhaseTwo*` |
| 저장 경로 | `data/logs/phase3/{session}.json` | `data/logs/phase2/{session}.json` |
| 저장 JSON `phase` 필드 | `3` | `2` |
| PlayScreen 레이아웃 | HUD 위, 캔버스 좌측 정렬 | 캔버스 가운데 + 좌측 통계 카드 + 우측 레시피 카드 |
| Replay 화면 좌:우 비율 | 약 77 : 23 (`1fr 23rem`) | **60 : 40** (`3fr 2fr`) |
| 미스얼라인먼트 구간 조정 | 새 구간 드래그만 가능 | 새 구간 + 기존 구간 좌/우 핸들 드래그로 끝점 수정 |

---

## 9. 실행 방법 (현 상태 기준)

```bash
# 1. Backend (Terminal A)
conda activate neurocontroller
cd /home/juliecandoit98/neurocontroller
uvicorn zsceval.human_exp.backend.main:app --reload --port 8000
#  → http://localhost:8000/health  로 헬스체크
#  ※ 'cd zsceval/human_exp && uvicorn backend.main:app' 도 동작 (backend.* 절대 임포트 때문)

# 2. Frontend (Terminal B)
cd /home/juliecandoit98/neurocontroller/zsceval/human_exp/frontend
npm install        # 최초 1회
npm run dev        # → http://localhost:5173

# 3. Browser
# Chrome 등에서 http://localhost:5173 열면 자동으로 ws://localhost:8000/ws 연결
```

---

## 10. 본 개편에서 건드릴 파일 (요약)

### Backend
- `backend/game/ai_loader.py` — `_EXPLICIT_CHECKPOINT_SPECS` 8개 MEP spec으로 교체, `CheckpointId` / `_LAYOUT_BY_CHECKPOINT_ID` 동기 수정
- `backend/trial/condition.py` — `ModelCheckpoint` Literal 교체
- `backend/trial/session.py` — `generate_trials(session_id)` 신규 + 호출
- `backend/trial/test_trial.py` — `test_session_manager_phase_two_sequence` 8-trial 분포 검증으로 재작성, `test_ws_session_flow` 8 trial + phase2_* 로 갱신
- `backend/api/websocket.py` — import / 메시지 타입 문자열 / 함수명 phase3 → phase2
- `backend/data/schema.py` — `PhaseThree*` → `PhaseTwo*`
- `backend/data/phase3_store.py` → `phase2_store.py` (파일 + 함수 + 경로 리네임)
- `backend/data/test_phase3_store.py` → `test_phase2_store.py` (동기 수정)

### Frontend
- `frontend/src/screens/IdleScreen.tsx` — 신규 (Press ENTER to start)
- `frontend/src/App.tsx` — IdleScreen 분기, `phase3` → `phase2`, 라벨 변경
- `frontend/src/screens/InstructionScreen.tsx`, `RatingScreen.tsx`, `BreakScreen.tsx` — "Phase 2" → "Phase 1"
- `frontend/src/screens/PlayScreen.tsx` — 가운데 정렬 + 좌/우 위젯 카드
- `frontend/src/screens/PhaseThreeScreen.tsx` → `PhaseTwoScreen.tsx` (파일 + 클래스 + 60/40 + 좌우 핸들)
- `frontend/src/lib/websocket.ts` — `PhaseThree*` → `PhaseTwo*`, message type 문자열 갱신

### Docs
- `README.md` — 실행 절차 보강
- `INTERFACE_ANALYSIS.md` — 이 문서 (신규)
