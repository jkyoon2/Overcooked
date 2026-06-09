# Task 5 Pre-Investigation

Date: 2026-05-31

Scope: inspection only. No Task 5 implementation changes have been made.

## Documents Read

- `zsceval/human_exp/CLAUDE.md`
- `zsceval/human_exp/ARCHITECTURE.md`
- `tasks/task_4_trial_state_machine.md`
- `tasks/task_5_instruction_screen.md`

Relevant constraints:

- Frontend is React 18 + TypeScript strict + Vite.
- WebSocket remains the only communication path for trial lifecycle messages.
- AI condition must stay backend-only; instruction UI may show role hat colors
  but must not expose checkpoint or condition.
- Task 5 is frontend-only. Backend code should not be modified.

## Current Frontend State

Existing files already present:

- `frontend/src/screens/InstructionScreen.tsx`: stub only, returns `null`.
- `frontend/src/components/HatLegend.tsx`: stub only, returns `null`.
- `frontend/src/App.tsx`: Task 2 flow only. It connects to the WebSocket and
  immediately renders `PlayScreen` with `layout="corner_onion_tomato"`.
- `frontend/src/lib/websocket.ts`: Task 2 message types only
  (`hello_ack`, `game_start`, `game_step`, `game_end`). It does not yet expose
  Task 4 lifecycle types or handlers.
- `frontend/src/screens/PlayScreen.tsx`: raw JSON play screen with its own
  `client.setHandlers(...)` call for game messages.
- `frontend/src/screens/RatingScreen.tsx`, `BreakScreen.tsx`,
  `components/GameView.tsx`, `components/RatingForm.tsx`: stubs only.
- `frontend/src/state/trialState.ts`: placeholder type only; no state store.

Baseline verification before implementation:

- `npm run typecheck` from `zsceval/human_exp/frontend`: passed.

## Backend Protocol Available From Task 4

Task 4 backend sends:

```json
{
  "type": "trial_start",
  "payload": {
    "trial_id": 1,
    "phase": "instruction",
    "duration_ms": 10000,
    "player_hat": "blue",
    "ai_hat": "red"
  }
}
```

Task 4 frontend must send:

```json
{ "type": "phase_ready", "payload": {} }
```

The backend also supports `start_session`, but the current frontend has no
typed client method for it and no UI path that sends it.

## Implementation Implications

1. `InstructionScreen` can be implemented directly from the Task 5 prop
   contract. It needs a countdown interval and a one-shot guard so React
   StrictMode does not cause duplicate `onReady()` sends.
2. `HatLegend` is pure display and should not contain condition logic.
3. `App.tsx` needs a trial phase state so it can render instruction instead of
   always rendering `PlayScreen`.
4. Although Task 5's file list only mentions `App.tsx`, receiving
   `trial_start` through the existing wrapper requires a minimal
   `frontend/src/lib/websocket.ts` update:
   - Add `TrialStartMessage` and `PhaseChangeMessage` types.
   - Add corresponding handler fields.
   - Add `startSession(sessionId)` and `sendPhaseReady()` helpers, or use
     existing `sendJson()` from `App.tsx`.
5. The current app has no way to initiate a Task 4 session. For a manual smoke
   test, the smallest frontend change is a connected idle state with a
   "Start Session" action that sends `start_session` for a test session id.
   Auto-start-on-connect is also possible, but a visible idle state is easier
   to debug and keeps connection separate from session lifecycle.

## Risks / Questions Before Implementation

- Scope mismatch: `websocket.ts` is not listed under Task 5 files, but without
  updating it, `App.tsx` cannot receive `trial_start` via the existing typed
  client wrapper. Recommended resolution: allow a minimal `websocket.ts` change
  because it is necessary protocol wiring, not backend or game-rendering scope.
- `PlayScreen` currently overwrites all WebSocket handlers via
  `client.setHandlers(...)`. When App starts owning trial lifecycle handlers,
  Task 6 may need handler composition or centralized routing so PlayScreen does
  not accidentally remove trial handlers.
- Task 5 should not implement Play rendering, Rating, Break countdowns, data
  logging, EEG, or backend protocol changes.

## Recommended Task 5 Plan

1. Extend `frontend/src/lib/websocket.ts` with Task 4 lifecycle message types,
   handlers, `startSession()`, and `sendPhaseReady()`.
2. Implement `HatLegend` as a pure role/hat display component.
3. Implement `InstructionScreen` with countdown, one-shot auto-advance, and no
   interactive controls during the instruction phase.
4. Update `App.tsx` to render a connected idle state, start a test session, and
   render `InstructionScreen` on `trial_start`.
5. Run `npm run typecheck && npm run build`.

No stop condition is triggered by the current codebase.
