# Task 7 Pre-Investigation

Date: 2026-05-31

Scope: inspection before Task 7 implementation. No Task 7 implementation
changes have been made at the time this file is written.

## Documents Read

- `zsceval/human_exp/CLAUDE.md`
- `zsceval/human_exp/ARCHITECTURE.md`
- `tasks/task_4_trial_state_machine.md`
- `tasks/task_7_rating_screen.md`

Relevant constraints:

- Task 7 is frontend-only.
- Manipulation check logic belongs to backend `TrialManager`; frontend only
  collects and sends `submit_rating`.
- The `rating_ack.excluded` value is backend-only and must not be shown to the
  participant.
- WebSocket is the only communication path.

## Current Frontend State

Task 6 has already changed the frontend from the Task 5 shell:

- `App.tsx` auto-connects, starts the default session, renders instruction,
  renders `PlayScreen` during `phase="play"`, and owns all WebSocket handlers.
- `PlayScreen.tsx` is presentational and sends player input via
  `onPlayerAction`.
- `GameView.tsx` renders the game grid.
- `RatingScreen.tsx` is still a stub that returns `null`.
- `RatingForm.tsx` is still a stub that returns `null`.
- `websocket.ts` already defines `RatingAckMessage` and exposes
  `onRatingAck`, but `WebSocketClient` has no typed `submitRating(...)` helper
  yet. `sendJson(...)` could work, but a typed helper is cleaner and consistent
  with `startSession`, `sendPhaseReady`, and `sendPlayerAction`.

Baseline verification before implementation:

- Frontend `npm run typecheck`: passed.

## Backend Rating Protocol

Backend receives:

```json
{
  "type": "submit_rating",
  "payload": {
    "quality": 4,
    "intent_alignment": "yes_somewhat"
  }
}
```

Backend responds:

```json
{
  "type": "rating_ack",
  "payload": {
    "trial_id": 1,
    "excluded": false,
    "exclusion_reason": null
  }
}
```

Immediately after `rating_ack`, backend sends:

```json
{
  "type": "phase_change",
  "payload": {
    "phase": "break",
    "duration_ms": 5000
  }
}
```

After that `phase_change("break")`, backend waits for another client
`phase_ready` before it sends the next `trial_start` or `session_complete`.
There is no implemented `BreakScreen` yet, so Task 7 wiring should include a
minimal App-level break timer that sends `phase_ready` after the backend-provided
break duration. Otherwise the session stalls after rating.

Current backend `TrialManager.submit_rating()` uses the required 4-point scale:

- `yes_clearly = 1`
- `yes_somewhat = 2`
- `no_somewhat = 3`
- `no_clearly = 4`

Expected score is `1` for aligned trials and `4` for misaligned trials.
Distance greater than `1` marks `excluded=True`.

## App Integration Point

Current `App.tsx` renders:

```tsx
{phase === 'rating' && <p>Rating phase ready.</p>}
```

Task 7 should replace this placeholder with `RatingScreen`.

App should also store:

- current `trialId` from `trial_start`
- rating phase `durationMs` from `phase_change`
- break phase `durationMs` from `phase_change`
- a submitted/waiting flag so duplicate submits are ignored

`rating_ack` handling should:

- clear submitted/waiting state
- not render or otherwise surface `excluded`
- wait for the backend's following `phase_change("break")`

## Form Behavior

Recommended defaults for timer expiry when incomplete:

- `quality = 4`
- `intent_alignment = "yes_somewhat"`

These match the Task 7 spec and provide neutral-ish values while guaranteeing
the backend never hangs waiting for a rating.

The Submit button should be disabled until both questions are answered. Timer
expiry is the only path that submits defaults.

## Risks / Questions Before Implementation

- `rating_ack` can be followed immediately by `phase_change("break")`; App must
  not show the exclusion result or an intermediate acknowledgement screen.
- React StrictMode can double-run effects in development. `RatingScreen` should
  use a one-shot guard for timer auto-submit so it sends at most one rating.
- Backend accepts any integer quality today; frontend should enforce the 1-7
  range by UI.
- Task 8 data logging is out of scope; no frontend logging should be added.

No stop condition is triggered.

## Recommended Task 7 Plan

1. Add rating payload/types and a typed `submitRating(...)` helper to
   `frontend/src/lib/websocket.ts`.
2. Implement pure `RatingForm` with Q1 1-7 radio buttons, Q2 4 radio buttons,
   and disabled submit until both are answered.
3. Implement `RatingScreen` with countdown and one-shot timer auto-submit using
   defaults for incomplete ratings.
4. Wire `App.tsx` rating phase to `RatingScreen`; send `submit_rating` and hide
   any `rating_ack.excluded` result from the participant.
5. Add minimal App-level break auto-ready so Task 4's client-driven BREAK phase
   can progress to the next trial.
6. Run frontend typecheck/build and backend pytest.
