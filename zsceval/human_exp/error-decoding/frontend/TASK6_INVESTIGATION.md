# Task 6 Pre-Investigation

Date: 2026-05-31

Scope: inspection only. No Task 6 implementation changes have been made.

## Documents Read

- `zsceval/human_exp/CLAUDE.md`
- `zsceval/human_exp/ARCHITECTURE.md`
- `tasks/task_4_trial_state_machine.md`
- `tasks/task_5_instruction_screen.md`
- `tasks/task_6_play_screen.md`

Relevant constraints:

- Task 6 is frontend-only. Do not modify backend code.
- WebSocket remains the only communication path.
- AI checkpoint/condition must remain invisible. Player and AI may differ by
  role hat color only; no visual should reveal tomato/onion checkpoint.
- No external sprite libraries.

## Current Frontend State

Current files:

- `frontend/src/App.tsx`: after Task 5, it auto-connects on page load, starts
  the default Task 4 test session, renders `InstructionScreen`, and then shows
  placeholder text for `play`, `rating`, and `break`.
- `frontend/src/lib/websocket.ts`: has typed Task 2 game messages and Task 4
  lifecycle messages. `GameStepMessage` currently includes
  `step_index`, `state`, `events`, `rewards`, and `server_timestamp_ms`.
- `frontend/src/screens/PlayScreen.tsx`: still the Task 2 raw-JSON/manual game
  screen. It owns its own local state and calls `client.setHandlers(...)`, which
  would overwrite App's lifecycle handlers if reused directly.
- `frontend/src/components/GameView.tsx`: stub only, returns `null`.
- `frontend/src/components/HatLegend.tsx`: implemented in Task 5 and can be
  reused for role/hat display if needed.

Baseline verification before implementation:

- `npm run typecheck` from `zsceval/human_exp/frontend`: passed.

## Backend Protocol / Data Shape

Task 4 play transition sequence:

1. Frontend sends `phase_ready` during instruction.
2. Backend sends `phase_change` with `phase="play"`.
3. Backend sends `game_start` with `initial_state`.
4. Backend starts sending `game_step` at 10 Hz.
5. Backend sends `game_end` when engine `done=True`.

Important wire details:

- `game_step` does **not** currently include a `done` boolean. Task 6 should use
  `game_end` for done-based auto-advance and `state.time_remaining` for time-up
  display.
- `GameState.time_remaining` is in engine ticks, not milliseconds. Current
  engine tick rate is 10 Hz, so frontend should convert ticks to milliseconds
  with `ticks * 100`.
- Current `OvercookedEngine.max_steps` is `600`, so actual engine play length is
  60 seconds, even though Task 4 `PLAY_DURATION_MS` is `75000`. Task 6 can
  faithfully display backend state, but the spec's "75s" expectation is not
  true for the current backend without a future backend change.
- `GameState.players[n].held_object` is the actual field name. The Task 6 spec
  mentions `held_item`, which is stale.
- `GameState.objects` contains dynamic objects only. At reset for `ttt` and
  `ooo`, it is empty. Static terrain cells such as counters, dispensers, pots,
  and delivery stations are not sent by the backend.

## Actual Layouts

Task 6 spec mentions `cramped_room` 5x4 as an example, but current Task 4/Task 3
trial flow uses checkpoint layouts:

- Trial 1 tomato checkpoint → `ttt`
- Trial 2 onion checkpoint → `ooo`

Inspection of ZSC-Eval layouts shows `ttt`, `ooo`, `tto`, and `too` all share
the same 13x5 geometry:

```text
XXXXXXXXXXXXX
O   DTXTD   O
XX    P    XX
S     P     S
XXXXXTXTXXXXX
```

Start positions from `OvercookedGridworld.get_standard_start_state()`:

- player index 0: `[8, 3]`, orientation `[0, -1]`
- player index 1: `[4, 3]`, orientation `[0, -1]`

For comparison:

- `corner_onion_tomato`: 7x6, old Task 2 debug layout.
- `cramped_room`: 5x4, only used in stale Task 6 examples.

## Rendering Implications

Because static terrain is absent from `GameState`, frontend-only Task 6 should
define a small local layout registry keyed by `state.layout_name`.

Recommended supported keys:

- `ttt`, `ooo`, `tto`, `too`: 13x5 shared geometry above.
- Optional fallback for `corner_onion_tomato` and `cramped_room` to preserve
  manual/debug compatibility.

Tile meanings from Overcooked layout files:

- `X`: counter
- space: floor
- `P`: pot
- `O`: onion dispenser
- `T`: tomato dispenser
- `D`: dish dispenser
- `S`: serving/delivery station

Dynamic soups are represented in `GameState.objects` as `name="soup"` with
`soup_state` and `ingredients`; their `position` should overlay the static pot
cell.

## App / Handler Risk

The current `PlayScreen` cannot be dropped into Task 5's `App` as-is because it
calls `client.setHandlers(...)` internally. That would remove App's
`onPhaseChange`, `onTrialStart`, and `onSessionComplete` handlers.

Recommended Task 6 approach:

1. Move game message handling into `App.tsx` so one place owns all WebSocket
   handlers.
2. Keep `PlayScreen` presentational with props from the task spec:
   `gameState`, `events`, `score`, `timeRemainingMs`, `playerHat`, `aiHat`,
   `onPhaseReady`.
3. Let `App` listen to `game_start`, `game_step`, and `game_end`.
4. Let `PlayScreen` own keyboard listeners only, using a passed
   `onPlayerAction(action)` prop or, if keeping the spec exactly, App can pass
   the client indirectly. The spec's prop list does not include this, but player
   movement requires sending `player_action`. Minimal deviation recommended:
   add `onPlayerAction(action: string)` to `PlayScreenProps`.

## Auto-Advance Considerations

Task 6 says auto-advance when `game_step.done=true` or time reaches zero. Since
`game_step.done` is not present, use:

- `game_end` → show "Time up!" overlay for 2 seconds → send `phase_ready`.
- `timeRemainingMs <= 0` as a fallback.

Need a one-shot guard so `phase_ready` is sent only once.

## Risks / Questions Before Implementation

- The spec's `PlayScreenProps` lacks an input callback, but AC2 requires WASD
  movement. Task 6 implementation needs either `onPlayerAction` in props or a
  retained `client` prop. Recommended: add `onPlayerAction` because it keeps
  `PlayScreen` mostly presentational and avoids handler ownership conflicts.
- The spec expects 75 seconds; backend state currently gives 60 seconds
  (`max_steps=600`). Frontend should not fake 75 seconds because backend will
  send `game_end` at 60 seconds.
- Static layout terrain must be duplicated in frontend unless backend schema is
  extended in a later task. This is acceptable for Task 6 because backend
  changes are out of scope.
- `GameView` must avoid labels or distinct shapes that reveal which role is AI.
  Both agents should use the same circle body shape and size; only hat color
  differs.

## Recommended Task 6 Plan

1. Implement local layout registry in `GameView.tsx`.
2. Implement `GameView` CSS grid renderer with static terrain + dynamic objects
   + two identical agent circles with role hat colors.
3. Rewrite `PlayScreen` as a presentational play UI with HUD, grid, last 5
   events, keyboard listener, and one-shot time-up overlay.
4. Update `App.tsx` to own all WS handlers and game state, render `PlayScreen`
   during `phase="play"`, and send `phase_ready` on game end/time-up.
5. Run `npm run typecheck && npm run build`, then backend `pytest -x -q`.

No stop condition is triggered, but the 75s-vs-60s mismatch should be called out
in Task 6 completion notes if implementation proceeds without backend changes.
