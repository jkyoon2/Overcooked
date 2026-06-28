# Phase 1 UI Revision — Plan & Analysis

> Date: 2026-06-10
> Scope: Phase 1 (PlayScreen + RatingScreen) UI fixes per Julie's request.
> Out of scope: Backend / data schema / Phase 2 screens.

---

## 1. Current state (what exists)

### `frontend/src/screens/PlayScreen.tsx`
- Layout: `display: grid; gridTemplateColumns: minmax(15rem, 1fr) minmax(0, auto) minmax(15rem, 1fr); maxWidth: 90rem` — fixed three-column grid with two side panels (each `maxWidth: 20rem`) and a center canvas.
- Canvas: `GameView` renders a `<canvas width={780} height={300}>` (13×5 tiles × 60px). No `max-width` / `height: auto`, so it does NOT shrink.
- Left panel ("Live status"):
  - Eyebrow: `Trial {trialId} of {totalTrials}` (purple uppercase) + title `Live status` (black).
  - Two stat rows (`Score` / `Time left`) using `justify-content: space-between` with a 1.7rem strong number.
  - Delivery list: each `<li>` shows `icon + label (flex:1) + count` per recipe.
  - Hat chips at bottom.
- Right panel ("Recipe values"):
  - Eyebrow: `Recipe values` + title `Cook for points`.
  - Recipe list: each row shows `icon + label (flex:1, fontWeight:600) + points badge`.
  - Footer note: "Tomato soup is closer to the pots — quicker but worth less. Onion soup needs more trips but pays four times as much."

### `frontend/src/screens/RatingScreen.tsx` + `components/RatingForm.tsx`
- Quality: `<fieldset>` with `gridTemplateColumns: repeat(7, 1fr)` of seven radio cards labeled 1–7.
- Intent: `<fieldset>` with `gridTemplateColumns: repeat(2, 1fr)` of four radio cards in a 2×2 layout.
- Submit button: standard rectangular blue button.

---

## 2. Identified issues (mapped to Julie's 8 items)

| # | Issue | Root cause |
|---|-------|------------|
| 1 | At narrow viewport (< ~1500px) side panels overlap the canvas | Grid forces `minmax(15rem, 1fr)` panels + 780px fixed canvas; canvas does not shrink. |
| 2 | Recipe rows in left panel show redundant "Tomato soup" text alongside emoji | Delivery list renders `row.label` |
| 3 | Two-line header `Trial X of Y / Live status` is verbose | `eyebrowStyle` + `cardTitleStyle` stacked |
| 4 | Two-line header `Recipe values / Cook for points` is verbose | Same pattern as #3 |
| 5 | The "Tomato soup is closer…" paragraph is unnecessary | `recipeNoteStyle` paragraph |
| 6 | Recipe rows in right panel show "Tomato·Onion·…" text alongside emoji | Recipe list renders `row.label` |
| 7 | Score number visually overflows the card | `statBigStyle` is 1.7rem on a horizontal `space-between` row inside a 20rem panel; on Korean OS / non-system fonts the digit width plus "Score" label exceeds available width when score ≥ 100. |
| 8 | Side panels feel too dominant; canvas too small | `maxWidth: 20rem` per panel + `auto` canvas column. |
| 9 (Rating) | Quality 1–7 are seven discrete buttons; intent 4 options in 2×2; both feel dated | Grid-of-radio pattern. User wants a slider + single-row segmented control with modern look. |

---

## 3. Design decisions

### 3.1 PlayScreen layout (items 1, 7, 8)

- **Side-panel width:** Reduce `maxWidth` from `20rem` → `13rem`. Reduce inner padding from `1.5rem` → `1rem`.
- **Grid columns:** `minmax(10rem, 13rem) minmax(0, 1fr) minmax(10rem, 13rem)`. Center column gets `1fr` so the canvas can grow into available space.
- **Canvas responsiveness:** Add `maxWidth: '100%'`, `height: 'auto'`, `width: '100%'` (with `aspectRatio` preserved via the intrinsic canvas attribute). `imageRendering: pixelated` keeps sprites crisp at non-integer scales.
- **Narrow-viewport stacking:** Inject a `<style>` block (scoped via a unique class on the content root) with `@media (max-width: 960px)` that switches the grid to a single column (`grid-template-columns: minmax(0, 1fr)`). Side panels then stack above/below the canvas.
- **Stat row fix (item 7):** Reflow Score / Time left from horizontal `space-between` to vertical stacking — small label on top, big number below. This (a) eliminates overflow at any panel width, (b) reads better with the smaller panel.

### 3.2 PlayScreen content trims (items 2, 3, 4, 5, 6)

- Replace `eyebrow + title` pair with a single purple `h2` per panel:
  - Left panel: `Live Status` (purple `#4338ca`, 1.05rem, bold).
  - Right panel: `Recipe Values` (same style).
- Remove `Trial X of Y` text from the panel — the App.tsx header already displays it (`Phase 1, Trial X of Y`).
- Drop `row.label` from both delivery list (left) and recipe list (right). Each row becomes `[icon] [count×]` or `[icon] [points-badge]`.
- Delete the `recipeNoteStyle` paragraph.

### 3.3 RatingForm redesign (item 9)

- **Quality (1–7) → slider:** Native `<input type="range" min={1} max={7} step={1}>` styled with a custom track + thumb (gradient track up to the value, neutral track after). Tick marks beneath with 1…7 labels and `Very poorly` / `Very well` end-anchors. The current value renders as a large numeric chip above the thumb.
- **Intent (4 options) → 1-row segmented control:** `gridTemplateColumns: repeat(4, 1fr)`. Each segment is a soft-bordered button with active state showing a gradient fill (`linear-gradient(135deg, #6366f1, #8b5cf6)`) and a subtle shadow. Inactive: white card with hover lift.
- **Submit button:** Pill-shaped, gradient fill matching active segment, with `box-shadow` for depth. Disabled state stays muted.
- **Quality state init:** Currently `null`; for slider UX we'll default the visual position to 4 while keeping `quality` state `null` until the user interacts (so we don't auto-submit a meaningless value). A small "Drag to rate" hint shows above the slider until the user changes it.
- The `frontend-design` skill will guide the actual visual choices (color tokens, shadow stack, motion).

---

## 4. Files to modify

| File | Change |
|------|--------|
| `frontend/src/screens/PlayScreen.tsx` | Layout grid, side-panel widths/padding, single purple title, drop labels in delivery/recipe lists, drop recipe note, vertical stat rows, responsive `<style>` block, pass-through props to GameView. |
| `frontend/src/components/GameView.tsx` | Add responsive `width: 100%, maxWidth: ${w}px, height: auto` to the `<canvas>` style. |
| `frontend/src/components/RatingForm.tsx` | Replace quality grid with custom slider; flatten intent to single-row segmented control; restyle submit button. |
| `frontend/src/screens/RatingScreen.tsx` | Minor: update title copy if needed; otherwise unchanged. |

No backend / schema changes.

---

## 5. Verification

1. `npm run typecheck` from `zsceval/human_exp/frontend/` — must pass.
2. `npm run build` — must pass.
3. Manual: resize browser from 1920px down to 800px; confirm:
   - Panels stay inside their cards, no overflow.
   - Canvas scales with center column.
   - At ≤ 960px panels stack above/below canvas.
4. Manual: trigger rating screen; confirm:
   - Slider value 1–7 selectable by drag, click, and arrow keys.
   - Intent segmented control selects on click, shows active state.
   - Submit enabled only after both quality and intent are touched.

---

## 6. Risks / non-decisions surfaced

- **Slider default vs. "untouched" state:** I chose to keep `quality === null` semantically (no auto-submit) but display the thumb at value 4 with a "Drag to rate" hint. Alternative is to initialize `quality = 4` and require an explicit confirm — riskier because participant might forget to drag.
- **Canvas at very small widths:** Below ~600px the 13-tile canvas becomes hard to read; we stack panels but don't otherwise resize. If this matters, we'd add a min-width with horizontal scroll on the canvas frame. Defer until Julie reports a use case.
- **Frontend-design skill scope:** I'll invoke it for the rating redesign only (tightest design surface). PlayScreen tweaks are mechanical and stay inline-styled to match the existing codebase pattern.

---

## 7. Survey UI follow-up — round 2 (2026-06-10)

### 7.1 Issues reported by Julie

| # | Issue | Root cause analysis |
|---|-------|---------------------|
| A | Header copy (`Phase 1, Trial X` eyebrow + `Trial X Rating` h1 + `Please answer both questions` helper) is unnecessary. | `RatingScreen.tsx` lines 108–125 hard-code three text elements. The App.tsx header already shows trial context. |
| B | Question legends (`How well…`, `Did the AI…`) visibly overflow the rounded fieldset card. | Native `<legend>` overlays the `<fieldset>` border by spec — when fieldset has `border-radius` + `padding`, the legend renders **above** the border (notched into it). At narrow viewports the text can also exceed the fieldset's inner box. |
| C | Tick numbers 1–7 don't line up with the slider thumb positions. | Tick row uses `grid-template-columns: repeat(7, 1fr)` over the full fieldset width, so tick centres sit at ≈7.14%, 21.43%, …, 92.86%. The native `<input type="range">` thumb centre, however, is inset by half the thumb diameter (14px) on each side — so its 1-position centre sits at ~14px from the track left, not at 7.14%. The two coordinate systems are mismatched. |
| D | Intent buttons (`Yes, clearly` etc.) are oversized. | `min-height: 3.4rem`, padding `0.75rem 0.6rem`, font 0.92rem. With four buttons in a row + 0.6rem gap, each cell is already wide enough — vertical height + padding dominate the visual mass. |

### 7.2 Design decisions

- **A — Strip RatingScreen header:** Delete the entire left-side text block (eyebrow + h1 + helper). Keep only the countdown timer chip, repositioned to the top-right corner of the form. Preserve `aria-label={`Trial ${trialId} rating`}` on the section so screen readers still announce context.
- **B — Replace `<fieldset>` + `<legend>` with semantic `<section>` + `<h3>`:** This eliminates the browser's special legend-on-border layout entirely. The heading sits cleanly inside the card padding, wraps normally, and respects `box-sizing: border-box`. Use `role="group"` + `aria-labelledby={headingId}` to preserve a11y semantics.
- **C — Align tick marks to slider thumb coordinate system:** Two changes:
  1. Wrap slider + tick row in a `position: relative` container with `padding: 0 14px` (thumb radius = 14px). This makes the inner coordinate space match where the thumb centre can travel.
  2. Render each tick as `position: absolute; left: ${(v-1)/6*100}%; transform: translateX(-50%)`. Now tick 1 sits at 14px from container-left, tick 7 sits at 14px from container-right — exactly where the slider thumb centres land.
- **D — Compact intent buttons:** Reduce to `min-height: 2.4rem`, padding `0.45rem 0.55rem`, font 0.85rem, gap 0.5rem. Keep the gradient-active state and lift hover. Active outer ring shrunk to `inset: -2px` to suit the smaller footprint.

### 7.3 frontend-design influence (without adding copy)

Per Julie's directive: no new text, only better visual technique on existing components. Applied:
- Slider track gets a soft inner shadow (`inset 0 1px 2px rgba(15,23,42,0.06)`) for depth.
- Thumb uses a layered shadow + ring (purple glow + 1px hairline) instead of the heavier stack.
- Intent buttons get a `transition` on transform/box-shadow only (no color thrash) and an active gradient that derives both stops from the same hue family (`hsl()` based) to avoid the prior multi-color jumble.
- Tick marker for the active value uses a 6px filled dot above the digit instead of a 2px bar — cleaner visual anchor next to the thumb.

### 7.4 Files to modify

| File | Change |
|------|--------|
| `frontend/src/screens/RatingScreen.tsx` | Strip three header text elements; reposition timer chip; trim outer padding. |
| `frontend/src/components/RatingForm.tsx` | Swap `<fieldset>`/`<legend>` → `<section>`/`<h3>`. New slider+tick container with `padding: 0 14px` and absolute-positioned tick marks. Tighten intent button sizing. Refresh shadow / motion tokens. |

No prop / state contract changes. `quality`, `intentAlignment`, `disabled`, `onSubmit` semantics unchanged.

### 7.5 Verification

- `npm run typecheck && npm run build` from `zsceval/human_exp/frontend/`.
- Manual: open rating screen at 1440 / 1024 / 768px widths; confirm:
  - No floating "Trial X Rating" / "Please answer both questions" text.
  - Question heading sits cleanly inside its card, no border notch / overflow.
  - Slider thumb at value 1 vertically lines up with the "1" tick; at value 7 lines up with "7".
  - Intent buttons feel ~30% shorter than before; four still fit one-row.
