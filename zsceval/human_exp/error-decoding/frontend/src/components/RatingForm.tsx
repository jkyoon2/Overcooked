import { useEffect, useId, useMemo, useState, type CSSProperties } from 'react'
import type { IntentAlignment, RatingPayload } from '../lib/websocket'

type RatingFormProps = {
  quality: number | null
  intentAlignment: IntentAlignment | null
  disabled?: boolean
  onQualityChange: (quality: number) => void
  onIntentAlignmentChange: (intentAlignment: IntentAlignment) => void
  onSubmit: (rating: RatingPayload) => void
}

const INTENT_OPTIONS: Array<{ value: IntentAlignment; label: string; hue: number }> = [
  { value: 'yes_clearly',  label: 'Yes, clearly',  hue: 152 },
  { value: 'yes_somewhat', label: 'Yes, somewhat', hue: 92 },
  { value: 'no_somewhat',  label: 'No, somewhat',  hue: 32 },
  { value: 'no_clearly',   label: 'No, clearly',   hue: 0 },
]

const QUALITY_MIN = 1
const QUALITY_MAX = 7
const QUALITY_DEFAULT_VISUAL = 4
const THUMB_DIAMETER = 28
const THUMB_RADIUS = THUMB_DIAMETER / 2

const SCOPED_CSS = `
.rating-slider {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 14px;
  background: transparent;
  outline: none;
  margin: 0;
  cursor: pointer;
  display: block;
}
.rating-slider::-webkit-slider-runnable-track {
  height: 14px;
  border-radius: 999px;
  background: var(--rating-track);
  border: 1px solid #e2e8f0;
  box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06);
}
.rating-slider::-moz-range-track {
  height: 14px;
  border-radius: 999px;
  background: var(--rating-track);
  border: 1px solid #e2e8f0;
  box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06);
}
.rating-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 28px;
  height: 28px;
  margin-top: -8px;
  border-radius: 999px;
  background: #ffffff;
  border: 2px solid #7c3aed;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35), inset 0 0 0 5px #7c3aed;
  cursor: grab;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.rating-slider::-moz-range-thumb {
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: #ffffff;
  border: 2px solid #7c3aed;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35), inset 0 0 0 5px #7c3aed;
  cursor: grab;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.rating-slider:focus-visible::-webkit-slider-thumb {
  box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.22), 0 4px 14px rgba(124, 58, 237, 0.35), inset 0 0 0 5px #7c3aed;
}
.rating-slider:active::-webkit-slider-thumb { transform: scale(1.06); cursor: grabbing; }
.rating-slider.untouched::-webkit-slider-thumb {
  border-color: #cbd5e1;
  box-shadow: 0 3px 10px rgba(148, 163, 184, 0.3), inset 0 0 0 5px #cbd5e1;
}
.rating-slider.untouched::-moz-range-thumb {
  border-color: #cbd5e1;
  box-shadow: 0 3px 10px rgba(148, 163, 184, 0.3), inset 0 0 0 5px #cbd5e1;
}

.intent-seg {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.5rem;
}
.intent-seg label {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 2.4rem;
  padding: 0.45rem 0.55rem;
  border-radius: 12px;
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  color: #334155;
  font-weight: 600;
  font-size: 0.85rem;
  text-align: center;
  line-height: 1.15;
  cursor: pointer;
  user-select: none;
  transition: transform 140ms ease, box-shadow 140ms ease;
}
.intent-seg label:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
}
.intent-seg input { position: absolute; opacity: 0; pointer-events: none; }
.intent-seg label.active {
  color: #ffffff;
  border-color: transparent;
  background: linear-gradient(135deg, hsl(var(--seg-hue), 70%, 52%), hsl(var(--seg-hue), 70%, 42%));
  box-shadow: 0 8px 18px hsla(var(--seg-hue), 70%, 45%, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.22);
}
.intent-seg label.active::after {
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: 14px;
  border: 1.5px solid hsla(var(--seg-hue), 70%, 60%, 0.35);
  pointer-events: none;
}

.rating-submit {
  align-self: stretch;
  border: 0;
  border-radius: 999px;
  padding: 0.8rem 1.4rem;
  font: inherit;
  font-weight: 800;
  font-size: 0.95rem;
  letter-spacing: 0.02em;
  cursor: pointer;
  background: linear-gradient(135deg, #6366f1, #8b5cf6 55%, #ec4899);
  color: #ffffff;
  box-shadow: 0 14px 28px rgba(99, 102, 241, 0.3);
  transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
}
.rating-submit:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.03); }
.rating-submit:active:not(:disabled) { transform: translateY(0); }
.rating-submit:disabled {
  background: #e2e8f0;
  color: #94a3b8;
  cursor: not-allowed;
  box-shadow: none;
}

.tick-row {
  position: relative;
  height: 1.4rem;
  margin-top: 0.5rem;
}
.tick-row .tick {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
  color: #94a3b8;
  font-weight: 700;
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
  line-height: 1;
}
.tick-row .tick::before {
  content: '';
  width: 2px;
  height: 6px;
  background: #cbd5e1;
  border-radius: 2px;
}
.tick-row .tick.on { color: #6d28d9; }
.tick-row .tick.on::before {
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #7c3aed;
  box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15);
}
`

function trackBackground(value: number): string {
  const pct = ((value - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN)) * 100
  return `linear-gradient(90deg, #8b5cf6 0%, #ec4899 ${pct}%, #e2e8f0 ${pct}%, #e2e8f0 100%)`
}

export default function RatingForm({
  quality,
  intentAlignment,
  disabled = false,
  onQualityChange,
  onIntentAlignmentChange,
  onSubmit,
}: RatingFormProps) {
  const [touched, setTouched] = useState(quality !== null)
  useEffect(() => {
    if (quality !== null) setTouched(true)
  }, [quality])

  const qualityHeadingId = useId()
  const intentHeadingId = useId()

  const visualValue = quality ?? QUALITY_DEFAULT_VISUAL
  const trackStyle = useMemo<CSSProperties>(
    () => ({ ['--rating-track' as string]: trackBackground(visualValue) }) as CSSProperties,
    [visualValue],
  )

  const complete = quality !== null && intentAlignment !== null

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        if (!complete || disabled) return
        onSubmit({ quality: quality!, intent_alignment: intentAlignment! })
      }}
      style={formStyle}
    >
      <style>{SCOPED_CSS}</style>

      <section
        role="group"
        aria-labelledby={qualityHeadingId}
        aria-disabled={disabled}
        style={cardStyle}
      >
        <h3 id={qualityHeadingId} style={headingStyle}>How well did the collaboration go?</h3>

        <div style={sliderHeaderStyle}>
          <span style={sliderHintStyle}>
            {touched ? 'Your rating' : 'Drag to rate'}
          </span>
          <div style={sliderValueChipStyle(touched)}>
            {touched ? visualValue : '–'}
            <span style={sliderValueSuffixStyle(touched)}>/ {QUALITY_MAX}</span>
          </div>
        </div>

        <div style={sliderTrackInsetStyle}>
          <input
            type="range"
            min={QUALITY_MIN}
            max={QUALITY_MAX}
            step={1}
            value={visualValue}
            onChange={(e) => {
              setTouched(true)
              onQualityChange(Number(e.target.value))
            }}
            onPointerDown={() => setTouched(true)}
            className={`rating-slider${touched ? '' : ' untouched'}`}
            style={trackStyle}
            aria-label="Collaboration quality, 1 to 7"
            disabled={disabled}
          />

          <div className="tick-row" aria-hidden="true">
            {[1, 2, 3, 4, 5, 6, 7].map((v) => (
              <div
                key={v}
                className={`tick${touched && v === visualValue ? ' on' : ''}`}
                style={{ left: `${((v - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN)) * 100}%` }}
              >
                {v}
              </div>
            ))}
          </div>
        </div>

        <div style={endLabelsStyle}>
          <span>Very poorly</span>
          <span>Very well</span>
        </div>
      </section>

      <section
        role="group"
        aria-labelledby={intentHeadingId}
        aria-disabled={disabled}
        style={cardStyle}
      >
        <h3 id={intentHeadingId} style={headingStyle}>Did the AI follow the same goal as you?</h3>
        <div className="intent-seg">
          {INTENT_OPTIONS.map((option) => {
            const active = intentAlignment === option.value
            return (
              <label
                key={option.value}
                className={active ? 'active' : ''}
                style={{ ['--seg-hue' as string]: String(option.hue) } as CSSProperties}
              >
                <input
                  type="radio"
                  name="intentAlignment"
                  value={option.value}
                  checked={active}
                  onChange={() => onIntentAlignmentChange(option.value)}
                  disabled={disabled}
                />
                <span>{option.label}</span>
              </label>
            )
          })}
        </div>
      </section>

      <button type="submit" disabled={!complete || disabled} className="rating-submit">
        Submit rating
      </button>
    </form>
  )
}

const formStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '1.5rem',
  width: 'min(46rem, 100%)',
}

const cardStyle: CSSProperties = {
  boxSizing: 'border-box',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.85rem',
  width: '100%',
  padding: '1.3rem 1.4rem 1.5rem',
  border: '1px solid #e2e8f0',
  borderRadius: '20px',
  background: 'linear-gradient(180deg, #ffffff 0%, #fafafe 100%)',
  boxShadow: '0 14px 36px rgba(15, 23, 42, 0.05)',
}

const headingStyle: CSSProperties = {
  margin: 0,
  color: '#0f172a',
  fontSize: '1.05rem',
  fontWeight: 700,
  lineHeight: 1.3,
  overflowWrap: 'break-word',
}

const sliderHeaderStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '0.5rem',
}

const sliderHintStyle: CSSProperties = {
  color: '#64748b',
  fontSize: '0.8rem',
  fontWeight: 600,
}

const sliderTrackInsetStyle: CSSProperties = {
  position: 'relative',
  padding: `0 ${THUMB_RADIUS}px`,
}

const sliderValueChipStyle = (touched: boolean): CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'baseline',
  gap: '0.15rem',
  minWidth: '3.2rem',
  justifyContent: 'center',
  padding: '0.3rem 0.7rem',
  borderRadius: '999px',
  background: touched ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : '#e2e8f0',
  color: touched ? '#ffffff' : '#64748b',
  fontVariantNumeric: 'tabular-nums',
  fontWeight: 800,
  fontSize: '0.95rem',
  letterSpacing: '0.01em',
  boxShadow: touched ? '0 6px 14px rgba(99, 102, 241, 0.28)' : 'none',
  transition: 'background 160ms ease, color 160ms ease, box-shadow 160ms ease',
})

const sliderValueSuffixStyle = (touched: boolean): CSSProperties => ({
  marginLeft: '0.25rem',
  color: touched ? 'rgba(255, 255, 255, 0.8)' : '#94a3b8',
  fontSize: '0.72rem',
  fontWeight: 700,
})

const endLabelsStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  marginTop: '0.1rem',
  padding: `0 ${THUMB_RADIUS}px`,
  color: '#94a3b8',
  fontSize: '0.78rem',
}
