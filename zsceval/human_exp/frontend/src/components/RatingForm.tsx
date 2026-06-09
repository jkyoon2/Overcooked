import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import type { IntentAlignment, RatingPayload } from '../lib/websocket'

type RatingFormProps = {
  quality: number | null
  intentAlignment: IntentAlignment | null
  disabled?: boolean
  onQualityChange: (quality: number) => void
  onIntentAlignmentChange: (intentAlignment: IntentAlignment) => void
  onSubmit: (rating: RatingPayload) => void
}

const INTENT_OPTIONS: Array<{ value: IntentAlignment; label: string; tone: string }> = [
  { value: 'yes_clearly',  label: 'Yes, clearly',  tone: '#16a34a' },
  { value: 'yes_somewhat', label: 'Yes, somewhat', tone: '#65a30d' },
  { value: 'no_somewhat',  label: 'No, somewhat',  tone: '#ea580c' },
  { value: 'no_clearly',   label: 'No, clearly',   tone: '#dc2626' },
]

const QUALITY_MIN = 1
const QUALITY_MAX = 7
const QUALITY_DEFAULT_VISUAL = 4

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
}
.rating-slider::-webkit-slider-runnable-track {
  height: 14px;
  border-radius: 999px;
  background: var(--rating-track);
  border: 1px solid #e2e8f0;
}
.rating-slider::-moz-range-track {
  height: 14px;
  border-radius: 999px;
  background: var(--rating-track);
  border: 1px solid #e2e8f0;
}
.rating-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 28px;
  height: 28px;
  margin-top: -8px;
  border-radius: 999px;
  background: #ffffff;
  border: 3px solid #7c3aed;
  box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35), inset 0 0 0 4px #7c3aed;
  cursor: grab;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.rating-slider::-moz-range-thumb {
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: #ffffff;
  border: 3px solid #7c3aed;
  box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35), inset 0 0 0 4px #7c3aed;
  cursor: grab;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.rating-slider:focus-visible::-webkit-slider-thumb {
  box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.25), 0 6px 18px rgba(124, 58, 237, 0.35), inset 0 0 0 4px #7c3aed;
}
.rating-slider:active::-webkit-slider-thumb { transform: scale(1.06); cursor: grabbing; }
.rating-slider.untouched::-webkit-slider-thumb {
  border-color: #94a3b8;
  box-shadow: 0 4px 12px rgba(148, 163, 184, 0.35), inset 0 0 0 4px #cbd5e1;
}
.rating-slider.untouched::-moz-range-thumb {
  border-color: #94a3b8;
  box-shadow: 0 4px 12px rgba(148, 163, 184, 0.35), inset 0 0 0 4px #cbd5e1;
}

.intent-seg {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.6rem;
}
.intent-seg label {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  min-height: 3.4rem;
  padding: 0.75rem 0.6rem;
  border-radius: 14px;
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  color: #334155;
  font-weight: 600;
  font-size: 0.92rem;
  cursor: pointer;
  user-select: none;
  transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background 140ms ease, color 140ms ease;
}
.intent-seg label:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
  border-color: #cbd5e1;
}
.intent-seg input { position: absolute; opacity: 0; pointer-events: none; }
.intent-seg label.active {
  color: #ffffff;
  border-color: transparent;
  background: linear-gradient(135deg, var(--seg-tone-a), var(--seg-tone-b));
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14), inset 0 1px 0 rgba(255,255,255,0.25);
}
.intent-seg label.active::after {
  content: '';
  position: absolute;
  inset: -3px;
  border-radius: 16px;
  border: 2px solid rgba(255,255,255,0.35);
  pointer-events: none;
}

.rating-submit {
  align-self: stretch;
  border: 0;
  border-radius: 999px;
  padding: 0.95rem 1.4rem;
  font: inherit;
  font-weight: 800;
  font-size: 1rem;
  letter-spacing: 0.02em;
  cursor: pointer;
  background: linear-gradient(135deg, #6366f1, #8b5cf6 60%, #ec4899);
  color: #ffffff;
  box-shadow: 0 18px 36px rgba(99, 102, 241, 0.35);
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
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  margin-top: 0.45rem;
}
.tick-row .tick {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  color: #94a3b8;
  font-weight: 700;
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
}
.tick-row .tick::before {
  content: '';
  width: 2px;
  height: 8px;
  background: #cbd5e1;
  border-radius: 2px;
}
.tick-row .tick.on { color: #6d28d9; }
.tick-row .tick.on::before { background: #7c3aed; }
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

      <fieldset style={fieldsetStyle} disabled={disabled}>
        <legend style={legendStyle}>How well did the collaboration go?</legend>

        <div style={sliderHeaderStyle}>
          <div>
            <span style={sliderHintStyle}>
              {touched ? 'Your rating' : 'Drag to rate'}
            </span>
          </div>
          <div style={sliderValueChipStyle(touched)}>
            {touched ? visualValue : '–'}
            <span style={sliderValueSuffixStyle}>/ {QUALITY_MAX}</span>
          </div>
        </div>

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
        />

        <div className="tick-row" aria-hidden="true">
          {[1, 2, 3, 4, 5, 6, 7].map((v) => (
            <div key={v} className={`tick${touched && v === visualValue ? ' on' : ''}`}>
              {v}
            </div>
          ))}
        </div>

        <div style={endLabelsStyle}>
          <span>Very poorly</span>
          <span>Very well</span>
        </div>
      </fieldset>

      <fieldset style={fieldsetStyle} disabled={disabled}>
        <legend style={legendStyle}>Did the AI follow the same goal as you?</legend>
        <div className="intent-seg">
          {INTENT_OPTIONS.map((option, idx) => {
            const active = intentAlignment === option.value
            const toneA = option.tone
            const toneB = INTENT_OPTIONS[(idx + 1) % INTENT_OPTIONS.length].tone
            return (
              <label
                key={option.value}
                className={active ? 'active' : ''}
                style={{
                  ['--seg-tone-a' as string]: toneA,
                  ['--seg-tone-b' as string]: toneB,
                } as CSSProperties}
              >
                <input
                  type="radio"
                  name="intentAlignment"
                  value={option.value}
                  checked={active}
                  onChange={() => onIntentAlignmentChange(option.value)}
                />
                <span>{option.label}</span>
              </label>
            )
          })}
        </div>
      </fieldset>

      <button type="submit" disabled={!complete || disabled} className="rating-submit">
        Submit rating
      </button>
    </form>
  )
}

const formStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '2rem',
  width: 'min(46rem, 100%)',
}

const fieldsetStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.9rem',
  margin: 0,
  padding: '1.4rem 1.5rem 1.6rem',
  border: '1px solid #e2e8f0',
  borderRadius: '20px',
  background: 'linear-gradient(180deg, #ffffff 0%, #fafafe 100%)',
  boxShadow: '0 14px 40px rgba(15, 23, 42, 0.06)',
}

const legendStyle: CSSProperties = {
  marginBottom: '0.4rem',
  padding: '0 0.4rem',
  color: '#0f172a',
  fontSize: '1.1rem',
  fontWeight: 700,
}

const sliderHeaderStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '0.5rem',
}

const sliderHintStyle: CSSProperties = {
  color: '#64748b',
  fontSize: '0.85rem',
  fontWeight: 600,
}

const sliderValueSuffixStyle: CSSProperties = {
  marginLeft: '0.3rem',
  color: 'rgba(255,255,255,0.85)',
  fontSize: '0.8rem',
  fontWeight: 600,
}

const sliderValueChipStyle = (touched: boolean): CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'baseline',
  gap: '0.1rem',
  minWidth: '3.6rem',
  justifyContent: 'center',
  padding: '0.35rem 0.75rem',
  borderRadius: '999px',
  background: touched ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : '#e2e8f0',
  color: touched ? '#ffffff' : '#64748b',
  fontVariantNumeric: 'tabular-nums',
  fontWeight: 800,
  fontSize: '1.05rem',
  letterSpacing: '0.01em',
  boxShadow: touched ? '0 8px 20px rgba(99, 102, 241, 0.3)' : 'none',
  transition: 'background 160ms ease, color 160ms ease, box-shadow 160ms ease',
})

const endLabelsStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  marginTop: '0.4rem',
  color: '#64748b',
  fontSize: '0.85rem',
}
