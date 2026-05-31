import type { CSSProperties } from 'react'
import type { IntentAlignment, RatingPayload } from '../lib/websocket'

type RatingFormProps = {
  quality: number | null
  intentAlignment: IntentAlignment | null
  disabled?: boolean
  onQualityChange: (quality: number) => void
  onIntentAlignmentChange: (intentAlignment: IntentAlignment) => void
  onSubmit: (rating: RatingPayload) => void
}

const INTENT_OPTIONS: Array<{ value: IntentAlignment; label: string }> = [
  { value: 'yes_clearly', label: 'Yes, clearly' },
  { value: 'yes_somewhat', label: 'Yes, somewhat' },
  { value: 'no_somewhat', label: 'No, somewhat' },
  { value: 'no_clearly', label: 'No, clearly' },
]

const QUALITY_VALUES = [1, 2, 3, 4, 5, 6, 7]

export default function RatingForm({
  quality,
  intentAlignment,
  disabled = false,
  onQualityChange,
  onIntentAlignmentChange,
  onSubmit,
}: RatingFormProps) {
  const complete = quality !== null && intentAlignment !== null

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        if (!complete || disabled) return
        onSubmit({ quality, intent_alignment: intentAlignment })
      }}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '2rem',
        width: 'min(42rem, 100%)',
      }}
    >
      <fieldset style={fieldsetStyle} disabled={disabled}>
        <legend style={legendStyle}>How well did the collaboration go?</legend>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(7, minmax(0, 1fr))',
            gap: '0.6rem',
            alignItems: 'stretch',
          }}
        >
          {QUALITY_VALUES.map((value) => (
            <label
              key={value}
              style={{
                ...optionStyle,
                borderColor: quality === value ? '#2563eb' : '#cbd5e1',
                background: quality === value ? '#eff6ff' : '#ffffff',
              }}
            >
              <input
                type="radio"
                name="quality"
                value={value}
                checked={quality === value}
                onChange={() => onQualityChange(value)}
                style={{ margin: 0 }}
              />
              <strong>{value}</strong>
            </label>
          ))}
        </div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            color: '#64748b',
            fontSize: '0.9rem',
          }}
        >
          <span>Very poorly</span>
          <span>Very well</span>
        </div>
      </fieldset>

      <fieldset style={fieldsetStyle} disabled={disabled}>
        <legend style={legendStyle}>Did the AI follow the same goal as you?</legend>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0.75rem' }}>
          {INTENT_OPTIONS.map((option) => (
            <label
              key={option.value}
              style={{
                ...optionStyle,
                justifyContent: 'flex-start',
                minHeight: '3rem',
                borderColor: intentAlignment === option.value ? '#2563eb' : '#cbd5e1',
                background: intentAlignment === option.value ? '#eff6ff' : '#ffffff',
              }}
            >
              <input
                type="radio"
                name="intentAlignment"
                value={option.value}
                checked={intentAlignment === option.value}
                onChange={() => onIntentAlignmentChange(option.value)}
                style={{ margin: 0 }}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <button
        type="submit"
        disabled={!complete || disabled}
        style={{
          alignSelf: 'flex-start',
          border: '1px solid transparent',
          borderRadius: '6px',
          padding: '0.75rem 1.15rem',
          background: complete && !disabled ? '#2563eb' : '#cbd5e1',
          color: complete && !disabled ? '#ffffff' : '#64748b',
          font: 'inherit',
          fontWeight: 700,
          cursor: complete && !disabled ? 'pointer' : 'not-allowed',
        }}
      >
        Submit
      </button>
    </form>
  )
}

const fieldsetStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem',
  margin: 0,
  padding: 0,
  border: 0,
} satisfies CSSProperties

const legendStyle = {
  marginBottom: '0.35rem',
  color: '#111827',
  fontSize: '1.35rem',
  fontWeight: 700,
} satisfies CSSProperties

const optionStyle = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '0.55rem',
  minHeight: '3.4rem',
  border: '1px solid #cbd5e1',
  borderRadius: '8px',
  padding: '0.55rem 0.65rem',
  color: '#111827',
  cursor: 'pointer',
  userSelect: 'none',
} satisfies CSSProperties
