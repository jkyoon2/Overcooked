import { useEffect, useState } from 'react'

type BreakScreenProps = {
  completedTrial: number
  durationMs: number
}

export default function BreakScreen({
  completedTrial,
  durationMs,
}: BreakScreenProps) {
  const [remainingMs, setRemainingMs] = useState(durationMs)

  useEffect(() => {
    setRemainingMs(durationMs)
    const endAt = Date.now() + durationMs
    const tick = () => setRemainingMs(Math.max(0, endAt - Date.now()))
    tick()
    const intervalId = window.setInterval(tick, 250)
    return () => window.clearInterval(intervalId)
  }, [completedTrial, durationMs])

  return (
    <section
      aria-label={`Break after trial ${completedTrial}`}
      style={{
        width: 'min(42rem, 100%)',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        padding: '2.5rem',
        border: '1px solid #dbe3ee',
        borderRadius: '16px',
        background: '#ffffff',
        boxShadow: '0 12px 32px rgba(15, 23, 42, 0.08)',
      }}
    >
      <span
        style={{
          color: '#2563eb',
          fontSize: '0.9rem',
          fontWeight: 800,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}
      >
        Phase 2, Trial {completedTrial} complete
      </span>
      <h1 style={{ margin: 0, fontSize: '2.4rem' }}>Take a short break</h1>
      <p style={{ margin: 0, color: '#475569', fontSize: '1.05rem' }}>
        The next trial will begin automatically.
      </p>
      <strong
        aria-live="polite"
        style={{ marginTop: '0.75rem', fontSize: '3.5rem', fontVariantNumeric: 'tabular-nums' }}
      >
        {Math.ceil(remainingMs / 1000)}
      </strong>
    </section>
  )
}
