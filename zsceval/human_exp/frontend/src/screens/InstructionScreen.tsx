import { useEffect, useRef, useState } from 'react'
import HatLegend from '../components/HatLegend'

type InstructionScreenProps = {
  trialId: number
  playerHat: string
  aiHat: string
  durationMs: number
  onReady: () => void
}

export default function InstructionScreen({
  trialId,
  playerHat,
  aiHat,
  durationMs,
  onReady,
}: InstructionScreenProps) {
  const [remainingMs, setRemainingMs] = useState(durationMs)
  const readySentRef = useRef(false)
  const onReadyRef = useRef(onReady)

  useEffect(() => {
    onReadyRef.current = onReady
  }, [onReady])

  useEffect(() => {
    readySentRef.current = false
    setRemainingMs(durationMs)

    const endAt = Date.now() + durationMs
    const tick = () => {
      const nextRemainingMs = Math.max(0, endAt - Date.now())
      setRemainingMs(nextRemainingMs)

      if (nextRemainingMs === 0 && !readySentRef.current) {
        readySentRef.current = true
        onReadyRef.current()
      }
    }

    tick()
    const intervalId = window.setInterval(tick, 250)
    return () => window.clearInterval(intervalId)
  }, [trialId, durationMs])

  const remainingSeconds = Math.ceil(remainingMs / 1000)

  return (
    <main
      style={{
        minHeight: '100vh',
        display: 'grid',
        placeItems: 'center',
        fontFamily: 'system-ui, sans-serif',
        background: '#f8fafc',
        color: '#111827',
        padding: '2rem',
      }}
    >
      <section
        aria-label={`Trial ${trialId} instruction`}
        style={{
          width: 'min(48rem, 100%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '2rem',
          textAlign: 'center',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          <h1
            style={{
              margin: 0,
              fontSize: 'clamp(2.5rem, 8vw, 5.5rem)',
              lineHeight: 1,
              letterSpacing: 0,
            }}
          >
            Trial {trialId}
          </h1>
          <p style={{ margin: 0, color: '#475569', fontSize: '1.1rem' }}>
            Get ready for the next round.
          </p>
        </div>

        <HatLegend playerHat={playerHat} aiHat={aiHat} />

        <div
          aria-live="polite"
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.35rem',
          }}
        >
          <span style={{ color: '#64748b', fontSize: '0.95rem', textTransform: 'uppercase' }}>
            Starting in
          </span>
          <strong
            style={{
              minWidth: '5rem',
              fontSize: 'clamp(3rem, 10vw, 6rem)',
              lineHeight: 1,
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {remainingSeconds}
          </strong>
        </div>
      </section>
    </main>
  )
}
