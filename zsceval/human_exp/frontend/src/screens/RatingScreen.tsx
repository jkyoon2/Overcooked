import { useEffect, useRef, useState } from 'react'
import RatingForm from '../components/RatingForm'
import type { IntentAlignment, RatingPayload } from '../lib/websocket'

type RatingScreenProps = {
  trialId: number
  durationMs: number
  onSubmit: (rating: RatingPayload) => void
}

const DEFAULT_RATING: RatingPayload = {
  quality: 4,
  intent_alignment: 'yes_somewhat',
}

export default function RatingScreen({ trialId, durationMs, onSubmit }: RatingScreenProps) {
  const [quality, setQuality] = useState<number | null>(null)
  const [intentAlignment, setIntentAlignment] = useState<IntentAlignment | null>(null)
  const [remainingMs, setRemainingMs] = useState(durationMs)
  const [submitted, setSubmitted] = useState(false)
  const submittedRef = useRef(false)
  const qualityRef = useRef<number | null>(quality)
  const intentAlignmentRef = useRef<IntentAlignment | null>(intentAlignment)
  const onSubmitRef = useRef(onSubmit)

  useEffect(() => {
    qualityRef.current = quality
  }, [quality])

  useEffect(() => {
    intentAlignmentRef.current = intentAlignment
  }, [intentAlignment])

  useEffect(() => {
    onSubmitRef.current = onSubmit
  }, [onSubmit])

  useEffect(() => {
    submittedRef.current = false
    setSubmitted(false)
    setQuality(null)
    setIntentAlignment(null)
    qualityRef.current = null
    intentAlignmentRef.current = null
    setRemainingMs(durationMs)

    const endAt = Date.now() + durationMs
    const submitOnce = () => {
      if (submittedRef.current) return
      submittedRef.current = true
      setSubmitted(true)
      onSubmitRef.current({
        quality: qualityRef.current ?? DEFAULT_RATING.quality,
        intent_alignment: intentAlignmentRef.current ?? DEFAULT_RATING.intent_alignment,
      })
    }

    const tick = () => {
      const nextRemainingMs = Math.max(0, endAt - Date.now())
      setRemainingMs(nextRemainingMs)
      if (nextRemainingMs === 0) {
        submitOnce()
      }
    }

    tick()
    const intervalId = window.setInterval(tick, 250)
    return () => window.clearInterval(intervalId)
  }, [trialId, durationMs])

  const submit = (rating: RatingPayload) => {
    if (submittedRef.current) return
    submittedRef.current = true
    setSubmitted(true)
    onSubmit(rating)
  }

  return (
    <main
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '1.5rem',
        padding: '2rem',
        background: '#f8fafc',
        color: '#111827',
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <section
        aria-label={`Trial ${trialId} rating`}
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          width: 'min(46rem, 100%)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '2.3rem', lineHeight: 1.05, letterSpacing: 0 }}>
              Trial {trialId} Rating
            </h1>
            <p style={{ margin: '0.5rem 0 0', color: '#475569' }}>
              Please answer both questions.
            </p>
          </div>
          <div
            aria-live="polite"
            style={{
              alignSelf: 'flex-start',
              minWidth: '5.5rem',
              border: '1px solid #cbd5e1',
              borderRadius: '8px',
              padding: '0.65rem 0.85rem',
              background: '#ffffff',
              textAlign: 'center',
              fontVariantNumeric: 'tabular-nums',
              fontWeight: 700,
            }}
          >
            {formatRemaining(remainingMs)}
          </div>
        </div>

        <RatingForm
          quality={quality}
          intentAlignment={intentAlignment}
          disabled={submitted}
          onQualityChange={setQuality}
          onIntentAlignmentChange={setIntentAlignment}
          onSubmit={submit}
        />
      </section>
    </main>
  )
}

function formatRemaining(remainingMs: number): string {
  const totalSeconds = Math.max(0, Math.ceil(remainingMs / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}
