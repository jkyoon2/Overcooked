import { useEffect, useRef, type CSSProperties } from 'react'

type IdleScreenProps = {
  connected: boolean
  onStart: () => void
}

export default function IdleScreen({ connected, onStart }: IdleScreenProps) {
  const triggeredRef = useRef(false)

  useEffect(() => {
    triggeredRef.current = false
  }, [connected])

  useEffect(() => {
    if (!connected) return
    const handler = (event: KeyboardEvent) => {
      if (event.key !== 'Enter') return
      if (triggeredRef.current) return
      event.preventDefault()
      triggeredRef.current = true
      onStart()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [connected, onStart])

  return (
    <main style={pageStyle}>
      <section style={cardStyle} aria-live="polite">
        <p style={eyebrowStyle}>Neurocontroller</p>
        <h1 style={titleStyle}>
          {connected ? 'Press ENTER to start' : 'Connecting to server…'}
        </h1>
        <p style={bodyStyle}>
          {connected
            ? 'When the EEG setup is calibrated and you are ready, press the ENTER key to begin Phase 1.'
            : 'Waiting for the experiment server. The start button will activate once the connection is established.'}
        </p>
        <div style={pillRowStyle}>
          <span style={{ ...pillStyle, ...(connected ? pillReadyStyle : pillWaitingStyle) }}>
            {connected ? '● Connected' : '○ Connecting'}
          </span>
          <span style={pillNeutralStyle}>Phase 1 — 8 trials</span>
          <span style={pillNeutralStyle}>Phase 2 — replay & annotate</span>
        </div>
        <button
          type="button"
          onClick={() => {
            if (!connected || triggeredRef.current) return
            triggeredRef.current = true
            onStart()
          }}
          disabled={!connected}
          style={{ ...startButtonStyle, opacity: connected ? 1 : 0.55 }}
        >
          Start session
        </button>
      </section>
    </main>
  )
}

const pageStyle: CSSProperties = {
  minHeight: '100vh',
  display: 'grid',
  placeItems: 'center',
  padding: '2.5rem',
  background:
    'radial-gradient(circle at top, #ffffff 0%, #eef2ff 45%, #e0e7ff 100%)',
  fontFamily: 'system-ui, sans-serif',
  color: '#0f172a',
}

const cardStyle: CSSProperties = {
  width: 'min(40rem, 100%)',
  padding: '3rem 3rem 2.5rem',
  border: '1px solid #c7d2fe',
  borderRadius: '20px',
  background: '#ffffff',
  boxShadow: '0 24px 60px rgba(15, 23, 42, 0.12)',
  display: 'flex',
  flexDirection: 'column',
  gap: '1.25rem',
  alignItems: 'center',
  textAlign: 'center',
}

const eyebrowStyle: CSSProperties = {
  margin: 0,
  color: '#4338ca',
  fontWeight: 800,
  fontSize: '0.85rem',
  letterSpacing: '0.18em',
  textTransform: 'uppercase',
}

const titleStyle: CSSProperties = {
  margin: 0,
  fontSize: 'clamp(2rem, 6vw, 3.2rem)',
  lineHeight: 1.1,
  letterSpacing: '-0.01em',
}

const bodyStyle: CSSProperties = {
  margin: 0,
  color: '#475569',
  fontSize: '1.05rem',
  lineHeight: 1.55,
  maxWidth: '32rem',
}

const pillRowStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  justifyContent: 'center',
  gap: '0.6rem',
}

const pillStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '0.4rem',
  padding: '0.4rem 0.75rem',
  borderRadius: '999px',
  fontSize: '0.8rem',
  fontWeight: 700,
}

const pillReadyStyle: CSSProperties = {
  background: '#dcfce7',
  color: '#166534',
}

const pillWaitingStyle: CSSProperties = {
  background: '#fef3c7',
  color: '#92400e',
}

const pillNeutralStyle: CSSProperties = {
  ...pillStyle,
  background: '#eef2ff',
  color: '#4338ca',
}

const startButtonStyle: CSSProperties = {
  marginTop: '0.5rem',
  border: 0,
  borderRadius: '999px',
  padding: '0.95rem 2.2rem',
  background: '#4338ca',
  color: '#ffffff',
  font: 'inherit',
  fontSize: '1rem',
  fontWeight: 800,
  cursor: 'pointer',
  boxShadow: '0 10px 24px rgba(67, 56, 202, 0.35)',
}
