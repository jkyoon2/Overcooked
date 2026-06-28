import { type CSSProperties } from 'react'

type Props = { participantId: string }

export default function DoneScreen({ participantId }: Props) {
  return (
    <main style={page}>
      <div style={card}>
        <div style={check}>✓</div>
        <h1 style={title}>All done!</h1>
        <p style={sub}>
          Data for <strong>{participantId}</strong> saved to{' '}
          <code style={codeSty}>data/trajectories/</code>.
        </p>
        <p style={note}>You may close this window.</p>
      </div>
    </main>
  )
}

const page: CSSProperties = {
  minHeight: '100vh', display: 'grid', placeItems: 'center',
  background: '#f8fafc', fontFamily: 'system-ui, sans-serif',
}
const card: CSSProperties = {
  background: '#fff', borderRadius: 20, padding: '3rem 3.5rem',
  boxShadow: '0 16px 48px rgba(15,23,42,0.10)', display: 'flex',
  flexDirection: 'column', alignItems: 'center', gap: '1rem', textAlign: 'center',
}
const check: CSSProperties = {
  width: '4rem', height: '4rem', borderRadius: '50%', background: '#dcfce7',
  color: '#16a34a', display: 'grid', placeItems: 'center', fontSize: '2rem', fontWeight: 700,
}
const title: CSSProperties = { margin: 0, fontSize: '1.8rem', fontWeight: 800, color: '#0f172a' }
const sub: CSSProperties = { margin: 0, color: '#475569', fontSize: '1rem' }
const codeSty: CSSProperties = {
  background: '#f1f5f9', padding: '0.15rem 0.45rem',
  borderRadius: 6, fontFamily: 'monospace', fontSize: '0.9rem',
}
const note: CSSProperties = { margin: 0, color: '#94a3b8', fontSize: '0.85rem' }
