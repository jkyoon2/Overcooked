import { type CSSProperties, useState } from 'react'

type Props = { onSubmit: (id: string) => void }

export default function IdScreen({ onSubmit }: Props) {
  const [value, setValue] = useState('')

  const handleSubmit = () => {
    const id = value.trim()
    if (id) onSubmit(id)
  }

  return (
    <main style={page}>
      <div style={card}>
        <h1 style={title}>Overcooked Data Collection</h1>
        <p style={sub}>Enter your participant ID to begin.</p>
        <input
          style={input}
          type="text"
          placeholder="e.g. P001"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          autoFocus
        />
        <button style={btn} onClick={handleSubmit} disabled={!value.trim()}>
          Start →
        </button>
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
  flexDirection: 'column', gap: '1.25rem', minWidth: 320, alignItems: 'center',
}
const title: CSSProperties = { margin: 0, fontSize: '1.6rem', fontWeight: 800, color: '#0f172a' }
const sub: CSSProperties = { margin: 0, color: '#64748b', fontSize: '0.95rem' }
const input: CSSProperties = {
  width: '100%', padding: '0.75rem 1rem', fontSize: '1.1rem',
  border: '1.5px solid #cbd5e1', borderRadius: 10, outline: 'none',
  fontFamily: 'inherit', boxSizing: 'border-box',
}
const btn: CSSProperties = {
  width: '100%', padding: '0.75rem', fontSize: '1rem', fontWeight: 700,
  background: '#6d28d9', color: '#fff', border: 'none', borderRadius: 10,
  cursor: 'pointer',
}
