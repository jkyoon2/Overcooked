import { type CSSProperties } from 'react'

type Props = {
  trialIndex: number
  totalTrials: number
  layout: string
  onSelect: (recipe: string) => void
}

const RECIPES = [
  { code: 'TTT', label: '토마토 × 3', icon: '🍅🍅🍅' },
  { code: 'TTO', label: '토마토 × 2 + 양파 × 1', icon: '🍅🍅🧅' },
  { code: 'TOO', label: '토마토 × 1 + 양파 × 2', icon: '🍅🧅🧅' },
  { code: 'OOO', label: '양파 × 3', icon: '🧅🧅🧅' },
]

export default function RecipeScreen({ trialIndex, totalTrials, layout, onSelect }: Props) {
  return (
    <main style={page}>
      <div style={card}>
        <p style={badge}>Trial {trialIndex + 1} / {totalTrials}</p>
        <h1 style={title}>레시피를 선택하세요</h1>
        <p style={sub}><code style={code}>{layout}</code></p>
        <div style={grid}>
          {RECIPES.map(r => (
            <button key={r.code} style={recipeBtn} onClick={() => onSelect(r.code)}>
              <span style={icon}>{r.icon}</span>
              <strong style={recipeCode}>{r.code}</strong>
              <span style={recipeLabel}>{r.label}</span>
            </button>
          ))}
        </div>
      </div>
    </main>
  )
}

const page: CSSProperties = {
  minHeight: '100vh', display: 'grid', placeItems: 'center',
  background: '#f8fafc', fontFamily: 'system-ui, sans-serif',
}
const card: CSSProperties = {
  background: '#fff', borderRadius: 20, padding: '2.5rem',
  boxShadow: '0 16px 48px rgba(15,23,42,0.10)', display: 'flex',
  flexDirection: 'column', gap: '1.25rem', alignItems: 'center', minWidth: 360,
}
const badge: CSSProperties = {
  margin: 0, background: '#ede9fe', color: '#5b21b6', padding: '0.3rem 0.9rem',
  borderRadius: 999, fontWeight: 700, fontSize: '0.85rem',
}
const title: CSSProperties = { margin: 0, fontSize: '1.5rem', fontWeight: 800, color: '#0f172a' }
const sub: CSSProperties = { margin: 0, color: '#64748b', fontSize: '0.9rem' }
const code: CSSProperties = {
  background: '#f1f5f9', padding: '0.15rem 0.5rem',
  borderRadius: 6, fontFamily: 'monospace', fontSize: '0.9rem',
}
const grid: CSSProperties = {
  display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', width: '100%',
}
const recipeBtn: CSSProperties = {
  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem',
  padding: '1.1rem 0.75rem', borderRadius: 14, border: '2px solid #e2e8f0',
  background: '#fafafa', cursor: 'pointer', transition: 'border-color 0.15s',
}
const icon: CSSProperties = { fontSize: '1.6rem' }
const recipeCode: CSSProperties = { fontSize: '1rem', color: '#6d28d9', fontWeight: 800 }
const recipeLabel: CSSProperties = { fontSize: '0.78rem', color: '#64748b', textAlign: 'center' }
