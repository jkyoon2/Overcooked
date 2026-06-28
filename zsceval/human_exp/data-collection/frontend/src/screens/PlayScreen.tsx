import { type CSSProperties, useEffect, useRef, useState } from 'react'
import type { GameState } from '../App'
import GameView from '../components/GameView'

// P0: WASD + Space   P1: Arrow keys + Enter
const KEY_MAP: Record<string, { player: 0 | 1; action: string }> = {
  w: { player: 0, action: 'NORTH' }, W: { player: 0, action: 'NORTH' },
  s: { player: 0, action: 'SOUTH' }, S: { player: 0, action: 'SOUTH' },
  d: { player: 0, action: 'EAST' },  D: { player: 0, action: 'EAST' },
  a: { player: 0, action: 'WEST' },  A: { player: 0, action: 'WEST' },
  ' ': { player: 0, action: 'INTERACT' },
  ArrowUp:    { player: 1, action: 'NORTH' },
  ArrowDown:  { player: 1, action: 'SOUTH' },
  ArrowRight: { player: 1, action: 'EAST' },
  ArrowLeft:  { player: 1, action: 'WEST' },
  Enter: { player: 1, action: 'INTERACT' },
}

type Props = {
  participantId: string
  trialIndex: number
  totalTrials: number
  gameState: GameState | null
  score: number
  timeRemaining: number
  onPlayerAction: (player: 0 | 1, action: string) => void
}

export default function PlayScreen({
  participantId, trialIndex, totalTrials, gameState, score, timeRemaining, onPlayerAction,
}: Props) {
  const [saved, setSaved] = useState(false)
  const doneRef = useRef(false)

  // Keep a ref to the latest onPlayerAction so the keydown listener
  // doesn't need to be re-added on every game_step re-render (10 Hz).
  const onPlayerActionRef = useRef(onPlayerAction)
  onPlayerActionRef.current = onPlayerAction

  useEffect(() => {
    doneRef.current = false
    setSaved(false)
  }, [trialIndex])

  useEffect(() => {
    if (timeRemaining <= 0 && !doneRef.current) {
      doneRef.current = true
      setSaved(true)
    }
  }, [timeRemaining])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const mapped = KEY_MAP[e.key]
      if (mapped) {
        e.preventDefault()
        onPlayerActionRef.current(mapped.player, mapped.action)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])  // mount/unmount only — ref keeps latest callback

  const totalSecs = Math.max(0, Math.ceil(timeRemaining))
  const mins = Math.floor(totalSecs / 60)
  const secs = totalSecs % 60
  const timeStr = `${mins}:${secs.toString().padStart(2, '0')}`

  return (
    <main style={page}>
      {saved && (
        <div style={overlay}>
          <div style={overlayCard}>
            <p style={{ margin: 0, fontSize: '1.8rem', fontWeight: 800 }}>Time&apos;s up!</p>
            <p style={{ margin: '0.5rem 0 0', color: '#64748b' }}>Saving data…</p>
          </div>
        </div>
      )}

      <div style={layout}>
        {/* Left panel */}
        <aside style={panel}>
          <p style={panelTitle}>Status</p>
          <Stat label="Participant" value={participantId} />
          <Stat label="Trial" value={`${trialIndex + 1} / ${totalTrials}`} />
          <Stat label="Score" value={String(Math.round(score))} big />
          <Stat label="Time" value={timeStr} big />
        </aside>

        {/* Game */}
        <section style={canvas}>
          {gameState
            ? <GameView state={gameState} hat0="blue" hat1="orange" />
            : <div style={waiting}>Waiting…</div>}
        </section>

        {/* Right panel */}
        <aside style={panel}>
          <p style={panelTitle}>Controls</p>
          <ControlRow label="P0 (blue)" keys={['W','A','S','D','Space']} />
          <ControlRow label="P1 (orange)" keys={['↑','←','↓','→','Enter']} />
        </aside>
      </div>
    </main>
  )
}

function Stat({ label, value, big }: { label: string; value: string; big?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
      <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
      <strong style={{ fontSize: big ? '1.8rem' : '1rem', fontVariantNumeric: 'tabular-nums' }}>{value}</strong>
    </div>
  )
}

function ControlRow({ label, keys }: { label: string; keys: string[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
      <span style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>{label}</span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
        {keys.map(k => (
          <kbd key={k} style={kbd}>{k}</kbd>
        ))}
      </div>
    </div>
  )
}

const page: CSSProperties = {
  position: 'relative', minHeight: '100vh', display: 'grid', placeItems: 'center',
  padding: '1.25rem', background: '#f8fafc', fontFamily: 'system-ui, sans-serif', boxSizing: 'border-box',
}
const layout: CSSProperties = {
  display: 'grid', gridTemplateColumns: '11rem 1fr 11rem', gap: '1rem',
  alignItems: 'start', width: '100%', maxWidth: '72rem',
}
const panel: CSSProperties = {
  background: '#fff', borderRadius: 16, padding: '1rem', display: 'flex',
  flexDirection: 'column', gap: '0.8rem', border: '1px solid #e2e8f0',
  boxShadow: '0 4px 16px rgba(15,23,42,0.06)', color: '#0f172a',
}
const panelTitle: CSSProperties = {
  margin: 0, fontWeight: 800, color: '#6d28d9', fontSize: '0.9rem',
  letterSpacing: '0.05em', textTransform: 'uppercase',
}
const canvas: CSSProperties = {
  background: '#fff', borderRadius: 16, padding: '1.25rem', display: 'flex',
  flexDirection: 'column', alignItems: 'center', gap: '0.5rem',
  border: '1px solid #e2e8f0', boxShadow: '0 8px 24px rgba(15,23,42,0.07)',
}
const waiting: CSSProperties = {
  padding: '3rem 2rem', color: '#94a3b8', border: '2px dashed #e2e8f0', borderRadius: 8,
}
const kbd: CSSProperties = {
  display: 'inline-block', padding: '0.1rem 0.4rem', border: '1px solid #cbd5e1',
  borderRadius: 4, background: '#f8fafc', fontSize: '0.75rem', fontWeight: 700,
  fontFamily: 'inherit',
}
const overlay: CSSProperties = {
  position: 'fixed', inset: 0, zIndex: 100, display: 'flex',
  alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)',
}
const overlayCard: CSSProperties = {
  background: '#fff', borderRadius: 14, padding: '2rem 3.5rem',
  textAlign: 'center', boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
}
