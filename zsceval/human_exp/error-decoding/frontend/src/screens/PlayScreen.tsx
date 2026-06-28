import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from 'react'
import type { GameState } from '../lib/websocket'
import GameView from '../components/GameView'

const KEY_ACTION_MAP: Record<string, string> = {
  w: 'NORTH', W: 'NORTH', ArrowUp: 'NORTH',
  s: 'SOUTH', S: 'SOUTH', ArrowDown: 'SOUTH',
  d: 'EAST',  D: 'EAST',  ArrowRight: 'EAST',
  a: 'WEST',  A: 'WEST',  ArrowLeft: 'WEST',
  ' ': 'INTERACT',
}

type Deliveries = { ttt: number; tto: number; too: number; ooo: number }

const RECIPE_ROWS: ReadonlyArray<{
  key: keyof Deliveries
  icon: string
  points: number
}> = [
  { key: 'ttt', icon: '🍅🍅🍅', points: 5 },
  { key: 'tto', icon: '🍅🍅🧅', points: 10 },
  { key: 'too', icon: '🍅🧅🧅', points: 15 },
  { key: 'ooo', icon: '🧅🧅🧅', points: 20 },
]

type PlayScreenProps = {
  trialId: number
  gameState: GameState | null
  score: number
  timeRemainingMs: number
  playerHat: string
  aiHat: string
  gameEnded: boolean
  deliveries: Deliveries
  onPhaseReady: () => void
  onPlayerAction: (action: string) => void
}

const RESPONSIVE_CSS = `
.play-grid {
  width: 100%;
  max-width: 84rem;
  display: grid;
  grid-template-columns: minmax(10rem, 13rem) minmax(0, 1fr) minmax(10rem, 13rem);
  gap: 1.25rem;
  align-items: stretch;
  justify-items: stretch;
}
@media (max-width: 960px) {
  .play-grid {
    grid-template-columns: minmax(0, 1fr);
  }
  .play-side {
    max-width: none !important;
  }
}
`

export default function PlayScreen({
  trialId,
  gameState,
  score,
  timeRemainingMs,
  playerHat,
  aiHat,
  gameEnded,
  deliveries,
  onPhaseReady,
  onPlayerAction,
}: PlayScreenProps) {
  const [overlayVisible, setOverlayVisible] = useState(false)
  const phaseReadySentRef = useRef(false)
  const onPhaseReadyRef = useRef(onPhaseReady)
  useLayoutEffect(() => { onPhaseReadyRef.current = onPhaseReady })

  const playFinished = gameEnded || timeRemainingMs <= 0

  useEffect(() => {
    phaseReadySentRef.current = false
    setOverlayVisible(false)
  }, [trialId])

  useEffect(() => {
    if (!playFinished || phaseReadySentRef.current) return
    phaseReadySentRef.current = true
    setOverlayVisible(true)
    const timer = window.setTimeout(() => {
      onPhaseReadyRef.current()
    }, 2000)
    return () => window.clearTimeout(timer)
  }, [playFinished, trialId])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const action = KEY_ACTION_MAP[e.key]
      if (action) {
        e.preventDefault()
        onPlayerAction(action)
      }
    },
    [onPlayerAction],
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const totalSeconds = Math.max(0, Math.ceil(timeRemainingMs / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`
  const totalDeliveries = RECIPE_ROWS.reduce(
    (sum, row) => sum + deliveries[row.key],
    0,
  )

  return (
    <main style={pageStyle}>
      <style>{RESPONSIVE_CSS}</style>
      {overlayVisible && (
        <div style={overlayBackdropStyle}>
          <div style={overlayCardStyle}>Time&apos;s up!</div>
        </div>
      )}

      <div className="play-grid">
        <aside className="play-side" style={sideCardStyle}>
          <h2 style={panelTitleStyle}>Live Status</h2>

          <div style={statBlockStyle}>
            <span style={statLabelStyle}>Score</span>
            <strong style={statBigStyle}>{Math.round(score)}</strong>
          </div>
          <div style={statBlockStyle}>
            <span style={statLabelStyle}>Time left</span>
            <strong style={statBigStyle}>{timeStr}</strong>
          </div>

          <div style={statBlockStyle}>
            <span style={statLabelStyle}>Deliveries</span>
            <strong style={{ ...statBigStyle, fontSize: '1.3rem' }}>{totalDeliveries}</strong>
            <ul style={deliveryListStyle}>
              {RECIPE_ROWS.map((row) => (
                <li key={row.key} style={deliveryItemStyle}>
                  <span style={{ fontSize: '1.05rem' }}>{row.icon}</span>
                  <strong style={deliveryCountStyle}>×{deliveries[row.key]}</strong>
                </li>
              ))}
            </ul>
          </div>

          <div style={hatLineStyle}>
            <HatChip label="You" hat={playerHat} />
            <HatChip label="AI" hat={aiHat} />
          </div>
        </aside>

        <section style={canvasFrameStyle}>
          {gameState ? (
            <GameView state={gameState} playerHat={playerHat} aiHat={aiHat} />
          ) : (
            <div style={waitingStyle}>Waiting for game to start…</div>
          )}
          <p style={controlHintStyle}>
            Move with <kbd style={kbdStyle}>W</kbd> <kbd style={kbdStyle}>A</kbd>{' '}
            <kbd style={kbdStyle}>S</kbd> <kbd style={kbdStyle}>D</kbd> or the
            arrow keys. Press <kbd style={kbdStyle}>Space</kbd> to interact.
          </p>
        </section>

        <aside className="play-side" style={sideCardStyle}>
          <h2 style={panelTitleStyle}>Recipe Values</h2>
          <ul style={recipeListStyle}>
            {RECIPE_ROWS.map((row) => (
              <li key={row.key} style={recipeRowStyle}>
                <span style={{ fontSize: '1.25rem' }}>{row.icon}</span>
                <strong style={pointsBadgeStyle}>{row.points} pt</strong>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </main>
  )
}

type HatChipProps = { label: string; hat: string }

const HAT_COLORS: Record<string, string> = {
  blue: '#2563eb',
  red: '#dc2626',
  green: '#16a34a',
  yellow: '#ca8a04',
  purple: '#7c3aed',
  orange: '#ea580c',
  black: '#111827',
  white: '#f8fafc',
}

function HatChip({ label, hat }: HatChipProps) {
  const color = HAT_COLORS[hat.toLowerCase()] ?? hat
  return (
    <div style={hatChipStyle}>
      <span
        aria-hidden="true"
        style={{
          width: '0.9rem',
          height: '0.9rem',
          borderRadius: '999px 999px 4px 4px',
          background: color,
          border: hat.toLowerCase() === 'white' ? '1px solid #94a3b8' : 'none',
        }}
      />
      <span style={{ fontWeight: 700 }}>{label}</span>
    </div>
  )
}

const pageStyle: CSSProperties = {
  position: 'relative',
  width: '100%',
  minHeight: 'calc(100vh - 4.5rem)',
  display: 'grid',
  placeItems: 'center',
  padding: '1.25rem',
  fontFamily: 'system-ui, sans-serif',
  boxSizing: 'border-box',
}

const canvasFrameStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '0.75rem',
  padding: '1.25rem',
  borderRadius: '20px',
  background: '#ffffff',
  boxShadow: '0 18px 40px rgba(15, 23, 42, 0.08)',
  border: '1px solid #e2e8f0',
  minWidth: 0,
  width: '100%',
  boxSizing: 'border-box',
}

const sideCardStyle: CSSProperties = {
  width: '100%',
  maxWidth: '13rem',
  padding: '1rem',
  borderRadius: '18px',
  background: '#ffffff',
  border: '1px solid #e2e8f0',
  boxShadow: '0 12px 32px rgba(15, 23, 42, 0.06)',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.85rem',
  boxSizing: 'border-box',
  color: '#0f172a',
  minWidth: 0,
}

const panelTitleStyle: CSSProperties = {
  margin: 0,
  color: '#6d28d9',
  fontSize: '1.05rem',
  fontWeight: 800,
  letterSpacing: '0.04em',
}

const statBlockStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.15rem',
  minWidth: 0,
}

const statLabelStyle: CSSProperties = {
  color: '#64748b',
  fontSize: '0.75rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
}

const statBigStyle: CSSProperties = {
  fontSize: '1.6rem',
  fontVariantNumeric: 'tabular-nums',
  lineHeight: 1.1,
  overflowWrap: 'anywhere',
}

const deliveryListStyle: CSSProperties = {
  listStyle: 'none',
  margin: '0.4rem 0 0',
  padding: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: '0.3rem',
}

const deliveryItemStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '0.4rem',
  padding: '0.35rem 0.55rem',
  borderRadius: '10px',
  background: '#f8fafc',
  border: '1px solid #e2e8f0',
  color: '#0f172a',
  fontSize: '0.85rem',
}

const deliveryCountStyle: CSSProperties = {
  fontVariantNumeric: 'tabular-nums',
  color: '#6d28d9',
  fontWeight: 700,
}

const hatLineStyle: CSSProperties = {
  display: 'flex',
  gap: '0.4rem',
  flexWrap: 'wrap',
  marginTop: 'auto',
  paddingTop: '0.5rem',
  borderTop: '1px solid #e2e8f0',
}

const hatChipStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '0.35rem',
  padding: '0.3rem 0.55rem',
  borderRadius: '999px',
  background: '#eef2ff',
  color: '#4338ca',
  fontSize: '0.75rem',
}

const recipeListStyle: CSSProperties = {
  listStyle: 'none',
  margin: 0,
  padding: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: '0.45rem',
}

const recipeRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '0.5rem',
  padding: '0.5rem 0.65rem',
  borderRadius: '12px',
  background: '#fafaf9',
  border: '1px solid #e7e5e4',
  color: '#0f172a',
  fontSize: '0.9rem',
}

const pointsBadgeStyle: CSSProperties = {
  minWidth: '2.8rem',
  textAlign: 'center',
  padding: '0.25rem 0.5rem',
  borderRadius: '999px',
  background: '#fef3c7',
  color: '#92400e',
  fontWeight: 800,
  fontSize: '0.8rem',
}

const waitingStyle: CSSProperties = {
  padding: '3rem 2rem',
  color: '#6b7280',
  border: '2px dashed #d1d5db',
  borderRadius: 8,
  fontFamily: 'monospace',
}

const controlHintStyle: CSSProperties = {
  margin: 0,
  color: '#475569',
  fontSize: '0.85rem',
  textAlign: 'center',
}

const kbdStyle: CSSProperties = {
  display: 'inline-block',
  padding: '0.1rem 0.45rem',
  border: '1px solid #cbd5e1',
  borderRadius: '4px',
  background: '#f8fafc',
  color: '#0f172a',
  fontFamily: 'inherit',
  fontSize: '0.78rem',
  fontWeight: 700,
  margin: '0 0.05rem',
}

const overlayBackdropStyle: CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 100,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: 'rgba(0, 0, 0, 0.55)',
}

const overlayCardStyle: CSSProperties = {
  background: '#ffffff',
  borderRadius: 14,
  padding: '2rem 3.5rem',
  fontSize: '2rem',
  fontWeight: 700,
  textAlign: 'center',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
}
