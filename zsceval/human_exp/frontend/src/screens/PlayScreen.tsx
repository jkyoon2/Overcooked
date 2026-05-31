import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
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

const RECIPE_ICONS: Record<string, string> = {
  ttt: '🍅🍅🍅',
  tto: '🍅🍅🧅',
  too: '🍅🧅🧅',
  ooo: '🧅🧅🧅',
}

type PlayScreenProps = {
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

export default function PlayScreen({
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

  // Keep a stable ref to onPhaseReady so the trigger effect doesn't re-fire
  // when the parent re-renders and passes a new function reference.
  const onPhaseReadyRef = useRef(onPhaseReady)
  useLayoutEffect(() => { onPhaseReadyRef.current = onPhaseReady })

  // Reset guard on each new mount (new trial)
  useEffect(() => {
    phaseReadySentRef.current = false
    setOverlayVisible(false)
  }, [])

  // Show "Time's up!" overlay for 2s then advance.
  // Depends only on gameEnded / timeRemainingMs so function-ref churn
  // (onPhaseReady recreated on every App render) never cancels the timer.
  useEffect(() => {
    if ((gameEnded || timeRemainingMs <= 0) && !phaseReadySentRef.current) {
      phaseReadySentRef.current = true
      setOverlayVisible(true)
      const timer = window.setTimeout(() => {
        setOverlayVisible(false)
        onPhaseReadyRef.current()
      }, 2000)
      return () => window.clearTimeout(timer)
    }
  }, [gameEnded, timeRemainingMs])

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
  const timeStr  = `${minutes}:${seconds.toString().padStart(2, '0')}`

  return (
    <div style={{ position: 'relative', fontFamily: 'system-ui, sans-serif' }}>
      {/* "Time's up!" overlay */}
      {overlayVisible && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 100,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.55)',
          }}
        >
          <div
            style={{
              background: '#ffffff', borderRadius: 14,
              padding: '2rem 3.5rem', fontSize: '2rem',
              fontWeight: 700, textAlign: 'center',
              boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
            }}
          >
            Time&apos;s up!
          </div>
        </div>
      )}

      {/* HUD */}
      <div
        style={{
          display: 'flex', gap: '2rem', flexWrap: 'wrap',
          marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: 600, color: '#111827',
          alignItems: 'center',
        }}
      >
        <span>Score: {score}</span>
        <span>Time: {timeStr}</span>
        {(Object.keys(RECIPE_ICONS) as Array<keyof Deliveries>).map(key => (
          <span key={key} style={{ fontSize: '1rem', letterSpacing: '0.02em' }}>
            {RECIPE_ICONS[key]} ×{deliveries[key]}
          </span>
        ))}
      </div>

      {/* Game grid */}
      {gameState ? (
        <GameView state={gameState} playerHat={playerHat} aiHat={aiHat} />
      ) : (
        <div
          style={{
            padding: '3rem 2rem', color: '#6b7280',
            border: '2px dashed #d1d5db', borderRadius: 6, fontFamily: 'monospace',
          }}
        >
          Waiting for game to start…
        </div>
      )}

    </div>
  )
}
