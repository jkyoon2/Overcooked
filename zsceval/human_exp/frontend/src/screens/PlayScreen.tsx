import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { ClientTimingPayload, GameState, RenderAckPayload } from '../lib/websocket'
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
  trialId: number
  gameState: GameState | null
  score: number
  timeRemainingMs: number
  playerHat: string
  aiHat: string
  gameEnded: boolean
  deliveries: Deliveries
  onPhaseReady: () => void
  onPlayerAction: (action: string, timing: ClientTimingPayload) => void
  onRenderAck: (payload: RenderAckPayload) => void
}

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
  onRenderAck,
}: PlayScreenProps) {
  const [overlayVisible, setOverlayVisible] = useState(false)
  const phaseReadySentRef = useRef(false)
  const playRenderedAckSentRef = useRef(false)

  // Keep a stable ref to onPhaseReady so the trigger effect doesn't re-fire
  // when the parent re-renders and passes a new function reference.
  const onPhaseReadyRef = useRef(onPhaseReady)
  useLayoutEffect(() => { onPhaseReadyRef.current = onPhaseReady })

  const playFinished = gameEnded || timeRemainingMs <= 0

  // Reset guard for each trial.
  useEffect(() => {
    phaseReadySentRef.current = false
    playRenderedAckSentRef.current = false
    setOverlayVisible(false)
  }, [trialId])

  // Show "Time's up!" overlay for 2s then advance.
  // Depend on the combined edge so a trailing game_end message cannot cancel
  // the timer started by the final game_step.
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
      if (!action) return

      const clientKeydownWallTimeMs = Date.now()
      const clientKeydownPerfMs = performance.now()

      e.preventDefault()
      onPlayerAction(action, {
        client_keydown_wall_time_ms: clientKeydownWallTimeMs,
        client_keydown_perf_ms: clientKeydownPerfMs,
      })
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

  const handleGameDrawn = useCallback(
    (drawnState: GameState) => {
      if (playRenderedAckSentRef.current) return

      playRenderedAckSentRef.current = true
      onRenderAck({
        event_type: 'PLAY_RENDERED',
        trial_id: trialId,
        step_index: drawnState.step_index,
        client_render_wall_time_ms: Date.now(),
        client_render_perf_ms: performance.now(),
      })
    },
    [onRenderAck, trialId],
  )

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
        <GameView state={gameState} playerHat={playerHat} aiHat={aiHat} onDrawn={handleGameDrawn}/>
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
