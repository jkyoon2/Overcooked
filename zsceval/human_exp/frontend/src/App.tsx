import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import {
  createWebSocketClient,
  type ClientTimingPayload,
  type GameState,
  type PhaseThreeStartPayload,
  type PhaseThreeTrialSelection,
  type RatingPayload,
  type RenderAckPayload,
  type TrialPhase,
  type TrialStartMessage,
  type WebSocketClient,
} from './lib/websocket'
import InstructionScreen from './screens/InstructionScreen'
import PlayScreen from './screens/PlayScreen'
import RatingScreen from './screens/RatingScreen'
import BreakScreen from './screens/BreakScreen'
import PhaseThreeScreen from './screens/PhaseThreeScreen'

const DEFAULT_SESSION_ID = 'TEST_S01'

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected'
type AppPhase = 'idle' | 'waiting' | TrialPhase | 'phase3' | 'complete'
type InstructionPayload = TrialStartMessage['payload']

export default function App() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting')
  const [phase, setPhase] = useState<AppPhase>('idle')
  const [instruction, setInstruction] = useState<InstructionPayload | null>(null)
  const [playerHat, setPlayerHat] = useState('')
  const [aiHat, setAiHat] = useState('')
  const [currentTrialId, setCurrentTrialId] = useState<number | null>(null)
  const [ratingDurationMs, setRatingDurationMs] = useState(20_000)
  const [breakDurationMs, setBreakDurationMs] = useState(5_000)
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [gameEnded, setGameEnded] = useState(false)
  const [phaseThreeData, setPhaseThreeData] = useState<PhaseThreeStartPayload | null>(null)
  const [phaseThreeSubmitting, setPhaseThreeSubmitting] = useState(false)
  const [deliveries, setDeliveries] = useState({ ttt: 0, tto: 0, too: 0, ooo: 0 })
  const clientRef = useRef<WebSocketClient | null>(null)
  const connectPollRef = useRef<number | null>(null)
  const connectTimeoutRef = useRef<number | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const sessionStartedRef = useRef(false)

  const connect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.close()
    }
    if (connectPollRef.current !== null) {
      window.clearInterval(connectPollRef.current)
      connectPollRef.current = null
    }
    if (connectTimeoutRef.current !== null) {
      window.clearTimeout(connectTimeoutRef.current)
      connectTimeoutRef.current = null
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    setConnectionStatus('connecting')
    setPhase('idle')
    setInstruction(null)
    sessionStartedRef.current = false

    const client = createWebSocketClient(getWebSocketUrl())
    client.setHandlers({
      onTrialStart: (msg) => {
        setInstruction(msg.payload)
        setPhase(msg.payload.phase)
        setPlayerHat(msg.payload.player_hat)
        setAiHat(msg.payload.ai_hat)
        setCurrentTrialId(msg.payload.trial_id)
      },
      onPhaseChange: (msg) => {
        setInstruction(null)
        setPhase(msg.payload.phase)
        if (msg.payload.phase === 'play') {
          // reset game state before game_start arrives
          setGameState(null)
          setGameEnded(false)
          setDeliveries({ ttt: 0, tto: 0, too: 0, ooo: 0 })
        }
        if (msg.payload.phase === 'rating') {
          setRatingDurationMs(msg.payload.duration_ms)
        }
        if (msg.payload.phase === 'break') {
          setBreakDurationMs(msg.payload.duration_ms)
        }
      },
      onSessionComplete: () => {
        setInstruction(null)
        setPhase('complete')
      },
      onPhaseThreeStart: (msg) => {
        setInstruction(null)
        setPhaseThreeData(msg.payload)
        setPhaseThreeSubmitting(false)
        setPhase('phase3')
      },
      onPhaseThreeComplete: () => {
        setPhaseThreeSubmitting(false)
        setPhase('complete')
      },
      onRatingAck: () => {
        setPhase('waiting')
      },
      onGameStart: (msg) => {
        setGameState(msg.payload.initial_state)
        setGameEnded(false)
        setDeliveries({ ttt: 0, tto: 0, too: 0, ooo: 0 })
      },
      onGameStep: (msg) => {
        setGameState(msg.payload.state)
        if (msg.payload.events.length > 0) {
          for (const e of msg.payload.events) {
            if (e.event_type === 'player_deliver' || e.event_type === 'ai_deliver') {
              const recipe = String(e.payload['recipe']) as 'ttt' | 'tto' | 'too' | 'ooo'
              if (recipe === 'ttt' || recipe === 'tto' || recipe === 'too' || recipe === 'ooo') {
                setDeliveries((prev) => ({ ...prev, [recipe]: prev[recipe] + 1 }))
              }
            }
          }
        }
      },
      onGameEnd: () => {
        setGameEnded(true)
      },
    })
    clientRef.current = client

    // Poll readyState until open (WebSocket does not expose an onopen via our wrapper)
    connectPollRef.current = window.setInterval(() => {
      if (client.isOpen()) {
        setConnectionStatus('connected')
        if (connectPollRef.current !== null) {
          window.clearInterval(connectPollRef.current)
          connectPollRef.current = null
        }
        if (connectTimeoutRef.current !== null) {
          window.clearTimeout(connectTimeoutRef.current)
          connectTimeoutRef.current = null
        }
      }
    }, 50)

    connectTimeoutRef.current = window.setTimeout(() => {
      if (clientRef.current === client && !client.isOpen()) {
        client.close()
        clientRef.current = null
        setConnectionStatus('disconnected')
        if (connectPollRef.current !== null) {
          window.clearInterval(connectPollRef.current)
          connectPollRef.current = null
        }
      }
    }, 5000)
  }, [])

  const startSession = useCallback(() => {
    if (!clientRef.current?.isOpen()) return
    setPhase('waiting')
    setInstruction(null)
    sessionStartedRef.current = true
    clientRef.current.startSession(DEFAULT_SESSION_ID)
  }, [])

  const handleInstructionReady = useCallback(() => {
    clientRef.current?.sendPhaseReady()
    setPhase('waiting')
  }, [])

  // Stable callbacks so PlayScreen effects don't re-fire on every App re-render
  const handlePhaseReady = useCallback(() => {
    clientRef.current?.sendPhaseReady()
    setPhase('waiting')
  }, [])

  const handlePlayerAction = useCallback((
    action: string,
    timing: ClientTimingPayload,
  ) => {
    clientRef.current?.sendPlayerAction(action, timing)
  }, [])

  const handleRenderAck = useCallback((payload: RenderAckPayload) => {
    clientRef.current?.sendRenderAck(payload)
  }, [])

  const handleRatingSubmit = useCallback((rating: RatingPayload) => {
    clientRef.current?.submitRating(rating)
  }, [])

  const handlePhaseThreeSubmit = useCallback((trials: PhaseThreeTrialSelection[]) => {
    if (!clientRef.current?.isOpen()) return
    setPhaseThreeSubmitting(true)
    clientRef.current.submitPhaseThree(trials)
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clientRef.current?.close()
      if (connectPollRef.current !== null) {
        window.clearInterval(connectPollRef.current)
      }
      if (connectTimeoutRef.current !== null) {
        window.clearTimeout(connectTimeoutRef.current)
      }
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
      }
    }
  }, [connect])

  useEffect(() => {
    if (connectionStatus !== 'disconnected') return

    reconnectTimerRef.current = window.setTimeout(() => {
      connect()
    }, 2000)

    return () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
    }
  }, [connectionStatus, connect])

  useEffect(() => {
    if (
      connectionStatus === 'connected'
      && phase === 'idle'
      && !sessionStartedRef.current
      && clientRef.current?.isOpen()
    ) {
      startSession()
    }
  }, [connectionStatus, phase, startSession])

  useEffect(() => {
    if (phase !== 'break') return
    const timer = window.setTimeout(() => {
      clientRef.current?.sendPhaseReady()
      setPhase('waiting')
    }, breakDurationMs)
    return () => window.clearTimeout(timer)
  }, [phase, breakDurationMs])

  if (connectionStatus !== 'connected' || !clientRef.current) {
    return (
      <div style={pageStyle}>
        <h1>Neurocontroller</h1>
        <p>
          {connectionStatus === 'connecting'
            ? 'Connecting to the experiment server...'
            : 'Connection lost. Retrying automatically...'}
        </p>
      </div>
    )
  }

  if (phase === 'instruction' && instruction) {
    return (
      <InstructionScreen
        trialId={instruction.trial_id}
        playerHat={instruction.player_hat}
        aiHat={instruction.ai_hat}
        durationMs={instruction.duration_ms}
        onReady={handleInstructionReady}
      />
    )
  }

  if (phase === 'rating' && currentTrialId !== null) {
    return (
      <RatingScreen
        trialId={currentTrialId}
        durationMs={ratingDurationMs}
        onSubmit={handleRatingSubmit}
      />
    )
  }

  if (phase === 'phase3' && phaseThreeData !== null) {
    return (
      <PhaseThreeScreen
        data={phaseThreeData}
        submitting={phaseThreeSubmitting}
        onSubmit={handlePhaseThreeSubmit}
      />
    )
  }

  return (
    <div style={{ minHeight: '100vh', fontFamily: 'system-ui, sans-serif', background: '#f8fafc' }}>
      <header
        style={{
          display: 'flex',
          gap: '1rem',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '1rem 1.25rem',
          borderBottom: '1px solid #d7dde8',
          background: '#ffffff',
        }}
      >
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', minWidth: 0 }}>
          <strong>Neurocontroller</strong>
          <span style={{ color: '#2563eb', fontWeight: 700 }}>
            {phase === 'complete'
              ? 'Experiment complete'
              : currentTrialId === null
                ? 'Phase 2'
                : `Phase 2, Trial ${currentTrialId}`}
          </span>
        </div>
      </header>

      <main style={pageStyle}>
        {phase === 'idle' && (
          <>
            <h1>Session</h1>
            <p>Starting session...</p>
          </>
        )}

        {phase === 'waiting' && <p>Waiting for the next phase...</p>}
        {phase === 'play' && (
          <PlayScreen
            trialId={currentTrialId ?? 1}
            gameState={gameState}
            score={gameState?.score ?? 0}
            timeRemainingMs={gameState !== null ? gameState.time_remaining * 100 : Infinity}
            playerHat={playerHat}
            aiHat={aiHat}
            gameEnded={gameEnded}
            deliveries={deliveries}
            onPhaseReady={handlePhaseReady}
            onPlayerAction={handlePlayerAction}
            onRenderAck={handleRenderAck}
          />
        )}
        {phase === 'rating' && <p>Preparing rating screen...</p>}
        {phase === 'break' && currentTrialId !== null && (
          <BreakScreen
            completedTrial={currentTrialId}
            durationMs={breakDurationMs}
          />
        )}
        {phase === 'complete' && (
          <section
            style={{
              width: 'min(42rem, 100%)',
              padding: '2.5rem',
              border: '1px solid #bbf7d0',
              borderRadius: '16px',
              background: '#f0fdf4',
            }}
          >
            <p
              style={{
                margin: '0 0 0.75rem',
                color: '#15803d',
                fontWeight: 800,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              Phase 3 complete
            </p>
            <h1 style={{ margin: 0 }}>Replay annotations are saved.</h1>
            <p style={{ margin: '0.75rem 0 0', color: '#475569' }}>
              Thank you. The experiment is complete.
            </p>
          </section>
        )}
      </main>
    </div>
  )
}

function getWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.hostname}:8000/ws`
}

const pageStyle = {
  display: 'flex',
  minHeight: 'calc(100vh - 4.5rem)',
  flexDirection: 'column',
  alignItems: 'flex-start',
  justifyContent: 'center',
  gap: '1rem',
  padding: '2rem',
  fontFamily: 'system-ui, sans-serif',
  color: '#111827',
} satisfies CSSProperties
