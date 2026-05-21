import { useCallback, useEffect, useRef, useState } from 'react'
import { type GameEndMessage, type GameStartMessage, type GameState, type GameEvent, type WebSocketClient } from '../lib/websocket'

type Props = {
  client: WebSocketClient
  layout?: string
}

const KEY_ACTION_MAP: Record<string, string> = {
  w: 'NORTH',
  W: 'NORTH',
  ArrowUp: 'NORTH',
  s: 'SOUTH',
  S: 'SOUTH',
  ArrowDown: 'SOUTH',
  d: 'EAST',
  D: 'EAST',
  ArrowRight: 'EAST',
  a: 'WEST',
  A: 'WEST',
  ArrowLeft: 'WEST',
  ' ': 'INTERACT',
}

export default function PlayScreen({ client, layout = 'corner_onion_tomato' }: Props) {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [events, setEvents] = useState<GameEvent[]>([])
  const [score, setScore] = useState(0)
  const [stepIndex, setStepIndex] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [gameActive, setGameActive] = useState(false)
  const [gameOver, setGameOver] = useState(false)
  const [finalScore, setFinalScore] = useState<number | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!gameActive) return
      const action = KEY_ACTION_MAP[e.key]
      if (action) {
        e.preventDefault()
        client.sendPlayerAction(action)
      }
    },
    [client, gameActive],
  )

  useEffect(() => {
    client.setHandlers({
      onGameStart: (msg: GameStartMessage) => {
        setGameState(msg.payload.initial_state)
        setScore(0)
        setStepIndex(0)
        setTimeRemaining(msg.payload.initial_state.time_remaining)
        setEvents([])
        setGameActive(true)
        setGameOver(false)
        setFinalScore(null)
        containerRef.current?.focus()
      },
      onGameStep: (msg) => {
        setGameState(msg.payload.state)
        setScore(msg.payload.state.score)
        setStepIndex(msg.payload.step_index)
        setTimeRemaining(msg.payload.state.time_remaining)
        if (msg.payload.events.length > 0) {
          setEvents((prev) => [...msg.payload.events, ...prev].slice(0, 20))
        }
      },
      onGameEnd: (msg: GameEndMessage) => {
        setGameActive(false)
        setGameOver(true)
        setFinalScore(msg.payload.final_score)
      },
    })
  }, [client])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const startGame = () => {
    client.startGame(layout)
  }

  const stopGame = () => {
    client.endGame()
    setGameActive(false)
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      style={{ fontFamily: 'monospace', padding: '1.5rem', outline: 'none' }}
    >
      <h2>PlayScreen — {layout}</h2>

      {/* HUD */}
      <div style={{ marginBottom: '1rem', display: 'flex', gap: '2rem' }}>
        <span>Score: <strong>{score}</strong></span>
        <span>Step: <strong>{stepIndex}</strong></span>
        <span>Time remaining: <strong>{timeRemaining}</strong> ticks</span>
        <span>Status: <strong>{gameOver ? 'GAME OVER' : gameActive ? 'PLAYING' : 'idle'}</strong></span>
      </div>

      {/* Controls */}
      <div style={{ marginBottom: '1rem' }}>
        <button onClick={startGame} disabled={gameActive}>
          Start Game
        </button>
        &nbsp;
        <button onClick={stopGame} disabled={!gameActive}>
          Stop Game
        </button>
        &nbsp;
        <small>WASD/arrows = move &nbsp;|&nbsp; Space = interact</small>
      </div>

      {gameOver && finalScore !== null && (
        <p style={{ color: 'green', fontWeight: 'bold' }}>
          Game over! Final score: {finalScore}
        </p>
      )}

      {/* Recent events */}
      <details open style={{ marginBottom: '1rem' }}>
        <summary>Recent events ({events.length})</summary>
        <ul style={{ maxHeight: '8rem', overflowY: 'auto', margin: 0, padding: '0 0 0 1.2rem' }}>
          {events.map((e, i) => (
            <li key={i} style={{ fontSize: '0.85rem' }}>
              [step {e.step_index}] agent{e.agent_id}: {e.event_type}
              {e.event_type.includes('deliver') ? ` (${(e.payload as Record<string,unknown>)['recipe']}, +${(e.payload as Record<string,unknown>)['reward']})` : ''}
            </li>
          ))}
        </ul>
      </details>

      {/* Raw game state JSON — no sprite rendering (Task 6 will do that) */}
      <details>
        <summary>Raw game state (JSON)</summary>
        <pre
          style={{
            background: '#f4f4f4',
            padding: '0.75rem',
            fontSize: '0.75rem',
            maxHeight: '30rem',
            overflowY: 'auto',
            border: '1px solid #ccc',
          }}
        >
          {gameState ? JSON.stringify(gameState, null, 2) : '(no game state yet)'}
        </pre>
      </details>
    </div>
  )
}
