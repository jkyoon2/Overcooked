// WebSocket client wrapper with typed game message support (Task 2).

// ---------------------------------------------------------------------------
// Game message types (mirrors backend/data/schema.py)
// ---------------------------------------------------------------------------

export type PlayerSnapshot = {
  position: [number, number]
  orientation: [number, number]
  held_object: string | null
}

export type ObjectSnapshot = {
  name: string
  position: [number, number]
  soup_state?: string | null
  ingredients?: string[]
}

export type GameState = {
  players: PlayerSnapshot[]
  objects: ObjectSnapshot[]
  score: number
  step_index: number
  time_remaining: number
  layout_name: string
}

export type GameEvent = {
  event_type:
    | 'player_pickup'
    | 'player_deliver'
    | 'ai_pickup'
    | 'ai_deliver'
    | 'soup_drop'
    | 'soup_cooked'
  step_index: number
  agent_id: 0 | 1
  payload: Record<string, unknown>
}

// ---------------------------------------------------------------------------
// Inbound message discriminated union
// ---------------------------------------------------------------------------

export type HelloAckMessage = {
  type: 'hello_ack'
  payload: { echo: string; server_timestamp_ms: number }
}

export type GameStartMessage = {
  type: 'game_start'
  payload: { layout: string; initial_state: GameState }
}

export type GameStepMessage = {
  type: 'game_step'
  payload: {
    step_index: number
    state: GameState
    events: GameEvent[]
    rewards: [number, number]
    server_timestamp_ms: number
  }
}

export type GameEndMessage = {
  type: 'game_end'
  payload: { step_index: number; final_score: number }
}

export type ServerMessage = HelloAckMessage | GameStartMessage | GameStepMessage | GameEndMessage

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export type GameMessageHandlers = {
  onHelloAck?: (msg: HelloAckMessage) => void
  onGameStart?: (msg: GameStartMessage) => void
  onGameStep?: (msg: GameStepMessage) => void
  onGameEnd?: (msg: GameEndMessage) => void
}

export type WebSocketClient = {
  sendRaw: (data: string) => void
  sendJson: (obj: unknown) => void
  startGame: (layout: string) => void
  sendPlayerAction: (action: string) => void
  endGame: () => void
  setHandlers: (handlers: GameMessageHandlers) => void
  close: () => void
  isOpen: () => boolean
}

export function createWebSocketClient(url: string): WebSocketClient {
  const socket = new WebSocket(url)
  let handlers: GameMessageHandlers = {}

  socket.onmessage = (event) => {
    let msg: ServerMessage
    try {
      msg = JSON.parse(event.data as string) as ServerMessage
    } catch {
      return
    }

    switch (msg.type) {
      case 'hello_ack':
        handlers.onHelloAck?.(msg)
        break
      case 'game_start':
        handlers.onGameStart?.(msg)
        break
      case 'game_step':
        handlers.onGameStep?.(msg)
        break
      case 'game_end':
        handlers.onGameEnd?.(msg)
        break
    }
  }

  return {
    sendRaw: (data) => socket.send(data),
    sendJson: (obj) => socket.send(JSON.stringify(obj)),
    startGame: (layout) =>
      socket.send(JSON.stringify({ type: 'start_game', payload: { layout } })),
    sendPlayerAction: (action) =>
      socket.send(JSON.stringify({ type: 'player_action', payload: { action } })),
    endGame: () => socket.send(JSON.stringify({ type: 'end_game', payload: {} })),
    setHandlers: (h) => {
      handlers = h
    },
    close: () => socket.close(),
    isOpen: () => socket.readyState === WebSocket.OPEN,
  }
}
