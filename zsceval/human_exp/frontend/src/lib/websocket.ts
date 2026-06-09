// WebSocket client wrapper with typed game message support (Task 2).

// ---------------------------------------------------------------------------
// Game message types (mirrors backend/data/schema.py)
// ---------------------------------------------------------------------------

export type ClientTimingPayload = {
  client_keydown_wall_time_ms?: number
  client_keydown_perf_ms?: number
  client_send_wall_time_ms?: number
  client_send_perf_ms?: number
}

export type PlayerSnapshot = {
  position: [number, number]
  orientation: [number, number]
  held_object: string | null
  held_object_ingredients: string[]
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

export type IntentAlignment =
  | 'yes_clearly'
  | 'yes_somewhat'
  | 'no_somewhat'
  | 'no_clearly'

export type RatingPayload = {
  quality: number
  intent_alignment: IntentAlignment
}

export type RenderAckPayload = {
  event_type: 'PLAY_RENDERED'
  trial_id?: number | null
  step_index?: number | null
  client_render_wall_time_ms: number
  client_render_perf_ms: number
}

export type TrialPhase = 'instruction' | 'play' | 'rating' | 'break'

export type TrialStartMessage = {
  type: 'trial_start'
  payload: {
    trial_id: number
    total_trials: number
    phase: 'instruction'
    duration_ms: number
    player_hat: string
    ai_hat: string
  }
}

export type PhaseChangeMessage = {
  type: 'phase_change'
  payload: { phase: TrialPhase; duration_ms: number }
}

export type RatingAckMessage = {
  type: 'rating_ack'
  payload: {
    trial_id: number
    excluded: boolean
    exclusion_reason: string | null
  }
}

export type SessionCompleteMessage = {
  type: 'session_complete'
  payload: { session_id: string }
}

export type PhaseThreeReplayTrial = {
  trial_id: number
  frames: GameState[]
}

export type PhaseThreeStartPayload = {
  session_id: string
  frame_duration_ms: number
  player_hat: string
  ai_hat: string
  trials: PhaseThreeReplayTrial[]
}

export type PhaseThreeStartMessage = {
  type: 'phase3_start'
  payload: PhaseThreeStartPayload
}

export type PhaseThreeSegmentSelection = {
  segment_id: string
  start_frame: number
  end_frame: number
  created_at_ms: number
}

export type PhaseThreeTrialSelection = {
  trial_id: number
  segments: PhaseThreeSegmentSelection[]
}

export type PhaseThreeCompleteMessage = {
  type: 'phase3_complete'
  payload: { session_id: string; saved: boolean }
}

export type ServerMessage =
  | HelloAckMessage
  | GameStartMessage
  | GameStepMessage
  | GameEndMessage
  | TrialStartMessage
  | PhaseChangeMessage
  | RatingAckMessage
  | SessionCompleteMessage
  | PhaseThreeStartMessage
  | PhaseThreeCompleteMessage

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export type GameMessageHandlers = {
  onHelloAck?: (msg: HelloAckMessage) => void
  onGameStart?: (msg: GameStartMessage) => void
  onGameStep?: (msg: GameStepMessage) => void
  onGameEnd?: (msg: GameEndMessage) => void
  onTrialStart?: (msg: TrialStartMessage) => void
  onPhaseChange?: (msg: PhaseChangeMessage) => void
  onRatingAck?: (msg: RatingAckMessage) => void
  onSessionComplete?: (msg: SessionCompleteMessage) => void
  onPhaseThreeStart?: (msg: PhaseThreeStartMessage) => void
  onPhaseThreeComplete?: (msg: PhaseThreeCompleteMessage) => void
}

export type WebSocketClient = {
  sendRaw: (data: string) => void
  sendJson: (obj: unknown) => void
  startGame: (layout: string) => void
  startSession: (sessionId: string) => void
  sendPhaseReady: () => void
  submitRating: (rating: RatingPayload) => void
  submitPhaseThree: (trials: PhaseThreeTrialSelection[]) => void
  sendPlayerAction: (action: string, timing?: ClientTimingPayload) => void
  sendRenderAck: (payload: RenderAckPayload) => void
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
      case 'trial_start':
        handlers.onTrialStart?.(msg)
        break
      case 'phase_change':
        handlers.onPhaseChange?.(msg)
        break
      case 'rating_ack':
        handlers.onRatingAck?.(msg)
        break
      case 'session_complete':
        handlers.onSessionComplete?.(msg)
        break
      case 'phase3_start':
        handlers.onPhaseThreeStart?.(msg)
        break
      case 'phase3_complete':
        handlers.onPhaseThreeComplete?.(msg)
        break
    }
  }

  return {
    sendRaw: (data) => socket.send(data),
    sendJson: (obj) => socket.send(JSON.stringify(obj)),
    startGame: (layout) =>
      socket.send(JSON.stringify({ type: 'start_game', payload: { layout } })),
    startSession: (sessionId) =>
      socket.send(JSON.stringify({ type: 'start_session', payload: { session_id: sessionId } })),
    sendPhaseReady: () =>
      socket.send(JSON.stringify({ type: 'phase_ready', payload: {} })),
    submitRating: (rating) =>
      socket.send(JSON.stringify({ type: 'submit_rating', payload: rating })),
    submitPhaseThree: (trials) =>
      socket.send(JSON.stringify({ type: 'submit_phase3', payload: { trials } })),
    sendPlayerAction: (action, timing = {}) =>
      socket.send(JSON.stringify({
        type: 'player_action',
        payload: {
          action,
          ...timing,
          client_send_wall_time_ms: Date.now(),
          client_send_perf_ms: performance.now(),
        },
      })),
    sendRenderAck: (payload) =>
      socket.send(JSON.stringify({
        type: 'render_ack',
        payload,
      })),
    endGame: () => socket.send(JSON.stringify({ type: 'end_game', payload: {} })),
    setHandlers: (h) => {
      handlers = h
    },
    close: () => socket.close(),
    isOpen: () => socket.readyState === WebSocket.OPEN,
  }
}
