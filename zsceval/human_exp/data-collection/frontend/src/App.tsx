import { useEffect, useState } from 'react'
import socket from './lib/socket'
import IdScreen from './screens/IdScreen'
import RecipeScreen from './screens/RecipeScreen'
import PlayScreen from './screens/PlayScreen'
import DoneScreen from './screens/DoneScreen'

const LAYOUTS = ['1_forced_easy', '1_forced_hard', '1_incentivized_easy', '1_incentivized_hard']

type Screen = 'id' | 'recipe' | 'play' | 'done'

export type GameState = {
  players: Array<{
    position: [number, number]
    orientation: [number, number]
    held_object: string | null
    held_object_ingredients: string[]
  }>
  objects: Array<{
    name: string
    position: [number, number]
    soup_state?: string
    ingredients?: string[]
  }>
  score: number
  step_index: number
  time_remaining: number
  layout_name: string
}

export default function App() {
  const [screen, setScreen] = useState<Screen>('id')
  const [participantId, setParticipantId] = useState('')
  const [trialIndex, setTrialIndex] = useState(0)
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [score, setScore] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(400)

  useEffect(() => {
    socket.on('participant_ack', () => setScreen('recipe'))

    socket.on('trial_start', (data: { initial_state: GameState; trial_index: number }) => {
      setGameState(data.initial_state)
      setScore(0)
      setTimeRemaining(400)
      setScreen('play')
    })

    socket.on('game_step', (data: { state: GameState }) => {
      setGameState(data.state)
      setScore(data.state.score)
      setTimeRemaining(data.state.time_remaining)
    })

    socket.on('trial_end', (data: { next_trial_index: number; all_done: boolean }) => {
      if (data.all_done) {
        setScreen('done')
      } else {
        setTrialIndex(data.next_trial_index)
        setScreen('recipe')
      }
    })

    return () => {
      socket.off('participant_ack')
      socket.off('trial_start')
      socket.off('game_step')
      socket.off('trial_end')
    }
  }, [])

  const handleSetId = (id: string) => {
    setParticipantId(id)
    socket.emit('set_participant', { participant_id: id })
  }

  const handleRecipeSelect = (recipe: string) => {
    socket.emit('start_trial', { recipe })
  }

  const handlePlayerAction = (player: 0 | 1, action: string) => {
    socket.emit('player_action', { player, action })
  }

  if (screen === 'id') return <IdScreen onSubmit={handleSetId} />
  if (screen === 'recipe') return (
    <RecipeScreen
      trialIndex={trialIndex}
      totalTrials={LAYOUTS.length}
      layout={LAYOUTS[trialIndex]}
      onSelect={handleRecipeSelect}
    />
  )
  if (screen === 'play') return (
    <PlayScreen
      participantId={participantId}
      trialIndex={trialIndex}
      totalTrials={LAYOUTS.length}
      gameState={gameState}
      score={score}
      timeRemaining={timeRemaining}
      onPlayerAction={handlePlayerAction}
    />
  )
  return <DoneScreen participantId={participantId} />
}
