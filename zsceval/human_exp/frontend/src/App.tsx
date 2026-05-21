import { useEffect, useRef, useState } from 'react'
import { createWebSocketClient, type WebSocketClient } from './lib/websocket'
import PlayScreen from './screens/PlayScreen'

const WS_URL = 'ws://localhost:8000/ws'

export default function App() {
  const [connected, setConnected] = useState(false)
  const clientRef = useRef<WebSocketClient | null>(null)

  const connect = () => {
    if (clientRef.current) {
      clientRef.current.close()
    }
    const client = createWebSocketClient(WS_URL)
    clientRef.current = client

    // Poll readyState until open (WebSocket does not expose an onopen via our wrapper)
    const interval = setInterval(() => {
      if (client.isOpen()) {
        setConnected(true)
        clearInterval(interval)
      }
    }, 50)
  }

  const disconnect = () => {
    clientRef.current?.endGame()
    clientRef.current?.close()
    clientRef.current = null
    setConnected(false)
  }

  // Clean up on unmount
  useEffect(() => {
    return () => {
      clientRef.current?.close()
    }
  }, [])

  if (!connected || !clientRef.current) {
    return (
      <div style={{ fontFamily: 'monospace', padding: '2rem' }}>
        <h1>Neurocontroller</h1>
        <p>WebSocket: <strong>disconnected</strong></p>
        <button onClick={connect}>Connect</button>
      </div>
    )
  }

  return (
    <div style={{ fontFamily: 'monospace', padding: '0.5rem' }}>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem' }}>
        <strong>Neurocontroller</strong>
        <span style={{ color: 'green' }}>● connected</span>
        <button onClick={disconnect}>Disconnect</button>
      </div>
      <PlayScreen client={clientRef.current} layout="corner_onion_tomato" />
    </div>
  )
}
