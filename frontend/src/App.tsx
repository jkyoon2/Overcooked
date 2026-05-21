import { useEffect, useRef, useState } from 'react'
import { createWebSocketClient } from './lib/websocket'

type WSClient = ReturnType<typeof createWebSocketClient>

export default function App() {
  const [messages, setMessages] = useState<string[]>([])
  const clientRef = useRef<WSClient | null>(null)

  useEffect(() => {
    const client = createWebSocketClient('ws://localhost:8000/ws')
    clientRef.current = client
    client.onMessage((event) => {
      setMessages((prev) => [...prev, event.data as string])
    })
    return () => client.close()
  }, [])

  const sendHello = () => {
    clientRef.current?.send(
      JSON.stringify({ type: 'hello', payload: { message: 'hello from frontend' } }),
    )
  }

  return (
    <div style={{ fontFamily: 'monospace', padding: '2rem' }}>
      <h1>Neurocontroller — WebSocket Test</h1>
      <button onClick={sendHello}>Send hello</button>
      <br />
      <br />
      <textarea
        readOnly
        value={messages.join('\n')}
        rows={12}
        cols={80}
        placeholder="Received messages will appear here..."
      />
    </div>
  )
}
