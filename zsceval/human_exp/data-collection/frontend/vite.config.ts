import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendPort = process.env.VITE_BACKEND_PORT ?? '5050'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/socket.io': {
        target: `http://localhost:${backendPort}`,
        ws: true,
        changeOrigin: true,
      },
      '/health': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
})
