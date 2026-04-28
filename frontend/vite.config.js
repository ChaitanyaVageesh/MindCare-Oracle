import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Inside Docker: API container is "api:8000". Outside Docker: localhost:8001.
const API_HOST = process.env.VITE_API_HOST || 'localhost'
const API_PORT = process.env.VITE_API_HOST ? '8000' : '8001'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        target: `http://${API_HOST}:${API_PORT}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
