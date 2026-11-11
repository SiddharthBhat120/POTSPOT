import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: API on :8002. Production: FastAPI serves dist/. base './' lets the same build run on GitHub Pages.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 5175, proxy: { '/api': 'http://127.0.0.1:8002', '/media': 'http://127.0.0.1:8002' } },
  build: { outDir: 'dist', chunkSizeWarningLimit: 1500 },
})
