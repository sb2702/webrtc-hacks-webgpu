import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

const page = (name) => fileURLToPath(new URL(`./${name}.html`, import.meta.url))

export default defineConfig({
  server: {
    // To open the demo from another device through a tunnel (ngrok, cloudflared…):
    //   ALLOWED_HOSTS=your-tunnel.example.com npm run dev
    allowedHosts: process.env.ALLOWED_HOSTS?.split(',') ?? [],
    // Cross-origin isolation lets LiteRT use its threaded (SharedArrayBuffer) WASM build.
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
  },
  build: { rolldownOptions: { input: { index: page('index'), check: page('check') } } },
  optimizeDeps: { exclude: ['@litertjs/core', '@mediapipe/tasks-vision'] },
})
