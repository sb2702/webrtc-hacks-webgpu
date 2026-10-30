import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

const page = (name) => fileURLToPath(new URL(`./web/${name}.html`, import.meta.url))

export default defineConfig({
  root: 'web',
  server: {
    // To open the pages from another device through a tunnel (ngrok, cloudflared…):
    //   ALLOWED_HOSTS=your-tunnel.example.com npm run dev
    allowedHosts: process.env.ALLOWED_HOSTS?.split(',') ?? [],
    // Cross-origin isolation lets LiteRT use its threaded (SharedArrayBuffer) build.
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
  },
  build: {
    outDir: '../dist',
    rolldownOptions: {
      input: Object.fromEntries(['index', 'bench', 'test', 'sustained', 'profile'].map((name) => [name, page(name)])),
    },
  },
  optimizeDeps: { exclude: ['@litertjs/core', '@mediapipe/tasks-vision'] },
})
