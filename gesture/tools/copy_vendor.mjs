// Copies runtime WASM assets out of node_modules so Vite serves them statically.
import { cpSync, mkdirSync } from 'node:fs'

const root = new URL('../', import.meta.url).pathname
const copies = {
  'node_modules/@litertjs/core/wasm': 'public/vendor/litert',
  // Only used by check.html, which compares our pipeline against MediaPipe's own.
  'node_modules/@mediapipe/tasks-vision/wasm': 'public/vendor/mediapipe',
}
mkdirSync(`${root}public/vendor`, { recursive: true })
for (const [from, to] of Object.entries(copies)) cpSync(root + from, root + to, { recursive: true })
