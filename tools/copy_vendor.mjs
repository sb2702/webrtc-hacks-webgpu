// Copies runtime WASM assets out of node_modules so Vite serves them statically.
import { cpSync, mkdirSync } from 'node:fs'

const root = new URL('../', import.meta.url).pathname
const copies = {
  'node_modules/@litertjs/core/wasm': 'web/public/vendor/litert',
  'node_modules/@mediapipe/tasks-vision/wasm': 'web/public/vendor/mediapipe',
  'node_modules/web-demuxer/dist/wasm-files/web-demuxer.wasm': 'web/public/vendor/web-demuxer.wasm',
}
mkdirSync(`${root}web/public/vendor`, { recursive: true })
for (const [from, to] of Object.entries(copies)) cpSync(root + from, root + to, { recursive: true })
