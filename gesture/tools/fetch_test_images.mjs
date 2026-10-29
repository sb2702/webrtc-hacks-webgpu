// Downloads a few hand photos from MediaPipe's public test assets into
// public/test_images/ for check.html. Not committed: they're Google's.
import { mkdirSync, writeFileSync } from 'node:fs'

const NAMES = ['thumb_up', 'pointing_up', 'pointing_up_rotated', 'victory', 'fist', 'left_hands', 'right_hands']
const dir = new URL('../public/test_images/', import.meta.url)
mkdirSync(dir, { recursive: true })
for (const name of NAMES) {
  const response = await fetch(`https://storage.googleapis.com/mediapipe-assets/${name}.jpg`)
  if (!response.ok) throw new Error(`${name}.jpg: HTTP ${response.status}`)
  writeFileSync(new URL(`${name}.jpg`, dir), Buffer.from(await response.arrayBuffer()))
  console.log(`public/test_images/${name}.jpg`)
}
writeFileSync(new URL('index.json', dir), JSON.stringify(NAMES.map((n) => `${n}.jpg`)))
