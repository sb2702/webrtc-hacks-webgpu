// Runs check.html in headless Chrome and exits non-zero if any image fails.
// Needs `npm run fetch-test-images` first.
import { startServer, launch, waitForGPU } from './browser.mjs'

const server = await startServer()
const { browser, page } = await launch()
try {
  await waitForGPU(page, server.url)
  await page.goto(`${server.url}/check.html`)
  const { rows, pass } = await page.waitForFunction(() => window.checkResults, null, { timeout: 120_000 }).then((h) => h.jsonValue())
  const label = (x) => (x ? `${x.name} ${x.score.toFixed(2)}` : '—')
  for (const r of rows) {
    console.log(
      `${r.name.padEnd(24)} ${r.pass ? 'PASS' : 'FAIL'}  gesture ${label(r.ours?.gesture)} / ${label(r.mediapipe?.gesture)}` +
        `  hand ${label(r.ours?.handedness)} / ${label(r.mediapipe?.handedness)}` +
        (r.landmarkError ? `  landmarks mean ${(r.landmarkError.mean * 100).toFixed(2)}% max ${(r.landmarkError.max * 100).toFixed(2)}%` : '') +
        (r.landmarkBias ? ` bias ${r.landmarkBias.map((v) => v.toFixed(1)).join(',')}px` : '') +
        (r.mediapipeGpuVsCpu ? `  (MediaPipe GPU vs CPU ${(r.mediapipeGpuVsCpu.mean * 100).toFixed(2)}%)` : '') +
        (r.worldError !== undefined ? `  world ${(r.worldError * 1000).toFixed(1)} mm` : '') +
        (r.nmsDifference !== undefined ? `  nms Δ ${r.nmsDifference.toExponential(1)}` : '') +
        (r.notes.length ? `  (${r.notes.join(', ')})` : ''),
    )
  }
  if (!rows.length) console.log('No test images: run `npm run fetch-test-images` first.')
  process.exitCode = pass ? 0 : 1
} finally {
  await browser.close()
  await server.close()
}
