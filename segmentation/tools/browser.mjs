// Shared by the test / bench runners: a Vite dev server plus a local Chrome
// with real GPU access.
import { createServer } from 'vite'
import { chromium } from 'playwright-core'

// WebNN is behind a flag in Chrome as of 2026.
const FEATURES = ['WebMachineLearningNeuralNetwork']

// Linux needs Vulkan enabled explicitly, or WebGPU silently falls back to
// SwiftShader (a CPU emulation). Headless additionally needs ANGLE on Vulkan
// to reach the GPU — but that flag removes WebGL2 and breaks WebGPU canvas
// snapshots (new VideoFrame(canvas) reads back blank), so headed runs must
// not use it.
function chromeArgs(headed) {
  if (process.platform !== 'linux') return ['--enable-unsafe-webgpu', `--enable-features=${FEATURES.join(',')}`]
  return ['--enable-unsafe-webgpu', `--enable-features=Vulkan,${FEATURES.join(',')}`, '--ignore-gpu-blocklist',
    ...(headed ? [] : ['--use-angle=vulkan'])]
}

export async function startServer() {
  const server = await createServer({ configFile: new URL('../vite.config.js', import.meta.url).pathname, server: { port: 0, forwardConsole: false }, logLevel: 'error' })
  await server.listen()
  return { url: server.resolvedUrls.local[0].replace(/\/$/, ''), close: () => server.close() }
}

// Headless Chrome is fine for compute (tests), but can't turn WebGPU canvases
// into VideoFrames (see chromeArgs), so the benchmark always runs headed.
export async function launch({ headed = false } = {}) {
  const browser = await chromium.launch({
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : { channel: 'chrome' }),
    headless: !headed,
    args: [...chromeArgs(headed), ...(headed ? ['--window-size=1000,800'] : [])],
  })
  const page = await browser.newPage()
  // LiteRT / MediaPipe log INFO lines through console.error; skip those and other known noise.
  const NOISE = /404|GPU stall due to ReadPixels|is intended for demos|^(INFO|WARNING): |^[IW]\d{4} |=== Source Location/
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && !NOISE.test(m.text()) && console.error(`[page ${m.type()}]`, m.text()))
  page.on('pageerror', (e) => console.error('[page]', e.message))
  return { browser, page }
}

// Headless Chrome on Vulkan returns a null adapter until the GPU process is
// up, so poll for one before running anything.
export async function waitForGPU(page, origin) {
  await page.goto(`${origin}/404`)
  await page.waitForFunction(async () => !!(await navigator.gpu?.requestAdapter()), null, { timeout: 15_000, polling: 250 })
}
