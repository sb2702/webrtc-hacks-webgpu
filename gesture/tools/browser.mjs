// Shared by the command-line check: a Vite dev server plus your local Chrome
// with real GPU access.
import { createServer } from 'vite'
import { chromium } from 'playwright-core'

// Linux needs Vulkan enabled explicitly, or WebGPU silently falls back to
// SwiftShader (a CPU emulation); headless also needs ANGLE on Vulkan to reach
// the GPU.
const ARGS = process.platform === 'linux'
  ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--use-angle=vulkan']
  : ['--enable-unsafe-webgpu']

export async function startServer() {
  const server = await createServer({ configFile: new URL('../vite.config.js', import.meta.url).pathname, server: { port: 0, forwardConsole: false }, logLevel: 'error' })
  await server.listen()
  return { url: server.resolvedUrls.local[0].replace(/\/$/, ''), close: () => server.close() }
}

export async function launch() {
  const browser = await chromium.launch({
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : { channel: 'chrome' }),
    headless: true,
    args: ARGS,
  })
  const page = await browser.newPage()
  // LiteRT / MediaPipe log INFO lines through console.error; skip those and other known noise.
  const NOISE = /404|^(INFO|WARNING): |^[IW]\d{4} |=== Source Location|Created TensorFlow Lite/
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && !NOISE.test(m.text()) && console.error(`[page ${m.type()}]`, m.text()))
  page.on('pageerror', (e) => console.error('[page]', e.message))
  return { browser, page }
}

// Headless Chrome on Vulkan returns a null adapter until the GPU process is
// up, so poll for one (from a page that isn't the demo) before running anything.
export async function waitForGPU(page, origin) {
  await page.goto(`${origin}/src/style.css`)
  await page.waitForFunction(async () => !!(await navigator.gpu?.requestAdapter()), null, { timeout: 15_000, polling: 250 })
}
