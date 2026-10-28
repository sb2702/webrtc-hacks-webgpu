// Runs web/test.html in headless Chrome and exits non-zero if a backend fails.
import { startServer, launch, waitForGPU } from './browser.mjs'

const server = await startServer()
const { browser, page } = await launch()
await waitForGPU(page, server.url)
await page.goto(`${server.url}/test.html`)
const results = await page.waitForFunction(() => window.testResults, null, { timeout: 60_000 }).then((h) => h.jsonValue())
for (const [name, r] of Object.entries(results))
  console.log(`${name.padEnd(7)} ${r.pass ? 'PASS' : 'FAIL'}  ${r.error ?? `max err ${r.maxErr.toExponential(2)}, ${r.dispatches} passes`}`)
await browser.close()
await server.close()
process.exit(Object.values(results).every((r) => r.pass) ? 0 : 1)
