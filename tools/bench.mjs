// Runs web/bench.html in a headed Chrome window on this machine, prints the
// results and saves the full report (config, environment, every run) as JSON.
//
//   npm run bench -- [--sample 360p|720p] [--mode transcode|process] [--runs 5]
//                    [--order sequential|interleaved] [--pipelines webgpu,webgl,...]
//                    [--check] [--out results/my-machine.json]
import { mkdirSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { startServer, launch, waitForGPU } from './browser.mjs'

const { values: args } = parseArgs({
  options: {
    mode: { type: 'string', default: 'transcode' },
    runs: { type: 'string', default: '5' },
    order: { type: 'string', default: 'sequential' },
    sample: { type: 'string', default: '360p' },
    pipelines: { type: 'string' },
    check: { type: 'boolean' },
    out: { type: 'string' },
  },
})
const server = await startServer()
const { browser, page } = await launch({ headed: true })
await waitForGPU(page, server.url)
const query = new URLSearchParams({
  mode: args.mode, runs: args.runs, order: args.order, sample: args.sample, ...(args.pipelines && { pipelines: args.pipelines }),
})
await page.goto(`${server.url}/bench.html?${query}`)
await page.waitForFunction(() => window.bench)
console.log(await page.textContent('#env'))
page.setDefaultTimeout(0)

if (args.check) {
  const checks = await page.evaluate(() => window.bench.checkAll())
  for (const [name, r] of Object.entries(checks)) {
    console.log(name.padEnd(20) + (r.error ? `ERROR ${r.error}` :
      `background ${(r.background * 100).toFixed(1)}%` + (r.agreement != null ? `  agrees with webgpu on ${(r.agreement * 100).toFixed(1)}% of pixels` : '')))
  }
} else {
  const report = await page.evaluate(() => window.bench.runAll())
  const { config: c, environment: env } = report
  console.log(`\n${c.width}x${c.height}, ${c.frames} frames, mode ${c.mode}, order ${c.order}, ${c.runs} timed runs`)
  console.log('pipeline'.padEnd(20) + 'median'.padStart(8) + 'min'.padStart(8) + 'max'.padStart(8) + 'init ms'.padStart(9))
  for (const r of report.results) {
    if (r.error) { console.log(`${r.name.padEnd(20)}ERROR ${r.error}`); continue }
    console.log(r.name.padEnd(20) + [r.medianFps, r.minFps, r.maxFps].map((f) => f.toFixed(1).padStart(8)).join('') + r.initMs.toFixed(0).padStart(9))
  }
  const gpu = [env.webgpu?.vendor, env.webgpu?.architecture].filter(Boolean).join('-') || 'nogpu'
  const out = args.out ?? `results/${env.date.slice(0, 10)}-${process.platform}-${gpu}-${c.mode}-${c.height}p.json`
  mkdirSync('results', { recursive: true })
  writeFileSync(out, JSON.stringify(report, null, 2) + '\n')
  console.log(`\nsaved ${out}`)
}
await browser.close()
await server.close()
