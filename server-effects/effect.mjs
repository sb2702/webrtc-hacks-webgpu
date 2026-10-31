// Server-side virtual background: video in, green-screened video out, on the GPU via Dawn.
//
//   node effect.mjs input.mp4 output.mp4 [--color 00b140] [--encoder libx264|h264_videotoolbox] [--passthrough]
//
// The input can be anything ffmpeg reads (file, RTMP, SRT, RTSP, ...).
// --passthrough skips the effect (decode → upload → readback → encode): the ceiling.
import { parseArgs } from 'node:util'
import { getDevice, releaseGPU } from './src/gpu.mjs'
import { probe, decode, encode } from './src/ffmpeg.mjs'
import { ServerEffect } from './src/effect.mjs'

const { values: opts, positionals: [input, output] } = parseArgs({
  allowPositionals: true,
  options: {
    color: { type: 'string', default: '00b140' },
    encoder: { type: 'string', default: 'libx264' },
    'in-flight': { type: 'string', default: '3' },
    passthrough: { type: 'boolean', default: false },
  },
})
if (!input || !output) {
  console.error('usage: node effect.mjs input.mp4 output.mp4 [--color 00b140] [--encoder libx264|h264_videotoolbox] [--passthrough]')
  process.exit(1)
}

const video = probe(input)
const { device, description } = await getDevice()
console.log(`GPU: ${description}`)
console.log(`${input}: ${video.width}×${video.height} @ ${video.fps.toFixed(2)} fps → ${output} (${opts.encoder})${opts.passthrough ? ', passthrough' : ''}`)

const color = [0, 2, 4].map((i) => parseInt(opts.color.slice(i, i + 2), 16) / 255).concat(1)
const inFlight = Number(opts['in-flight'])
const effect = await ServerEffect.create(device, { width: video.width, height: video.height, color, inFlight, passthrough: opts.passthrough })
const encoder = encode(output, video, opts.encoder)

// Keep up to `inFlight` frames on the GPU; write results to the encoder in order.
const start = performance.now()
const queue = []
let frames = 0
for await (const rgba of decode(input, video)) {
  if (queue.length === inFlight) await encoder.write(await queue.shift())
  queue.push(effect.process(rgba))
  frames++
}
for (const pending of queue) await encoder.write(await pending)
await encoder.close()
const seconds = (performance.now() - start) / 1000

console.log(`${frames} frames in ${seconds.toFixed(2)} s: ${(frames / seconds).toFixed(1)} fps (decode → effect → encode)`)
releaseGPU(device)
