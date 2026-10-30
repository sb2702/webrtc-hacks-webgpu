// End-to-end benchmark: decode -> background replacement -> encode -> mux, as
// fast as the pipeline allows (no real-time pacing), for each pipeline.
//
// What is and isn't timed (see docs/METHODOLOGY.md):
//   - The file is demuxed once up front; each run replays the cached
//     EncodedVideoChunks, so demuxing is excluded.
//   - Pipeline creation (model download, shader / graph compilation) is
//     excluded and reported separately as initMs.
//   - Each pipeline gets one untimed warm-up pass over the clip.
//   - A timed run is wall-clock time from the first chunk entering the decoder
//     to the muxer finishing (transcode) or the last frame being drained
//     (process).

import {
  SimpleDemuxer, SimpleMuxer, VideoDecodeStream, VideoProcessStream, VideoEncodeStream, getBitrate, getCodecString,
} from 'webcodecs-utils'
import { PIPELINES } from './pipelines.js'

// Forces a frame's GPU work to finish without reading back the whole frame:
// draw it into a 1x1 (GPU-backed) canvas and read that single pixel.
const probe = new OffscreenCanvas(1, 1).getContext('2d')
function waitForPixels(frame) {
  probe.drawImage(frame, 0, 0, 1, 1)
  probe.getImageData(0, 0, 1, 1)
}

// web-demuxer fetches this from a worker, so it needs an absolute URL.
const DEMUXER_WASM = new URL('/vendor/web-demuxer.wasm', location.href).href

// Each SimpleDemuxer holds its own WASM instance, so demux each file once and
// keep the chunks (EncodedVideoChunks are immutable and can be decoded again).
const demuxed = new WeakMap()
function demux(file) {
  if (!demuxed.has(file)) {
    demuxed.set(file, (async () => {
      const demuxer = new SimpleDemuxer(file, { wasmFilePath: DEMUXER_WASM })
      await demuxer.load()
      const config = await demuxer.getVideoDecoderConfig()
      const chunks = []
      await demuxer.videoStream().pipeTo(new WritableStream({ write: (chunk) => { chunks.push(chunk) } }))
      return { config, chunks }
    })())
  }
  return demuxed.get(file)
}

const chunkStream = (chunks) => new ReadableStream({
  start(controller) {
    for (const chunk of chunks) controller.enqueue(chunk)
    controller.close()
  },
})

const sorted = (xs) => [...xs].sort((a, b) => a - b)
const median = (xs) => {
  const s = sorted(xs), m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
const percentile = (xs, p) => sorted(xs)[Math.min(xs.length - 1, Math.floor(xs.length * p))]

// The H.264 settings every pipeline's output is encoded with.
export function encoderConfig(width, height) {
  const bitrate = getBitrate(width, height, 30, 'good')
  return { codec: getCodecString('avc', width, height, bitrate), width, height, bitrate }
}

// mode 'transcode': the full decode -> process -> encode -> mux pipeline.
// mode 'process':   decode -> process -> wait for the output's pixels, one
//                   frame at a time — per-frame latency without the encoder.
export async function runOnce(pipeline, file, mode, { keepOutput = false } = {}) {
  const { config: decoderConfig, chunks } = await demux(file)
  const { codedWidth: width, codedHeight: height } = decoderConfig
  const latencies = []

  const start = performance.now()
  const processed = chunkStream(chunks)
    .pipeThrough(new VideoDecodeStream(decoderConfig))
    .pipeThrough(new VideoProcessStream(async (frame) => {
      const t = performance.now()
      const out = await pipeline.process(frame)
      if (mode === 'process') waitForPixels(out)
      latencies.push(performance.now() - t)
      return out
    }))

  let blob = null
  if (mode === 'transcode') {
    const muxer = new SimpleMuxer({ video: 'avc' })
    await processed.pipeThrough(new VideoEncodeStream(encoderConfig(width, height))).pipeTo(muxer.videoSink())
    blob = await muxer.finalize()
  } else {
    await processed.pipeTo(new WritableStream({ write: (frame) => frame.close() }))
  }
  const ms = performance.now() - start
  return {
    frames: latencies.length,
    ms,
    fps: (latencies.length * 1000) / ms,
    // Time spent inside process() per frame. For pipelines that only submit
    // GPU work this is CPU-side cost, not GPU time.
    meanProcessMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    p95ProcessMs: percentile(latencies, 0.95),
    ...(keepOutput && { blob }),
  }
}

const summarize = (runs) => {
  const fps = runs.map((r) => r.fps)
  return { medianFps: median(fps), minFps: Math.min(...fps), maxFps: Math.max(...fps) }
}

function shuffle(xs) {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// Benchmark several pipelines on one file.
//
// order 'sequential':  each pipeline in turn: create, warm up, `runs` timed
//                      runs, destroy. (Used for the published M4 numbers.)
// order 'interleaved': create and warm up all pipelines, then `runs` rounds;
//                      each round runs every pipeline once, in a fresh random
//                      order — spreads thermal drift and background load
//                      across pipelines instead of penalising whoever runs last.
export async function benchmarkAll(names, file, { mode = 'transcode', runs = 5, order = 'sequential', onProgress, onResult } = {}) {
  const results = new Map(names.map((name) => [name, { name, label: PIPELINES[name].label, runs: [] }]))
  const create = async (name) => {
    const t = performance.now()
    const pipeline = await PIPELINES[name].create()
    results.get(name).initMs = performance.now() - t
    onProgress?.(`${name}: warming up…`)
    results.get(name).warmupFps = (await runOnce(pipeline, file, mode)).fps
    return pipeline
  }
  const fail = (name, e) => {
    console.error(e)
    results.get(name).error = String(e?.message ?? e)
  }
  const finish = (name, blob) => {
    const r = results.get(name)
    if (!r.error) Object.assign(r, summarize(r.runs))
    onResult?.(r, blob)
  }

  if (order === 'sequential') {
    for (const name of names) {
      let blob = null
      try {
        const pipeline = await create(name)
        for (let i = 0; i < runs; i++) {
          onProgress?.(`${name}: run ${i + 1}/${runs}`)
          const { blob: b, ...run } = await runOnce(pipeline, file, mode, { keepOutput: i === runs - 1 })
          results.get(name).runs.push(run)
          blob = b ?? null
        }
        pipeline.destroy?.()
      } catch (e) {
        fail(name, e)
      }
      finish(name, blob)
    }
  } else {
    const pipelines = new Map()
    for (const name of names) {
      try { pipelines.set(name, await create(name)) } catch (e) { fail(name, e) }
    }
    const blobs = new Map()
    for (let round = 0; round < runs; round++) {
      for (const name of shuffle([...pipelines.keys()])) {
        if (results.get(name).error) continue
        onProgress?.(`round ${round + 1}/${runs}: ${name}`)
        try {
          const { blob, ...run } = await runOnce(pipelines.get(name), file, mode, { keepOutput: round === runs - 1 })
          results.get(name).runs.push({ ...run, round })
          if (blob) blobs.set(name, blob)
        } catch (e) {
          fail(name, e)
        }
      }
    }
    for (const [name, pipeline] of pipelines) pipeline.destroy?.()
    for (const name of names) finish(name, blobs.get(name) ?? null)
  }

  const { config } = await demux(file)
  return {
    config: {
      mode, runs, order, file: file.name, frames: [...results.values()].find((r) => r.runs.length)?.runs[0].frames,
      width: config.codedWidth, height: config.codedHeight, decoder: config.codec,
      ...(mode === 'transcode' && { encoder: encoderConfig(config.codedWidth, config.codedHeight) }),
    },
    environment: await environment(),
    results: [...results.values()],
  }
}

// Everything about the machine a reader would need to compare results.
export async function environment() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null)
  const ua = await navigator.userAgentData?.getHighEntropyValues?.(['platform', 'platformVersion', 'architecture', 'model', 'fullVersionList']).catch(() => null)
  return {
    date: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: ua && { platform: ua.platform, platformVersion: ua.platformVersion, architecture: ua.architecture, model: ua.model, browsers: ua.fullVersionList },
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGB: navigator.deviceMemory,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: adapter && {
      vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device,
      description: adapter.info.description, features: [...adapter.features].sort(),
    },
    webnn: 'ml' in navigator,
  }
}

// Run each pipeline on one frame and summarise its output, to catch broken
// masks (flipped, inverted, empty) before trusting the timings.
export async function checkOutputs(names, file, frameIndex = 96) {
  const { config, chunks } = await demux(file)
  let target = null, i = 0
  await chunkStream(chunks).pipeThrough(new VideoDecodeStream(config)).pipeTo(new WritableStream({
    write(frame) { if (i++ === frameIndex) target = frame; else frame.close() },
  }))

  const out = {}
  for (const name of names) {
    try {
      const pipeline = await PIPELINES[name].create()
      const frame = await pipeline.process(target)
      const bitmap = await createImageBitmap(frame)
      if (frame !== target) frame.close()
      pipeline.destroy?.()
      const canvas = new OffscreenCanvas(bitmap.width / 4, bitmap.height / 4)
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
      out[name] = { bitmap, pixels: ctx.getImageData(0, 0, canvas.width, canvas.height).data }
    } catch (e) {
      out[name] = { error: String(e?.message ?? e) }
    }
  }
  target.close()

  // Background share = pixels that are (nearly) the green fill color.
  const isGreen = (d, j) => Math.abs(d[j]) < 12 && Math.abs(d[j + 1] - 177) < 12 && Math.abs(d[j + 2] - 64) < 12
  const ref = out.webgpu?.pixels
  for (const r of Object.values(out)) {
    if (r.error) continue
    let green = 0, agree = 0, n = r.pixels.length / 4
    for (let j = 0; j < r.pixels.length; j += 4) {
      const g = isGreen(r.pixels, j)
      green += g
      if (ref) agree += g === isGreen(ref, j)
    }
    r.background = green / n
    r.agreement = ref ? agree / n : null
  }
  return out
}
