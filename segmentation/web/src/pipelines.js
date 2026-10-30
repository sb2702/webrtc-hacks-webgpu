// Background replacement pipelines, all with the same shape:
//
//   const p = await PIPELINES[name].create()
//   const out = await p.process(frame)   // VideoFrame in, new VideoFrame out
//
// so any of them drops into a VideoProcessStream (webcodecs-utils) or a
// MediaStreamTrackProcessor -> Generator chain. The input frame stays owned by
// the caller; the returned frame is the caller's to close.

import { loadWeights } from './selfie/weights.js'
import { buildSelfie } from './selfie/model.js'
import { WebGPUBackend } from './selfie/webgpu.js'
import { WebGLBackend } from './selfie/webgl.js'
import { WebNNBackend } from './selfie/webnn.js'
import { WebGPUFrames } from './frames/webgpu.js'
import { WebGLFrames } from './frames/webgl.js'

const MODEL_URL = '/models/selfie_segmenter.tflite'
const SIZE = 256

let devicePromise, weightsPromise
const getWeights = () => (weightsPromise ??= loadWeights('/models/selfie_segmenter'))

// One GPUDevice per page, shared by our network, the frame I/O and LiteRT.
export function getDevice() {
  return (devicePromise ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('WebGPU is not available')
    const device = await adapter.requestDevice()
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message))
    return device
  })())
}

// ---- baseline ---------------------------------------------------------------

async function passthrough() {
  return { process: (frame) => frame }
}

// ---- ours: the whole frame -> mask -> frame path in one WebGPU submit -------

async function oursWebGPU() {
  const device = await getDevice()
  const net = await WebGPUBackend.create(device)
  const { input, mask } = buildSelfie(net, await getWeights())
  const frames = new WebGPUFrames(device)
  return {
    frames,
    process(frame) {
      const src = frames.import(frame)
      const encoder = device.createCommandEncoder()
      const pass = encoder.beginComputePass()
      frames.preprocess(pass, src, input.buffer, { width: SIZE, height: SIZE, stride: 4 })
      net.encode(pass)
      pass.end()
      frames.composite(encoder, src, mask.buffer, { width: SIZE, height: SIZE, stride: 4 })
      device.queue.submit([encoder.finish()])
      return frames.output(frame)
    },
  }
}

// ---- ours: WebGL2, one draw call per layer ----------------------------------

async function oursWebGL() {
  const canvas = new OffscreenCanvas(1, 1)
  const net = await WebGLBackend.create(canvas)
  const { input, mask } = buildSelfie(net, await getWeights())
  const frames = new WebGLFrames(net, canvas)
  return {
    frames,
    process(frame) {
      frames.upload(frame)
      frames.preprocess(input)
      net.run()
      frames.composite(mask.texture, { width: SIZE, height: SIZE })
      return frames.output(frame)
    },
  }
}

// ---- WebGPU pre/post-processing around a model that runs somewhere else ------
//
// The frame is resized into a dense [256, 256, 3] f32 buffer and the mask comes
// back as [256, 256, 1]. If the model runtime can't take GPU buffers, the input
// is read back and the mask uploaded — the round trip the article is about.

function gpuIO(device) {
  const frames = new WebGPUFrames(device)
  const bytes = SIZE * SIZE * 3 * 4
  const inputBuffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const maskBuffer = device.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })

  const encodePreprocess = (frame, dst) => {
    const src = frames.import(frame)
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    frames.preprocess(pass, src, dst, { width: SIZE, height: SIZE, stride: 3 })
    pass.end()
    return { src, encoder }
  }

  return {
    frames,
    inputBuffer,
    // Resize the frame into `dst` on the GPU. Returns the imported frame for composite().
    preprocess(frame, dst = inputBuffer) {
      const { src, encoder } = encodePreprocess(frame, dst)
      device.queue.submit([encoder.finish()])
      return src
    },
    // Same, then read the tensor back to the CPU.
    async preprocessToCPU(frame) {
      const { src, encoder } = encodePreprocess(frame, inputBuffer)
      encoder.copyBufferToBuffer(inputBuffer, 0, staging, 0, bytes)
      device.queue.submit([encoder.finish()])
      await staging.mapAsync(GPUMapMode.READ)
      const data = new Float32Array(staging.getMappedRange()).slice()
      staging.unmap()
      return { src, data }
    },
    uploadMask(data) {
      device.queue.writeBuffer(maskBuffer, 0, data)
    },
    composite(src, mask = maskBuffer) {
      const encoder = device.createCommandEncoder()
      frames.composite(encoder, src, mask, { width: SIZE, height: SIZE, stride: 1 })
      device.queue.submit([encoder.finish()])
    },
  }
}

// ---- LiteRT.js (the official TFLite web runtime) with WebGPU pre/post ---------

let liteRtPromise
async function getLiteRt() {
  return (liteRtPromise ??= (async () => {
    const litert = await import('@litertjs/core')
    await litert.loadLiteRt('/vendor/litert/', { threads: self.crossOriginIsolated })
    litert.setWebGpuDevice(await getDevice())
    return litert
  })())
}

// accelerator 'webgpu': input and output stay in GPU buffers we own / LiteRT
// hands back — no readback. accelerator 'wasm': XNNPACK on the CPU, so the
// preprocessed tensor is read back and the mask uploaded every frame.
async function liteRt(accelerator, precision) {
  const [litert, device] = await Promise.all([getLiteRt(), getDevice()])
  const model = await litert.loadAndCompile(MODEL_URL, { accelerator, ...(precision && { gpuOptions: { precision } }) })
  const io = gpuIO(device)

  if (accelerator === 'webgpu') {
    const inputTensor = new litert.Tensor(io.inputBuffer, [1, SIZE, SIZE, 3], 'float32')
    return {
      frames: io.frames,
      async process(frame) {
        const src = io.preprocess(frame)
        const [out] = await model.run([inputTensor])
        io.composite(src, out.toGpuBuffer())
        out.delete()
        return io.frames.output(frame)
      },
    }
  }

  return {
    frames: io.frames,
    async process(frame) {
      const { src, data } = await io.preprocessToCPU(frame)
      const input = new litert.Tensor(data, [1, SIZE, SIZE, 3])
      const [out] = await model.run([input])
      io.uploadMask(out.toTypedArray())
      input.delete()
      out.delete()
      io.composite(src)
      return io.frames.output(frame)
    },
  }
}

// ---- WebNN: our graph, compiled by the browser for the OS ML stack ---------------

// deviceType is a hint: 'cpu' | 'gpu' | 'npu'. Tensors go through the CPU.
async function webNN(deviceType, dataType) {
  const device = await getDevice()
  const net = await WebNNBackend.create({ deviceType }, { dataType })
  const { input, mask } = buildSelfie(net, await getWeights())
  await net.compile(input, mask)
  const io = gpuIO(device)
  return {
    frames: io.frames,
    async process(frame) {
      const { src, data } = await io.preprocessToCPU(frame)
      io.uploadMask(await net.run(data))
      io.composite(src)
      return io.frames.output(frame)
    },
  }
}

// WebNN context created from our GPUDevice, with tensors exported to WebGPU as
// GPUBuffers (MLContext.exportToGPU) — no CPU round trip. Exported buffers are
// only lent to WebGPU until WebNN uses the tensor again, so they're
// re-exported every frame.
async function webNNInterop() {
  const device = await getDevice()
  const net = await WebNNBackend.create(device)
  const { input, mask } = buildSelfie(net, await getWeights())
  await net.compile(input, mask, { gpuDevice: device })
  const io = gpuIO(device)
  return {
    frames: io.frames,
    async process(frame) {
      const src = io.preprocess(frame, await net.context.exportToGPU(net.inputTensor))
      net.dispatch()
      io.composite(src, await net.context.exportToGPU(net.maskTensor))
      return io.frames.output(frame)
    },
  }
}

// ---- MediaPipe Tasks (ImageSegmenter), using its own pre/post-processing ----

let filesetPromise
async function getVisionTasks() {
  return (filesetPromise ??= (async () => {
    const vision = await import('@mediapipe/tasks-vision')
    return { vision, fileset: await vision.FilesetResolver.forVisionTasks('/vendor/mediapipe') }
  })())
}

const segmenterOptions = (delegate, canvas) => ({
  baseOptions: { modelAssetPath: MODEL_URL, delegate },
  runningMode: 'VIDEO',
  outputConfidenceMasks: true,
  outputCategoryMask: false,
  ...(canvas && { canvas }),
})

// VIDEO mode wants strictly increasing timestamps; benchmarks replay the same
// clip, so count instead of using frame.timestamp.
const clock = () => { let t = 0; return () => ++t }

// GPU delegate = WebGL. MediaPipe renders into our canvas's WebGL2 context, so
// its mask texture can be composited directly, no readback.
async function mediaPipeGPU() {
  const { vision, fileset } = await getVisionTasks()
  const canvas = new OffscreenCanvas(1, 1)
  const segmenter = await vision.ImageSegmenter.createFromOptions(fileset, segmenterOptions('GPU', canvas))
  const gl = canvas.getContext('webgl2')
  const frames = new WebGLFrames(await WebGLBackend.create(gl), canvas)
  const now = clock()
  return {
    frames,
    process(frame) {
      const result = segmenter.segmentForVideo(frame, now())
      const mask = result.confidenceMasks.at(-1)
      frames.upload(frame)
      frames.composite(mask.getAsWebGLTexture(), { width: mask.width, height: mask.height })
      gl.bindVertexArray(null)
      const out = frames.output(frame)
      result.close()
      return out
    },
    destroy: () => segmenter.close(),
  }
}

// CPU delegate = WASM (XNNPACK). MediaPipe returns the mask as a Float32Array
// at the frame's resolution; we upload it and composite on WebGPU.
async function mediaPipeCPU() {
  const { vision, fileset } = await getVisionTasks()
  const segmenter = await vision.ImageSegmenter.createFromOptions(fileset, segmenterOptions('CPU'))
  const device = await getDevice()
  const frames = new WebGPUFrames(device)
  let maskBuffer
  const now = clock()
  return {
    frames,
    process(frame) {
      const result = segmenter.segmentForVideo(frame, now())
      const mask = result.confidenceMasks.at(-1)
      const data = mask.getAsFloat32Array()
      if (maskBuffer?.size !== data.byteLength) {
        maskBuffer?.destroy()
        maskBuffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
      }
      device.queue.writeBuffer(maskBuffer, 0, data)
      const src = frames.import(frame)
      const encoder = device.createCommandEncoder()
      frames.composite(encoder, src, maskBuffer, { width: mask.width, height: mask.height, stride: 1 })
      device.queue.submit([encoder.finish()])
      result.close()
      return frames.output(frame)
    },
    destroy: () => segmenter.close(),
  }
}

export const PIPELINES = {
  'passthrough': { label: 'Passthrough (no effect)', create: passthrough },
  'webgpu': { label: 'Ours — WebGPU', create: oursWebGPU },
  'webgl': { label: 'Ours — WebGL2', create: oursWebGL },
  'litert-webgpu': { label: 'LiteRT.js — WebGPU (fp32, default)', create: () => liteRt('webgpu') },
  'litert-webgpu-fp16': { label: 'LiteRT.js — WebGPU (fp16)', create: () => liteRt('webgpu', 'fp16') },
  'litert-wasm': { label: 'LiteRT.js — WASM (WebGPU I/O)', create: () => liteRt('wasm') },
  'webnn-cpu': { label: 'WebNN — CPU (fp32)', create: () => webNN('cpu') },
  'webnn-gpu': { label: 'WebNN — GPU (fp32)', create: () => webNN('gpu') },
  'webnn-npu': { label: 'WebNN — NPU (fp32)', create: () => webNN('npu') },
  'webnn-gpu-fp16': { label: 'WebNN — GPU (fp16)', create: () => webNN('gpu', 'float16') },
  'webnn-npu-fp16': { label: 'WebNN — NPU (fp16)', create: () => webNN('npu', 'float16') },
  // Off by default: fails on macOS Chrome 156 (exportable tensors are fp16-only and too small for this model).
  'webnn-interop': { label: 'WebNN — WebGPU interop (experimental)', create: webNNInterop, default: false },
  'mediapipe-gpu': { label: 'MediaPipe — GPU (WebGL)', create: mediaPipeGPU },
  'mediapipe-cpu': { label: 'MediaPipe — CPU (WASM)', create: mediaPipeCPU },
}
