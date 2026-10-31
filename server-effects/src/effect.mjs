// The background-replacement effect, server-side. The network and both frame
// shaders are imported from the browser code in ../segmentation unchanged;
// only how a frame gets onto and off the GPU differs:
//
//   browser:  VideoFrame ─ importExternalTexture ─▶ GPU ─▶ canvas ─▶ new VideoFrame(canvas)   (no CPU copies)
//   server:   RGBA bytes ─ queue.writeTexture ────▶ GPU ─▶ texture ─ copyTextureToBuffer + mapAsync ─▶ RGBA bytes
//
// Per frame: one upload, one submit (resize → 40-odd network layers →
// composite → copy out), one readback.
import { readFileSync } from 'node:fs'
import { parseWeights } from '../../segmentation/web/src/selfie/weights.js'
import { buildSelfie } from '../../segmentation/web/src/selfie/model.js'
import { WebGPUBackend } from '../../segmentation/web/src/selfie/webgpu.js'
import { preprocessShader, compositeShader } from '../../segmentation/web/src/frames/webgpu.js'

const MODEL = new URL('../../segmentation/web/public/models/selfie_segmenter', import.meta.url).pathname
const SIZE = 256 // model input / mask size
const GREEN = [0, 177 / 255, 64 / 255, 1]

export class ServerEffect {
  static async create(device, { width, height, color = GREEN, inFlight = 3, passthrough = false }) {
    const bin = readFileSync(`${MODEL}.bin`)
    const weights = parseWeights(JSON.parse(readFileSync(`${MODEL}.json`, 'utf8')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length))
    const net = await WebGPUBackend.create(device)
    const { input, mask } = buildSelfie(net, weights)
    return new ServerEffect(device, net, input, mask, { width, height, color, inFlight, passthrough })
  }

  constructor(device, net, input, mask, { width, height, color, inFlight, passthrough }) {
    Object.assign(this, { device, net, input, mask, width, height, passthrough })

    // Frame in (written from the CPU) and frame out (rendered by the composite pass).
    this.frameIn = device.createTexture({
      size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    })
    this.frameOut = device.createTexture({
      size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    })
    // copyTextureToBuffer needs rows padded to 256 bytes.
    this.bytesPerRow = Math.ceil((width * 4) / 256) * 256

    // One staging buffer per frame in flight, so frame N+1 can be uploaded and
    // processed while frame N is still being read back.
    this.staging = Array.from({ length: inFlight }, () =>
      device.createBuffer({ size: this.bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }))
    this.next = 0

    // The same WGSL as the browser, in its texture_2d variant.
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    const preprocess = device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code: preprocessShader('texture') }), entryPoint: 'main' },
    })
    const compositeModule = device.createShaderModule({ code: compositeShader('texture') })
    const composite = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    })
    const uniform = (data) => {
      const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true })
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer ?? data))
      buffer.unmap()
      return buffer
    }
    const compositeParams = new ArrayBuffer(48)
    new Float32Array(compositeParams, 0, 6).set([...color, width, height])
    new Uint32Array(compositeParams, 24, 3).set([SIZE, SIZE, 4])

    // Unlike the browser (where an external texture is per-frame), every
    // binding here is fixed, so the bind groups are made once.
    const frameView = this.frameIn.createView()
    this.preprocess = {
      pipeline: preprocess,
      bindGroup: device.createBindGroup({
        layout: preprocess.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform(new Uint32Array([SIZE, SIZE, 4, 0])) } },
          { binding: 1, resource: { buffer: input.buffer } },
          { binding: 2, resource: frameView },
          { binding: 3, resource: sampler },
        ],
      }),
    }
    this.composite = {
      pipeline: composite,
      bindGroup: device.createBindGroup({
        layout: composite.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform(compositeParams) } },
          { binding: 1, resource: { buffer: mask.buffer } },
          { binding: 2, resource: frameView },
          { binding: 3, resource: sampler },
        ],
      }),
    }
  }

  // RGBA bytes in; resolves to RGBA bytes out. Call it again without waiting
  // to keep several frames in flight (up to `inFlight`); results resolve in order.
  async process(rgba) {
    const { device, width, height } = this
    const staging = this.staging[this.next++ % this.staging.length]
    if (staging.mapState !== 'unmapped') throw new Error('too many frames in flight')

    // CPU → GPU. Queue operations run in order, so this can't overwrite the
    // texture while an earlier frame's commands still need it.
    device.queue.writeTexture({ texture: this.frameIn }, rgba, { bytesPerRow: width * 4 }, [width, height])

    const encoder = device.createCommandEncoder()
    if (this.passthrough) {
      encoder.copyTextureToTexture({ texture: this.frameIn }, { texture: this.frameOut }, [width, height])
    } else {
      const pass = encoder.beginComputePass()
      pass.setPipeline(this.preprocess.pipeline)
      pass.setBindGroup(0, this.preprocess.bindGroup)
      pass.dispatchWorkgroups(SIZE / 8, SIZE / 8)
      this.net.encode(pass) // the whole selfie network, same code as the browser
      pass.end()
      const render = encoder.beginRenderPass({
        colorAttachments: [{ view: this.frameOut.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      })
      render.setPipeline(this.composite.pipeline)
      render.setBindGroup(0, this.composite.bindGroup)
      render.draw(3)
      render.end()
    }
    // GPU → CPU.
    encoder.copyTextureToBuffer({ texture: this.frameOut }, { buffer: staging, bytesPerRow: this.bytesPerRow }, [width, height])
    device.queue.submit([encoder.finish()])

    await staging.mapAsync(GPUMapMode.READ)
    const padded = new Uint8Array(staging.getMappedRange())
    const out = Buffer.allocUnsafe(width * height * 4)
    if (this.bytesPerRow === width * 4) out.set(padded)
    else for (let y = 0; y < height; y++) out.set(padded.subarray(y * this.bytesPerRow, y * this.bytesPerRow + width * 4), y * width * 4)
    staging.unmap()
    return out
  }
}
