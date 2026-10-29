// One GPUDevice per page, shared by our shaders and by LiteRT, so tensors can
// pass between them as plain GPUBuffers.
let devicePromise

export function getDevice() {
  return (devicePromise ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('WebGPU is not available in this browser')
    const device = await adapter.requestDevice()
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message))
    return device
  })())
}

// Reads several (small) GPU buffers back to the CPU with a single map: copy
// them all into one staging buffer, submit, await once. Each separate
// mapAsync is a round trip to the GPU process, so batching matters more than
// the byte count when the data is a few hundred bytes.
export class Readback {
  constructor(device) {
    this.device = device
    this.staging = null
  }

  // parts: [{ buffer: GPUBuffer, bytes: number }] -> [Float32Array, ...]
  async read(parts, encoder = this.device.createCommandEncoder()) {
    const total = parts.reduce((sum, p) => sum + p.bytes, 0)
    if (!this.staging || this.staging.size < total) {
      this.staging?.destroy()
      this.staging = this.device.createBuffer({ size: total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    }
    let offset = 0
    for (const { buffer, bytes } of parts) {
      encoder.copyBufferToBuffer(buffer, 0, this.staging, offset, bytes)
      offset += bytes
    }
    this.device.queue.submit([encoder.finish()])
    await this.staging.mapAsync(GPUMapMode.READ, 0, total)
    const data = new Float32Array(this.staging.getMappedRange(0, total).slice(0))
    this.staging.unmap()
    offset = 0
    return parts.map(({ bytes }) => {
      const part = data.subarray(offset / 4, (offset + bytes) / 4)
      offset += bytes
      return part
    })
  }
}
