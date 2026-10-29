// Draws a model input tensor (a GPU storage buffer of RGB floats) to a canvas,
// without reading it back: a fragment shader indexes the buffer directly. This
// is how the demo shows "what the model sees" at no cost to the pipeline.

const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> tensor: array<f32>;  // [size][size][3]
@group(0) @binding(1) var<uniform> size: u32;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));  // one triangle covering the screen
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let p = vec2u(position.xy);
  let i = (p.y * size + p.x) * 3u;
  return vec4f(tensor[i], tensor[i + 1u], tensor[i + 2u], 1.0);
}
`

export class TensorView {
  constructor(device, canvas, buffer, size) {
    canvas.width = canvas.height = size
    const format = navigator.gpu.getPreferredCanvasFormat()
    this.device = device
    this.context = canvas.getContext('webgpu')
    this.context.configure({ device, format, alphaMode: 'opaque' })
    const module = device.createShaderModule({ code: SHADER })
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    })
    const uniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(uniform, 0, new Uint32Array([size]))
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }, { binding: 1, resource: { buffer: uniform } }],
    })
  }

  // empty: just clear the canvas (nothing worth showing).
  draw({ empty = false } = {}) {
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' }],
    })
    if (!empty) {
      pass.setPipeline(this.pipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.draw(3)
    }
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }
}
