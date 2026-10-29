// VideoFrame -> model input tensor, entirely on the GPU.
//
// Both models want a small square RGB float image:
//   - the palm detector: the whole frame, resized to 192×192 with black bars
//     (letterboxed) so the hand isn't stretched;
//   - the landmark model: a rotated square around the hand, resized to 224×224.
//
// Both are the same operation. For each pixel of the output tensor, an affine
// transform says where it comes from in the video frame; the shader samples the
// frame there and writes three floats into a storage buffer. That buffer *is*
// the model's input tensor: LiteRT reads it in place (see recognizer.js), so the
// frame never visits the CPU and is never copied into a canvas.
//
// importExternalTexture() wraps the VideoFrame's own GPU memory (no copy on
// most platforms) and handles its colour space and any YUV -> RGB conversion.

const SHADER = /* wgsl */ `
struct Params {
  row0: vec4f,       // frame.x = dot(row0.xyz, (u, v, 1)), u,v = tensor pixel centre
  row1: vec4f,       // frame.y = dot(row1.xyz, (u, v, 1))
  frameSize: vec2f,  // frame size in pixels
  size: u32,         // tensor width = height
  zeroBorder: u32,   // 1: outside the frame is black (letterbox), 0: clamp to edge
}

@group(0) @binding(0) var frame: texture_external;
@group(0) @binding(1) var linear: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> tensor: array<f32>;  // [size][size][3], RGB in [0, 1]

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.size || id.y >= params.size) { return; }
  let uv = vec3f(vec2f(id.xy) + 0.5, 1.0);
  let st = vec2f(dot(params.row0.xyz, uv), dot(params.row1.xyz, uv)) / params.frameSize;
  var rgb = textureSampleBaseClampToEdge(frame, linear, st).rgb;
  if (params.zeroBorder == 1u && (any(st < vec2f(0.0)) || any(st > vec2f(1.0)))) {
    rgb = vec3f(0.0);
  }
  let i = (id.y * params.size + id.x) * 3u;
  tensor[i] = rgb.r;
  tensor[i + 1u] = rgb.g;
  tensor[i + 2u] = rgb.b;
}
`

export class FrameWarp {
  constructor(device, size, { zeroBorder }) {
    this.device = device
    this.size = size
    this.zeroBorder = zeroBorder
    // The model input tensor. LiteRT wraps it as-is: 1×size×size×3 float32, row-major (NHWC).
    this.buffer = device.createBuffer({
      size: size * size * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    this.uniforms = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: 'main' },
    })
  }

  // matrix: [a, b, c, d, e, f] mapping tensor pixel (u, v) to frame pixel
  // (a·u + b·v + c, d·u + e·v + f). Returns once the work is queued; anything
  // queued after it (LiteRT's inference) sees the result.
  run(frame, matrix) {
    const [a, b, c, d, e, f] = matrix
    const params = new ArrayBuffer(48)
    new Float32Array(params, 0, 10).set([a, b, c, 0, d, e, f, 0, frame.displayWidth, frame.displayHeight])
    new Uint32Array(params, 40, 2).set([this.size, this.zeroBorder ? 1 : 0])
    this.device.queue.writeBuffer(this.uniforms, 0, params)

    // An external texture is only valid briefly (until the frame is closed, or
    // the end of the task for a <video>), so it gets a fresh bind group each time.
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.device.importExternalTexture({ source: frame }) },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniforms } },
        { binding: 3, resource: { buffer: this.buffer } },
      ],
    })
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(this.size / 8), Math.ceil(this.size / 8))
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }
}

// Fit a width×height frame inside a size×size square, centred, keeping its
// aspect ratio. Returns the warp matrix and the padding on each side as a
// fraction of the square, which is needed to map detections back to the frame.
export function letterbox(width, height, size) {
  const scale = size / Math.max(width, height) // tensor pixels per frame pixel
  const padX = (size - width * scale) / 2
  const padY = (size - height * scale) / 2
  return {
    matrix: [1 / scale, 0, -padX / scale, 0, 1 / scale, -padY / scale],
    padding: { x: padX / size, y: padY / size },
  }
}

// The rotated crop: rect is { xCenter, yCenter, width, height, rotation } in
// frame-normalised coordinates (MediaPipe's NormalizedRect). Tensor pixel
// (u, v) maps to the point at offset (u/size − ½, v/size − ½) of the rect,
// rotated by rect.rotation about the rect's centre.
export function rectToMatrix(rect, frameWidth, frameHeight, size) {
  const cos = Math.cos(rect.rotation), sin = Math.sin(rect.rotation)
  const w = rect.width * frameWidth, h = rect.height * frameHeight // rect size in frame pixels
  const cx = rect.xCenter * frameWidth, cy = rect.yCenter * frameHeight
  return [
    (cos * w) / size, (-sin * h) / size, cx - 0.5 * (cos * w - sin * h),
    (sin * w) / size, (cos * h) / size, cy - 0.5 * (sin * w + cos * h),
  ]
}
