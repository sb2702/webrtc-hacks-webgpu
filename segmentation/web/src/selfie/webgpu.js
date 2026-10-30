// WebGPU backend: every layer is a compute shader over storage buffers.
//
// Tensors are flat array<vec4f> buffers in NHWC order with channels padded to a
// multiple of 4: element (y, x, channel group g) lives at (y * w + x) * groups + g.
// Building the model records one {pipeline, bindGroup, workgroups} per layer;
// encode() replays them all inside a single compute pass. Getting frames in
// and out lives in src/frames/webgpu.js.

import { packConv, packDepthwise, packBias, convGeometry, groups } from './weights.js'

const ACT = { none: 0, relu: 1, hswish: 2, sigmoid: 3 }

// Activation is a pipeline-overridable constant, so each (shader, activation)
// pair compiles to its own branch-free pipeline.
const ACTIVATE = /* wgsl */ `
override ACT: u32 = 0u;
fn activate(v: vec4f) -> vec4f {
  if (ACT == 1u) { return max(v, vec4f(0.0)); }
  if (ACT == 2u) { return v * clamp(v + 3.0, vec4f(0.0), vec4f(6.0)) / 6.0; }
  if (ACT == 3u) { return 1.0 / (1.0 + exp(-v)); }
  return v;
}`

const SHADERS = {
  conv: /* wgsl */ `
struct P { in_w: u32, in_h: u32, in_g: u32, out_w: u32, out_h: u32, out_g: u32, k: u32, stride: u32, pad_y: u32, pad_x: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
@group(0) @binding(3) var<storage, read> weight: array<mat4x4f>;
@group(0) @binding(4) var<storage, read> bias: array<vec4f>;
${ACTIVATE}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = id.x; let y = id.y; let o = id.z;
  if (x >= p.out_w || y >= p.out_h || o >= p.out_g) { return; }
  var acc = bias[o];
  for (var ky = 0u; ky < p.k; ky++) {
    let iy = i32(y * p.stride + ky) - i32(p.pad_y);
    if (iy < 0 || iy >= i32(p.in_h)) { continue; }
    for (var kx = 0u; kx < p.k; kx++) {
      let ix = i32(x * p.stride + kx) - i32(p.pad_x);
      if (ix < 0 || ix >= i32(p.in_w)) { continue; }
      let s = (u32(iy) * p.in_w + u32(ix)) * p.in_g;
      let w = ((ky * p.k + kx) * p.out_g + o) * p.in_g;
      for (var i = 0u; i < p.in_g; i++) { acc += weight[w + i] * src[s + i]; }
    }
  }
  dst[(y * p.out_w + x) * p.out_g + o] = activate(acc);
}`,

  depthwise: /* wgsl */ `
struct P { in_w: u32, in_h: u32, g: u32, out_w: u32, out_h: u32, k: u32, stride: u32, pad_y: u32, pad_x: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
@group(0) @binding(3) var<storage, read> weight: array<vec4f>;
@group(0) @binding(4) var<storage, read> bias: array<vec4f>;
${ACTIVATE}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = id.x; let y = id.y; let c = id.z;
  if (x >= p.out_w || y >= p.out_h || c >= p.g) { return; }
  var acc = bias[c];
  for (var ky = 0u; ky < p.k; ky++) {
    let iy = i32(y * p.stride + ky) - i32(p.pad_y);
    if (iy < 0 || iy >= i32(p.in_h)) { continue; }
    for (var kx = 0u; kx < p.k; kx++) {
      let ix = i32(x * p.stride + kx) - i32(p.pad_x);
      if (ix < 0 || ix >= i32(p.in_w)) { continue; }
      acc += weight[(ky * p.k + kx) * p.g + c] * src[(u32(iy) * p.in_w + u32(ix)) * p.g + c];
    }
  }
  dst[(y * p.out_w + x) * p.g + c] = activate(acc);
}`,

  // Output pixel (X, Y) comes from input pixel (X/2, Y/2) through tap (Y%2, X%2).
  convTranspose2x2: /* wgsl */ `
struct P { in_w: u32, in_h: u32, in_g: u32, out_g: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
@group(0) @binding(3) var<storage, read> weight: array<mat4x4f>;
@group(0) @binding(4) var<storage, read> bias: array<vec4f>;
${ACTIVATE}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = id.x; let y = id.y; let o = id.z;
  if (x >= p.in_w * 2u || y >= p.in_h * 2u || o >= p.out_g) { return; }
  var acc = bias[o];
  let s = ((y / 2u) * p.in_w + x / 2u) * p.in_g;
  let w = (((y % 2u) * 2u + x % 2u) * p.out_g + o) * p.in_g;
  for (var i = 0u; i < p.in_g; i++) { acc += weight[w + i] * src[s + i]; }
  dst[(y * p.in_w * 2u + x) * p.out_g + o] = activate(acc);
}`,

  // Global average pool: one workgroup per channel group. Each of the 256
  // threads sums a strided slice of the plane, then the partial sums are
  // tree-reduced in workgroup shared memory. (A single thread walking a 128x128
  // plane was 80% of the network's GPU time.)
  pool: /* wgsl */ `
struct P { n: u32, g: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
var<workgroup> partial: array<vec4f, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3u) {
  let c = wg.x;
  var sum = vec4f(0.0);
  for (var i = t; i < p.n; i += 256u) { sum += src[i * p.g + c]; }
  partial[t] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride) { partial[t] += partial[t + stride]; }
    workgroupBarrier();
  }
  if (t == 0u) { dst[c] = partial[0] / f32(p.n); }
}`,

  upsample2x: /* wgsl */ `
struct P { in_w: u32, in_h: u32, g: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
fn at(x: u32, y: u32, c: u32) -> vec4f { return src[(y * p.in_w + x) * p.g + c]; }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = id.x; let y = id.y; let c = id.z;
  if (x >= p.in_w * 2u || y >= p.in_h * 2u || c >= p.g) { return; }
  let s = clamp((vec2f(f32(x), f32(y)) + 0.5) * 0.5 - 0.5, vec2f(0.0), vec2f(f32(p.in_w - 1u), f32(p.in_h - 1u)));
  let p0 = vec2u(floor(s));
  let p1 = min(p0 + 1u, vec2u(p.in_w - 1u, p.in_h - 1u));
  let f = s - floor(s);
  let top = mix(at(p0.x, p0.y, c), at(p1.x, p0.y, c), f.x);
  let bottom = mix(at(p0.x, p1.y, c), at(p1.x, p1.y, c), f.x);
  dst[(y * p.in_w * 2u + x) * p.g + c] = mix(top, bottom, f.y);
}`,

  add: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> a: array<vec4f>;
@group(0) @binding(3) var<storage, read> b: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.n) { return; }
  dst[id.x] = a[id.x] + b[id.x];
}`,

  mul: /* wgsl */ `
struct P { n: u32, g: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read> x: array<vec4f>;
@group(0) @binding(3) var<storage, read> scale: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.n) { return; }
  dst[id.x] = x[id.x] * scale[id.x % p.g];
}`,
}

export class WebGPUBackend {
  static async create(device) {
    if (!device) {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) throw new Error('WebGPU is not available')
      device = await adapter.requestDevice()
      device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message))
    }
    return new WebGPUBackend(device)
  }

  constructor(device) {
    this.name = 'webgpu'
    this.device = device
    this.ops = []
    this.pipelines = new Map()
  }

  // ---- building blocks ------------------------------------------------------

  tensor(h, w, c) {
    const g = groups(c)
    const buffer = this.device.createBuffer({
      size: h * w * g * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    return { h, w, c: g * 4, g, buffer }
  }

  storage(data) {
    const buffer = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true })
    new Float32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    return buffer
  }

  uniform(values) {
    const data = new Uint32Array(Math.ceil(values.length / 4) * 4)
    data.set(values)
    const buffer = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true })
    new Uint32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    return buffer
  }

  pipeline(name, act) {
    const key = `${name}:${act}`
    if (!this.pipelines.has(key)) {
      const module = this.device.createShaderModule({ code: SHADERS[name] })
      const constants = act === undefined ? undefined : { ACT: ACT[act] }
      this.pipelines.set(key, this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main', constants } }))
    }
    return this.pipelines.get(key)
  }

  // Record one dispatch. Bindings: 0 = params, 1 = output, 2.. = inputs.
  op(name, { act, params, out, inputs, workgroups }) {
    const pipeline = this.pipeline(name, act)
    const entries = [this.uniform(params), out.buffer, ...inputs.map((t) => t.buffer ?? t)]
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
    })
    this.ops.push({ pipeline, bindGroup, workgroups })
    return out
  }

  // ---- model ops (see model.js) ---------------------------------------------

  input(h, w) {
    return this.tensor(h, w, 3)
  }

  conv(x, layer, act) {
    const [O, k, , I] = layer.shape
    if (groups(I) !== x.g) throw new Error(`${layer.name}: expected ${I} input channels, got ${x.c}`)
    const s = layer.stride[0]
    const gy = convGeometry(x.h, k, s, layer.pad), gx = convGeometry(x.w, k, s, layer.pad)
    const out = this.tensor(gy.out, gx.out, O)
    return this.op('conv', {
      act, out,
      params: [x.w, x.h, x.g, out.w, out.h, out.g, k, s, gy.pad, gx.pad],
      inputs: [x, this.storage(packConv(layer)), this.storage(packBias(layer))],
      workgroups: [Math.ceil(out.w / 8), Math.ceil(out.h / 8), out.g],
    })
  }

  depthwise(x, layer, act) {
    const k = layer.shape[1], s = layer.stride[0]
    const gy = convGeometry(x.h, k, s, layer.pad), gx = convGeometry(x.w, k, s, layer.pad)
    const out = this.tensor(gy.out, gx.out, x.c)
    return this.op('depthwise', {
      act, out,
      params: [x.w, x.h, x.g, out.w, out.h, k, s, gy.pad, gx.pad],
      inputs: [x, this.storage(packDepthwise(layer)), this.storage(packBias(layer))],
      workgroups: [Math.ceil(out.w / 8), Math.ceil(out.h / 8), x.g],
    })
  }

  convTranspose2x2(x, layer, act) {
    const out = this.tensor(x.h * 2, x.w * 2, layer.shape[0])
    return this.op('convTranspose2x2', {
      act, out,
      params: [x.w, x.h, x.g, out.g],
      inputs: [x, this.storage(packConv(layer)), this.storage(packBias(layer))],
      workgroups: [Math.ceil(out.w / 8), Math.ceil(out.h / 8), out.g],
    })
  }

  pool(x) {
    return this.op('pool', {
      out: this.tensor(1, 1, x.c), params: [x.h * x.w, x.g], inputs: [x],
      workgroups: [x.g],
    })
  }

  upsample2x(x) {
    const out = this.tensor(x.h * 2, x.w * 2, x.c)
    return this.op('upsample2x', {
      out, params: [x.w, x.h, x.g], inputs: [x],
      workgroups: [Math.ceil(out.w / 8), Math.ceil(out.h / 8), x.g],
    })
  }

  add(a, b) {
    const n = a.h * a.w * a.g
    return this.op('add', {
      out: this.tensor(a.h, a.w, a.c), params: [n], inputs: [a, b], workgroups: [Math.ceil(n / 64)],
    })
  }

  mul(x, scale) {
    const n = x.h * x.w * x.g
    return this.op('mul', {
      out: this.tensor(x.h, x.w, x.c), params: [n, x.g], inputs: [x, scale], workgroups: [Math.ceil(n / 64)],
    })
  }

  // ---- running ---------------------------------------------------------------

  // Append the whole network to a compute pass.
  encode(pass) {
    for (const { pipeline, bindGroup, workgroups } of this.ops) {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(...workgroups)
    }
  }

  // Run just the network on whatever is in the input tensor (for tests).
  infer() {
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    this.encode(pass)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  // Resolves when the GPU has finished all submitted work.
  sync() {
    return this.device.queue.onSubmittedWorkDone()
  }

  // ---- tensor I/O (tests / debugging) -----------------------------------------

  write(t, data, channels) {
    const packed = new Float32Array(t.h * t.w * t.c)
    for (let p = 0; p < t.h * t.w; p++)
      for (let c = 0; c < channels; c++) packed[p * t.c + c] = data[p * channels + c]
    this.device.queue.writeBuffer(t.buffer, 0, packed)
  }

  async read(t, channels = t.c) {
    const staging = this.device.createBuffer({ size: t.buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const encoder = this.device.createCommandEncoder()
    encoder.copyBufferToBuffer(t.buffer, 0, staging, 0, t.buffer.size)
    this.device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const packed = new Float32Array(staging.getMappedRange())
    const out = new Float32Array(t.h * t.w * channels)
    for (let p = 0; p < t.h * t.w; p++)
      for (let c = 0; c < channels; c++) out[p * channels + c] = packed[p * t.c + c]
    staging.destroy()
    return out
  }

  get dispatches() {
    return this.ops.length
  }
}
