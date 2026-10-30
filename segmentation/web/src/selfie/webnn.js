// WebNN backend: the same model.js graph, but instead of writing shaders we
// describe each layer to MLGraphBuilder and let the browser compile the whole
// graph for the OS ML stack (Core ML on macOS, DirectML / OpenVINO on
// Windows, TFLite on ChromeOS / Android), which can target CPU, GPU or NPU.
//
// No vec4 packing here: tensors are plain NHWC with their real channel
// counts, and weights go in exactly as TFLite stores them.

import { convGeometry } from './weights.js'

const tensorDesc = (shape, extra = {}, dataType = 'float32') => ({ dataType, shape, dimensions: shape, ...extra })

// Explicit [top, bottom, left, right] padding for TF 'SAME' / 'VALID'.
function padding(h, w, k, stride, pad) {
  const gy = convGeometry(h, k, stride, pad), gx = convGeometry(w, k, stride, pad)
  const extra = (size, out) => Math.max((out - 1) * stride + k - size, 0)
  return {
    h: gy.out, w: gx.out,
    padding: pad === 'VALID' ? [0, 0, 0, 0] : [gy.pad, extra(h, gy.out) - gy.pad, gx.pad, extra(w, gx.out) - gx.pad],
  }
}

export class WebNNBackend {
  // `options` is passed to navigator.ml.createContext: a { deviceType } hint,
  // or a GPUDevice for a context that can share buffers with WebGPU.
  // `dataType: 'float16'` builds the graph in half precision (inputs and
  // outputs stay float32, cast at the edges) — Apple's Neural Engine only
  // runs fp16.
  static async create(options, { dataType = 'float32' } = {}) {
    if (!('ml' in navigator)) throw new Error('WebNN is not available')
    return new WebNNBackend(await navigator.ml.createContext(options), dataType)
  }

  constructor(context, dataType = 'float32') {
    this.name = 'webnn'
    this.context = context
    this.dataType = dataType
    this.builder = new MLGraphBuilder(context)
  }

  // Operands carry their NHWC size so later layers can compute padding.
  tensor(op, h, w, c) {
    return { op, h, w, c }
  }

  constant(shape, data) {
    const half = this.dataType === 'float16'
    return this.builder.constant(tensorDesc(shape, {}, this.dataType), half ? new Float16Array(data) : data.slice())
  }

  activate(op, act) {
    if (act === 'relu') return this.builder.relu(op)
    if (act === 'hswish') return this.builder.hardSwish(op)
    if (act === 'sigmoid') return this.builder.sigmoid(op)
    return op
  }

  // ---- model ops (see model.js) ---------------------------------------------

  input(h, w) {
    const op = this.builder.input('input', tensorDesc([1, h, w, 3]))
    return this.tensor(this.dataType === 'float32' ? op : this.builder.cast(op, this.dataType), h, w, 3)
  }

  conv(x, layer, act) {
    const [O, k] = layer.shape, s = layer.stride[0]
    const { h, w, padding: pads } = padding(x.h, x.w, k, s, layer.pad)
    const op = this.builder.conv2d(x.op, this.constant(layer.shape, layer.weights), {
      inputLayout: 'nhwc', filterLayout: 'ohwi', strides: [s, s], padding: pads,
      bias: this.constant([O], layer.bias),
    })
    return this.tensor(this.activate(op, act), h, w, O)
  }

  // Depthwise = grouped conv with one group per channel; TFLite's [1, kh, kw, C]
  // filter is WebNN's 'ihwo' layout.
  depthwise(x, layer, act) {
    const k = layer.shape[1], s = layer.stride[0]
    const { h, w, padding: pads } = padding(x.h, x.w, k, s, layer.pad)
    const op = this.builder.conv2d(x.op, this.constant(layer.shape, layer.weights), {
      inputLayout: 'nhwc', filterLayout: 'ihwo', strides: [s, s], padding: pads, groups: x.c,
      bias: this.constant([x.c], layer.bias),
    })
    return this.tensor(this.activate(op, act), h, w, x.c)
  }

  convTranspose2x2(x, layer, act) {
    const O = layer.shape[0]
    const op = this.builder.convTranspose2d(x.op, this.constant(layer.shape, layer.weights), {
      inputLayout: 'nhwc', filterLayout: 'ohwi', strides: [2, 2],
      bias: this.constant([O], layer.bias),
    })
    return this.tensor(this.activate(op, act), x.h * 2, x.w * 2, O)
  }

  pool(x) {
    return this.tensor(this.builder.averagePool2d(x.op, { layout: 'nhwc' }), 1, 1, x.c)
  }

  upsample2x(x) {
    return this.tensor(this.builder.resample2d(x.op, { mode: 'linear', scales: [2, 2], axes: [1, 2] }), x.h * 2, x.w * 2, x.c)
  }

  add(a, b) {
    return this.tensor(this.builder.add(a.op, b.op), a.h, a.w, a.c)
  }

  mul(x, scale) {
    return this.tensor(this.builder.mul(x.op, scale.op), x.h, x.w, x.c)   // [1,1,1,C] broadcasts
  }

  // ---- running ---------------------------------------------------------------

  // Unlike the shader backends, nothing exists until the graph is compiled.
  // `gpuDevice` makes the input / mask tensors exportable to WebGPU.
  async compile(input, mask, { gpuDevice } = {}) {
    const out = this.dataType === 'float32' ? mask.op : this.builder.cast(mask.op, 'float32')
    this.graph = await this.builder.build({ mask: out })
    const make = (shape, usage) => gpuDevice
      ? this.context.createExportableTensor(tensorDesc(shape, usage), gpuDevice)
      : this.context.createTensor(tensorDesc(shape, usage))
    this.inputTensor = await make([1, input.h, input.w, 3], { writable: true, readable: true })
    this.maskTensor = await make([1, mask.h, mask.w, mask.c], { readable: true, writable: true })
    this.inputSize = input.h * input.w * 3
  }

  dispatch() {
    this.context.dispatch(this.graph, { input: this.inputTensor }, { mask: this.maskTensor })
  }

  // CPU in, CPU out.
  async run(data) {
    this.context.writeTensor(this.inputTensor, data)
    this.dispatch()
    return new Float32Array(await this.context.readTensor(this.maskTensor))
  }

  // ---- tensor I/O (tests) -------------------------------------------------------

  write(t, data) {
    this.pending = data
  }

  infer() {
    this.result = this.run(this.pending)
  }

  async read() {
    return this.result
  }

  get dispatches() {
    return 1
  }
}
