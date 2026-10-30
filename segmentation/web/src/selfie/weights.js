// Loads the weights exported by tools/export_weights.py and repacks them into
// the layout both GPU backends use.
//
// Activations are NHWC with channels packed 4 at a time into vec4s, so a
// convolution is a sum of (4x4 matrix) * (vec4) products. The packed conv
// weights are a [kernel tap][out group][in group] array of 4x4 blocks where
// column = input channel and row = output channel. WGSL reads that directly as
// array<mat4x4f> (column-major); WebGL reads the same floats from an RGBA32F
// texture, one texel per column.

export async function loadWeights(base) {
  const [manifest, blob] = await Promise.all([
    fetch(`${base}.json`).then((r) => r.json()),
    fetch(`${base}.bin`).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
  ])
  const layers = {}
  for (const [name, l] of Object.entries(manifest)) {
    const n = l.shape.reduce((a, b) => a * b, 1)
    const outChannels = l.kind === 'depthwise' ? l.shape[3] : l.shape[0]
    layers[name] = {
      ...l,
      name,
      weights: blob.subarray(l.w, l.w + n),
      bias: blob.subarray(l.b, l.b + outChannels),
    }
  }
  return layers
}

export const groups = (c) => Math.ceil(c / 4)

// TFLite conv weights [out, kh, kw, in] -> [kh*kw][out/4][in/4] of 4x4 blocks.
// Also used for the 2x2 transposed conv, whose taps are its 4 output phases.
export function packConv({ shape: [O, kh, kw, I], weights }) {
  const K = kh * kw, Og = groups(O), Ig = groups(I)
  const out = new Float32Array(K * Og * Ig * 16)
  for (let o = 0; o < O; o++)
    for (let z = 0; z < K; z++)
      for (let i = 0; i < I; i++)
        out[((z * Og + (o >> 2)) * Ig + (i >> 2)) * 16 + (i & 3) * 4 + (o & 3)] = weights[(o * K + z) * I + i]
  return out
}

// TFLite depthwise weights [1, kh, kw, C] -> [kh*kw][C/4] of vec4.
export function packDepthwise({ shape: [, kh, kw, C], weights }) {
  const Cg = groups(C)
  const out = new Float32Array(kh * kw * Cg * 4)
  for (let z = 0; z < kh * kw; z++)
    for (let c = 0; c < C; c++) out[z * Cg * 4 + c] = weights[z * C + c]
  return out
}

export function packBias({ bias }) {
  const out = new Float32Array(groups(bias.length) * 4)
  out.set(bias)
  return out
}

// Output size and top/left padding for TF 'SAME' / 'VALID' convolutions.
// SAME pads the extra row/column on the bottom/right, which the shaders get
// for free by skipping out-of-bounds reads.
export function convGeometry(size, k, stride, pad) {
  if (pad === 'VALID') return { out: Math.floor((size - k) / stride) + 1, pad: 0 }
  const out = Math.ceil(size / stride)
  return { out, pad: Math.floor(Math.max((out - 1) * stride + k - size, 0) / 2) }
}
