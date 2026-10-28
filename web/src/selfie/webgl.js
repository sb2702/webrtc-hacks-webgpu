// WebGL2 backend: every layer is a fragment shader drawing a full-screen
// triangle into an RGBA32F texture.
//
// Tensors use the same NHWC / vec4 layout as the WebGPU backend, laid out as a
// (w * groups) x h texture: texel (x * groups + g, y) holds channel group g of
// pixel (x, y). Weights are the same packed floats uploaded as RGBA32F
// textures — one texel per mat4 column. Getting frames in and out lives in
// src/frames/webgl.js, which shares this backend's GL context.

import { packConv, packDepthwise, packBias, convGeometry, groups } from './weights.js'

const ACT = { none: 0, relu: 1, hswish: 2, sigmoid: 3 }

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 result;
`

const ACTIVATE = `
vec4 activate(vec4 v) {
#if ACT == 1
  return max(v, vec4(0.0));
#elif ACT == 2
  return v * clamp(v + 3.0, vec4(0.0), vec4(6.0)) / 6.0;
#elif ACT == 3
  return 1.0 / (1.0 + exp(-v));
#else
  return v;
#endif
}`

const SHADERS = {
  conv: `
uniform sampler2D src, weight, bias;
uniform int in_w, in_h, in_g, out_g, k, stride, pad_y, pad_x;
${ACTIVATE}
void main() {
  int fx = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  int x = fx / out_g, o = fx - x * out_g;
  vec4 acc = texelFetch(bias, ivec2(o, 0), 0);
  for (int ky = 0; ky < k; ky++) {
    int iy = y * stride + ky - pad_y;
    if (iy < 0 || iy >= in_h) continue;
    for (int kx = 0; kx < k; kx++) {
      int ix = x * stride + kx - pad_x;
      if (ix < 0 || ix >= in_w) continue;
      int row = (ky * k + kx) * out_g + o;
      for (int i = 0; i < in_g; i++) {
        mat4 w = mat4(texelFetch(weight, ivec2(i * 4, row), 0), texelFetch(weight, ivec2(i * 4 + 1, row), 0),
                      texelFetch(weight, ivec2(i * 4 + 2, row), 0), texelFetch(weight, ivec2(i * 4 + 3, row), 0));
        acc += w * texelFetch(src, ivec2(ix * in_g + i, iy), 0);
      }
    }
  }
  result = activate(acc);
}`,

  depthwise: `
uniform sampler2D src, weight, bias;
uniform int in_w, in_h, g, k, stride, pad_y, pad_x;
${ACTIVATE}
void main() {
  int fx = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  int x = fx / g, c = fx - x * g;
  vec4 acc = texelFetch(bias, ivec2(c, 0), 0);
  for (int ky = 0; ky < k; ky++) {
    int iy = y * stride + ky - pad_y;
    if (iy < 0 || iy >= in_h) continue;
    for (int kx = 0; kx < k; kx++) {
      int ix = x * stride + kx - pad_x;
      if (ix < 0 || ix >= in_w) continue;
      acc += texelFetch(weight, ivec2(c, ky * k + kx), 0) * texelFetch(src, ivec2(ix * g + c, iy), 0);
    }
  }
  result = activate(acc);
}`,

  // Output pixel (X, Y) comes from input pixel (X/2, Y/2) through tap (Y%2, X%2).
  convTranspose2x2: `
uniform sampler2D src, weight, bias;
uniform int in_g, out_g;
${ACTIVATE}
void main() {
  int fx = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  int x = fx / out_g, o = fx - x * out_g;
  vec4 acc = texelFetch(bias, ivec2(o, 0), 0);
  int row = ((y % 2) * 2 + x % 2) * out_g + o;
  for (int i = 0; i < in_g; i++) {
    mat4 w = mat4(texelFetch(weight, ivec2(i * 4, row), 0), texelFetch(weight, ivec2(i * 4 + 1, row), 0),
                  texelFetch(weight, ivec2(i * 4 + 2, row), 0), texelFetch(weight, ivec2(i * 4 + 3, row), 0));
    acc += w * texelFetch(src, ivec2((x / 2) * in_g + i, y / 2), 0);
  }
  result = activate(acc);
}`,

  // Global average pool in two passes, since a fragment shader can't share
  // work across pixels: average each row (one fragment per row and channel
  // group), then average the column of row means.
  poolRows: `
uniform sampler2D src;
uniform int in_w, g;
void main() {
  int c = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  vec4 sum = vec4(0.0);
  for (int x = 0; x < in_w; x++) sum += texelFetch(src, ivec2(x * g + c, y), 0);
  result = sum / float(in_w);
}`,

  poolCols: `
uniform sampler2D src;
uniform int in_h;
void main() {
  int c = int(gl_FragCoord.x);
  vec4 sum = vec4(0.0);
  for (int y = 0; y < in_h; y++) sum += texelFetch(src, ivec2(c, y), 0);
  result = sum / float(in_h);
}`,

  upsample2x: `
uniform sampler2D src;
uniform int in_w, in_h, g;
vec4 at(ivec2 p, int c) { return texelFetch(src, ivec2(p.x * g + c, p.y), 0); }
void main() {
  int fx = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  int x = fx / g, c = fx - x * g;
  vec2 s = clamp((vec2(x, y) + 0.5) * 0.5 - 0.5, vec2(0.0), vec2(in_w - 1, in_h - 1));
  ivec2 p0 = ivec2(floor(s));
  ivec2 p1 = min(p0 + 1, ivec2(in_w - 1, in_h - 1));
  vec2 f = s - floor(s);
  result = mix(mix(at(p0, c), at(ivec2(p1.x, p0.y), c), f.x),
               mix(at(ivec2(p0.x, p1.y), c), at(p1, c), f.x), f.y);
}`,

  add: `
uniform sampler2D a, b;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  result = texelFetch(a, p, 0) + texelFetch(b, p, 0);
}`,

  mul: `
uniform sampler2D x, scale;
uniform int g;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  result = texelFetch(x, p, 0) * texelFetch(scale, ivec2(p.x % g, 0), 0);
}`,
}

export const VERTEX = `#version 300 es
void main() {
  vec2 xy = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(xy * 2.0 - 1.0, 0.0, 1.0);
}`

export class WebGLBackend {
  // Takes an existing WebGL2 context (so frame I/O and other libraries can
  // share it) or a canvas to create one on.
  static async create(target) {
    const gl = target instanceof WebGL2RenderingContext
      ? target
      : target.getContext('webgl2', { antialias: false, depth: false, alpha: false })
    if (!gl) throw new Error('WebGL2 is not available')
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float is not available')
    return new WebGLBackend(gl)
  }

  constructor(gl) {
    this.name = 'webgl'
    this.gl = gl
    this.ops = []
    this.programs = new Map()
    // Shared libraries (MediaPipe) may leave a VAO bound with attributes
    // enabled; our full-screen triangles use no attributes at all.
    this.vao = gl.createVertexArray()
  }

  // ---- building blocks ------------------------------------------------------

  floatTexture(w, h, data = null) {
    const gl = this.gl
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    return texture
  }

  tensor(h, w, c) {
    const gl = this.gl
    const g = groups(c)
    const texture = this.floatTexture(w * g, h)
    const fbo = gl.createFramebuffer()
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
    return { h, w, c: g * 4, g, texture, fbo }
  }

  // Compile (and cache) a fragment shader; `source` defaults to the layer shaders above.
  program(name, act = 'none', source = SHADERS[name]) {
    const key = `${name}:${act}`
    if (!this.programs.has(key)) {
      const gl = this.gl
      const compile = (type, source) => {
        const shader = gl.createShader(type)
        gl.shaderSource(shader, source)
        gl.compileShader(shader)
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(`${name}: ${gl.getShaderInfoLog(shader)}`)
        return shader
      }
      const program = gl.createProgram()
      gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX))
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, `${HEADER}#define ACT ${ACT[act]}\n${source}`))
      gl.linkProgram(program)
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`${name}: ${gl.getProgramInfoLog(program)}`)
      this.programs.set(key, program)
    }
    return this.programs.get(key)
  }

  // Resolve uniform locations once; returns a function that binds everything.
  binder(program, { textures = {}, ints = {}, floats = {} }) {
    const gl = this.gl
    const loc = (name) => gl.getUniformLocation(program, name)
    const tex = Object.entries(textures).map(([name, texture], unit) => [loc(name), unit, texture])
    const i = Object.entries(ints).map(([name, v]) => [loc(name), v])
    const f = Object.entries(floats).map(([name, v]) => [loc(name), v])
    return () => {
      gl.useProgram(program)
      for (const [l, unit, texture] of tex) {
        gl.activeTexture(gl.TEXTURE0 + unit)
        gl.bindTexture(gl.TEXTURE_2D, typeof texture === 'function' ? texture() : texture)
        gl.uniform1i(l, unit)
      }
      for (const [l, v] of i) gl.uniform1i(l, v)
      for (const [l, v] of f) Array.isArray(v) ? gl.uniform2fv(l, v) : gl.uniform1f(l, v)
    }
  }

  // Record one draw into `out`.
  op(name, { act, out, textures, ints }) {
    const bind = this.binder(this.program(name, act), {
      textures: Object.fromEntries(Object.entries(textures).map(([k, t]) => [k, t.texture ?? t])),
      ints,
    })
    this.ops.push({ bind, out })
    return out
  }

  weights(data, w, h) {
    return this.floatTexture(w, h, data)
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
      textures: {
        src: x,
        weight: this.weights(packConv(layer), x.g * 4, k * k * out.g),
        bias: this.weights(packBias(layer), out.g, 1),
      },
      ints: { in_w: x.w, in_h: x.h, in_g: x.g, out_g: out.g, k, stride: s, pad_y: gy.pad, pad_x: gx.pad },
    })
  }

  depthwise(x, layer, act) {
    const k = layer.shape[1], s = layer.stride[0]
    const gy = convGeometry(x.h, k, s, layer.pad), gx = convGeometry(x.w, k, s, layer.pad)
    const out = this.tensor(gy.out, gx.out, x.c)
    return this.op('depthwise', {
      act, out,
      textures: {
        src: x,
        weight: this.weights(packDepthwise(layer), x.g, k * k),
        bias: this.weights(packBias(layer), x.g, 1),
      },
      ints: { in_w: x.w, in_h: x.h, g: x.g, k, stride: s, pad_y: gy.pad, pad_x: gx.pad },
    })
  }

  convTranspose2x2(x, layer, act) {
    const out = this.tensor(x.h * 2, x.w * 2, layer.shape[0])
    return this.op('convTranspose2x2', {
      act, out,
      textures: {
        src: x,
        weight: this.weights(packConv(layer), x.g * 4, 4 * out.g),
        bias: this.weights(packBias(layer), out.g, 1),
      },
      ints: { in_g: x.g, out_g: out.g },
    })
  }

  pool(x) {
    const rows = this.op('poolRows', { out: this.tensor(x.h, 1, x.c), textures: { src: x }, ints: { in_w: x.w, g: x.g } })
    return this.op('poolCols', { out: this.tensor(1, 1, x.c), textures: { src: rows }, ints: { in_h: x.h } })
  }

  upsample2x(x) {
    return this.op('upsample2x', {
      out: this.tensor(x.h * 2, x.w * 2, x.c), textures: { src: x }, ints: { in_w: x.w, in_h: x.h, g: x.g },
    })
  }

  add(a, b) {
    return this.op('add', { out: this.tensor(a.h, a.w, a.c), textures: { a, b } })
  }

  mul(x, scale) {
    return this.op('mul', { out: this.tensor(x.h, x.w, x.c), textures: { x, scale }, ints: { g: x.g } })
  }

  // ---- running ---------------------------------------------------------------

  drawTo(target, w, h) {
    const gl = this.gl
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, target)
    gl.viewport(0, 0, w, h)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  // Undo any fixed-function state another user of the context left enabled.
  resetState() {
    const gl = this.gl
    for (const cap of [gl.BLEND, gl.SCISSOR_TEST, gl.DEPTH_TEST, gl.STENCIL_TEST, gl.CULL_FACE]) gl.disable(cap)
  }

  run() {
    this.resetState()
    for (const { bind, out } of this.ops) {
      bind()
      this.drawTo(out.fbo, out.w * out.g, out.h)
    }
  }

  // Run just the network on whatever is in the input tensor (for tests).
  infer() {
    this.run()
  }

  // WebGL has no "work done" promise; reading back a single texel forces the
  // pipeline to drain.
  async sync(t = this.ops.at(-1).out) {
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo)
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, (this.syncPixel ??= new Float32Array(4)))
  }

  // ---- tensor I/O (tests / debugging) -----------------------------------------

  write(t, data, channels) {
    const gl = this.gl
    const packed = new Float32Array(t.h * t.w * t.c)
    for (let p = 0; p < t.h * t.w; p++)
      for (let c = 0; c < channels; c++) packed[p * t.c + c] = data[p * channels + c]
    gl.bindTexture(gl.TEXTURE_2D, t.texture)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, t.w * t.g, t.h, gl.RGBA, gl.FLOAT, packed)
  }

  async read(t, channels = t.c) {
    const gl = this.gl
    const packed = new Float32Array(t.h * t.w * t.c)
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo)
    gl.readPixels(0, 0, t.w * t.g, t.h, gl.RGBA, gl.FLOAT, packed)
    const out = new Float32Array(t.h * t.w * channels)
    for (let p = 0; p < t.h * t.w; p++)
      for (let c = 0; c < channels; c++) out[p * channels + c] = packed[p * t.c + c]
    return out
  }

  get dispatches() {
    return this.ops.length
  }
}
