// MediaPipe's selfie segmenter, written against a tiny backend interface so
// the exact same graph runs on WebGPU and WebGL. Mirrors selfie() in
// tools/selfie_numpy.py, which is checked against the TFLite interpreter.
//
// A backend provides:
//   input(h, w)                      -> tensor the frame gets resized into
//   conv(x, layer, act)              -> standard conv (1x1 or kxk, stride 1|2)
//   depthwise(x, layer, act)         -> depthwise conv
//   convTranspose2x2(x, layer, act)  -> 2x2 stride-2 transposed conv
//   pool(x)                          -> global average pool, h x w x c -> 1 x 1 x c
//   upsample2x(x)                    -> bilinear, half-pixel centers
//   add(a, b)                        -> elementwise, same shape
//   mul(x, s)                        -> x * s, s is 1 x 1 x c (per-channel scale)
//
// Calls record GPU work; nothing runs until the backend's render()/infer().

export function buildSelfie(b, W) {
  const conv = (x, name, act = 'none') => b.conv(x, W[name], act)
  const depthwise = (x, name, act = 'none') => b.depthwise(x, W[name], act)

  // Channel attention: x * sigmoid(fc(relu(fc(mean(x)))))
  const squeezeExcite = (x, fc1, fc2) => b.mul(x, conv(conv(b.pool(x), fc1, 'relu'), fc2, 'sigmoid'))

  // MobileNetV3 block: 1x1 expand -> depthwise -> (squeeze-excite) -> 1x1 project
  const invertedResidual = (x, expand, dw, project, { se, act = 'relu', residual = false } = {}) => {
    let y = depthwise(conv(x, expand, act), dw, act)
    if (se) y = squeezeExcite(y, ...se)
    y = conv(y, project)
    return residual ? b.add(x, y) : y
  }

  // Upsample, blend with the encoder skip connection through an attention
  // gate, then refine with a 1x1 conv and a residual depthwise conv.
  const decoderBlock = (x, skip, proj, [fc1, fc2], conv1, dw) => {
    const up = conv(b.upsample2x(x), proj)
    const gate = conv(conv(b.pool(b.add(skip, up)), fc1, 'relu'), fc2, 'sigmoid')
    const y = conv(b.add(b.mul(skip, gate), up), conv1, 'relu')
    return b.add(y, depthwise(y, dw, 'relu'))
  }

  const input = b.input(256, 256)                                              // 256² x3, RGB in [0, 1]

  // Encoder
  const s1 = conv(input, 'conv2d', 'hswish')                                   // 128² x16
  let x = depthwise(conv(s1, 'conv2d_1', 'relu'), 'depthwise_conv2d', 'relu')
  x = squeezeExcite(x, 'conv2d_2', 'conv2d_3')
  const s2 = conv(x, 'conv2d_4')                                               // 64² x16
  x = invertedResidual(s2, 'conv2d_5', 'depthwise_conv2d_1', 'conv2d_6')       // 32² x24
  const s3 = invertedResidual(x, 'conv2d_7', 'depthwise_conv2d_2', 'conv2d_8', { residual: true })
  x = invertedResidual(s3, 'conv2d_9', 'depthwise_conv2d_3', 'conv2d_12',      // 16² x32
    { se: ['conv2d_10', 'conv2d_11'], act: 'hswish' })
  for (let i = 0; i < 4; i++) {
    const n = 13 + 4 * i
    x = invertedResidual(x, `conv2d_${n}`, `depthwise_conv2d_${4 + i}`, `conv2d_${n + 3}`,
      { se: [`conv2d_${n + 1}`, `conv2d_${n + 2}`], act: 'hswish', residual: true })
  }

  // LR-ASPP style head: features gated by a global-context branch
  x = b.mul(conv(x, 'conv2d_29', 'relu'), conv(b.pool(x), 'conv2d_30', 'sigmoid'))   // 16² x128

  // Decoder
  x = decoderBlock(x, s3, 'conv2d_31', ['conv2d_32', 'conv2d_33'], 'conv2d_34', 'depthwise_conv2d_8')   // 32² x24
  x = decoderBlock(x, s2, 'conv2d_35', ['conv2d_36', 'conv2d_37'], 'conv2d_38', 'depthwise_conv2d_9')   // 64² x16
  x = decoderBlock(x, s1, 'conv2d_39', ['conv2d_40', 'conv2d_41'], 'conv2d_42', 'depthwise_conv2d_10')  // 128² x16

  const mask = b.convTranspose2x2(x, W.segment, 'sigmoid')                    // 256² x1
  return { input, mask }
}
