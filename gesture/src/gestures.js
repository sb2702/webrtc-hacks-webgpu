// Stage 5: 21 landmarks -> a named gesture.
//
// Two small models:
//   gesture_embedder.tflite         landmarks + world landmarks + handedness -> 128 numbers
//   canned_gesture_classifier.tflite  128 numbers -> 8 gesture probabilities
//
// Neither looks at pixels, so there's nothing to keep on the GPU: inputs are
// 127 floats. The embedder runs on LiteRT's WASM (CPU) backend; its WebGPU
// backend can't compile this graph, and at this size the CPU is the right
// place anyway. The classifier is one scale/shift + ReLU + dense layer; its
// .tflite declares a dynamic batch size LiteRT.js rejects, so we evaluate it
// in a few lines of JS with weights exported by tools/export_classifier.py.
//
// Landmark preprocessing follows MediaPipe's landmarks_to_matrix_calculator.cc
// and handedness_to_matrix_calculator.cc.

export class GestureClassifier {
  static async create(litert, embedderBytes, classifierUrl) {
    const embedder = await litert.loadAndCompile(embedderBytes, { accelerator: 'wasm' })
    const classifier = await (await fetch(classifierUrl)).json()
    return new GestureClassifier(litert, embedder, classifier)
  }

  constructor(litert, embedder, classifier) {
    this.litert = litert
    this.embedder = embedder
    this.classifier = classifier
  }

  // landmarks: frame-normalised (projectLandmarks), worldLandmarks: metres,
  // rightScore: the landmark model's handedness output.
  async classify(landmarks, worldLandmarks, rightScore, frameWidth, frameHeight) {
    const { Tensor } = this.litert
    const inputs = {
      hand: new Tensor(handFeatures(landmarks, frameWidth, frameHeight), [1, 21, 3]),
      handedness: new Tensor(new Float32Array([rightScore]), [1, 1]),
      world_hand: new Tensor(normalizeObject(worldLandmarks), [1, 21, 3]),
    }
    const outputs = await this.embedder.run(inputs)
    const embedding = Object.values(outputs)[0]
    const probabilities = this.classifyEmbedding(embedding.toTypedArray())
    for (const t of [...Object.values(inputs), embedding]) t.delete()

    const ranked = this.classifier.labels.map((name, i) => ({ name, score: probabilities[i] })).sort((a, b) => b.score - a.score)
    return { ...ranked[0], all: ranked }
  }

  // softmax(W · relu(x * scale + shift) + b)
  classifyEmbedding(x) {
    const { scale, shift, weights, bias } = this.classifier
    const hidden = Array.from(x, (v, i) => Math.max(0, v * scale[i] + shift[i]))
    const logits = bias.map((b, o) => hidden.reduce((sum, h, i) => sum + h * weights[o * 128 + i], b))
    const max = Math.max(...logits)
    const exp = logits.map((l) => Math.exp(l - max))
    const total = exp.reduce((a, b) => a + b, 0)
    return exp.map((e) => e / total)
  }
}

// Frame-normalised landmarks are first made aspect-correct (so a hand looks
// the same shape in a 16:9 frame as in a square one), then normalised.
function handFeatures(landmarks, frameWidth, frameHeight) {
  const max = Math.max(frameWidth, frameHeight)
  const sx = frameWidth / max, sy = frameHeight / max
  return normalizeObject(landmarks.map(({ x, y, z }) => ({ x: (x - 0.5) * sx + 0.5, y: (y - 0.5) * sy + 0.5, z })))
}

// Make the hand position- and size-independent: wrist at the origin, scaled so
// the larger of its width and height is 1. Returns 21 × (x, y, z), flattened.
function normalizeObject(points) {
  const origin = points[0]
  const shifted = points.map((p) => ({ x: p.x - origin.x, y: p.y - origin.y, z: p.z - origin.z }))
  const xs = shifted.map((p) => p.x), ys = shifted.map((p) => p.y)
  const scale = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 1e-5
  return Float32Array.from(shifted.flatMap((p) => [p.x / scale, p.y / scale, p.z / scale]))
}
