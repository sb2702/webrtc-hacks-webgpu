// Stage 2: the palm detector's raw output -> at most a few palm detections.
//
// The detector is an SSD-style model. It doesn't output "the hand is here";
// it scores 2016 fixed candidate boxes ("anchors") laid on a grid over the
// 192×192 input, and for each one predicts an offset from the anchor's centre,
// a box size and 7 palm keypoints:
//
//   boxes:  [2016][18]  dx, dy, w, h, then 7 × (x, y) keypoints, in input pixels
//   scores: [2016]      logits (sigmoid -> probability)
//
// A real hand lights up dozens of neighbouring anchors with nearly the same
// box. Non-maximum suppression (NMS) collapses them: take the highest-scoring
// candidate, gather every remaining candidate that overlaps it by more than
// 30% (intersection over union), emit one detection, discard the group,
// repeat. MediaPipe uses the "weighted" variant: the emitted box and keypoints
// are the score-weighted average of the group rather than just the winner's,
// which makes them noticeably steadier.
//
// Two implementations with identical results:
//   decodeAndSuppress()  plain JS on the CPU, after reading back ~150 KB of model output
//   GpuPalmDecoder       one WGSL dispatch that reads LiteRT's output buffers in
//                        place and reads back only the final detections (~80 bytes)

export const INPUT_SIZE = 192
export const NUM_ANCHORS = 2016
const VALUES_PER_ANCHOR = 18
const MIN_SCORE = 0.5 // min_detection_confidence
const MIN_IOU = 0.3 // min_suppression_threshold
// One detection as returned by the GPU decoder: score, box (xmin, ymin, xmax, ymax), 7 keypoints.
const DETECTION_FLOATS = 1 + 4 + 14

// Anchor centres, normalised to the input. Same as MediaPipe's
// SsdAnchorsCalculator with the palm detector's options: strides 8, 16, 16, 16
// give a 24×24 grid with 2 anchors per cell, then a 12×12 grid with 6 (the
// three stride-16 layers share a grid). All anchors are 1×1, so the model's
// box sizes are used as-is.
export function palmAnchors() {
  const anchors = new Float32Array(NUM_ANCHORS * 2)
  let i = 0
  for (const [grid, perCell] of [[24, 2], [12, 6]]) {
    for (let y = 0; y < grid; y++)
      for (let x = 0; x < grid; x++)
        for (let k = 0; k < perCell; k++) {
          anchors[i++] = (x + 0.5) / grid
          anchors[i++] = (y + 0.5) / grid
        }
  }
  return anchors
}

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.min(100, Math.max(-100, x))))

function intersectionOverUnion(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  if (w <= 0 || h <= 0) return 0
  const intersection = w * h
  return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
}

// CPU version. Detections come out normalised to the 192×192 (letterboxed)
// input: { score, box: [xmin, ymin, xmax, ymax], keypoints: [x0, y0, … x6, y6] }.
export function decodeAndSuppress(boxes, scores, anchors, maxDetections) {
  // 1. Decode every anchor that clears the score threshold.
  let candidates = []
  for (let i = 0; i < NUM_ANCHORS; i++) {
    const score = sigmoid(scores[i])
    if (score < MIN_SCORE) continue
    const r = i * VALUES_PER_ANCHOR
    const ax = anchors[2 * i], ay = anchors[2 * i + 1]
    const cx = boxes[r] / INPUT_SIZE + ax, cy = boxes[r + 1] / INPUT_SIZE + ay
    const w = boxes[r + 2] / INPUT_SIZE, h = boxes[r + 3] / INPUT_SIZE
    const keypoints = new Float32Array(14)
    for (let k = 0; k < 7; k++) {
      keypoints[2 * k] = boxes[r + 4 + 2 * k] / INPUT_SIZE + ax
      keypoints[2 * k + 1] = boxes[r + 5 + 2 * k] / INPUT_SIZE + ay
    }
    candidates.push({ score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], keypoints })
  }

  // 2. Weighted non-maximum suppression.
  candidates.sort((a, b) => b.score - a.score)
  const detections = []
  while (candidates.length && detections.length < maxDetections) {
    const [top, ...rest] = candidates
    const overlaps = (c) => intersectionOverUnion(c.box, top.box) > MIN_IOU
    const group = [top, ...rest.filter(overlaps)]
    candidates = rest.filter((c) => !overlaps(c))
    const total = group.reduce((sum, c) => sum + c.score, 0)
    const box = [0, 1, 2, 3].map((j) => group.reduce((sum, c) => sum + c.box[j] * c.score, 0) / total)
    const keypoints = top.keypoints.map((_, j) => group.reduce((sum, c) => sum + c.keypoints[j] * c.score, 0) / total)
    detections.push({ score: top.score, box, keypoints })
  }
  return detections
}

// GPU version: the same algorithm as a single workgroup of 64 threads.
//
//   1. Each thread decodes the scores of every 64th anchor into workgroup
//      (shared) memory.
//   2. Repeat for each detection wanted:
//      a. argmax: each thread finds the best score among its anchors, then a
//         tree reduction halves the 64 candidates 6 times to find the winner;
//      b. each thread adds up score-weighted boxes and keypoints of its anchors
//         that overlap the winner by > 30% and marks them suppressed (score −1);
//      c. a second tree reduction sums those 64 partial sums, and thread 0
//         writes the averaged detection.
//
// 2016 anchors is small for a GPU: this isn't about raw speed but about not
// shipping 150 KB of raw model output to the CPU every frame just to throw
// 99% of it away. The detector's output buffers are bound directly.
const NMS_SHADER = /* wgsl */ `
override MAX_DETECTIONS: u32 = 1u;
const N = ${NUM_ANCHORS}u;
const T = 64u;                    // threads in the workgroup
const SIZE = ${INPUT_SIZE}.0;
const MIN_SCORE = ${MIN_SCORE};
const MIN_IOU = ${MIN_IOU};
const F = ${DETECTION_FLOATS}u;   // floats per detection: score, box, 7 keypoints

@group(0) @binding(0) var<storage, read> rawBoxes: array<f32>;   // [N][18] model output
@group(0) @binding(1) var<storage, read> rawScores: array<f32>;  // [N] model output (logits)
@group(0) @binding(2) var<storage, read> anchors: array<vec2f>;  // [N] anchor centres
@group(0) @binding(3) var<storage, read_write> result: array<f32>; // count, then MAX_DETECTIONS × F

var<workgroup> scores: array<f32, N>;
var<workgroup> bestScore: array<f32, T>;
var<workgroup> bestIndex: array<u32, T>;
var<workgroup> sums: array<array<f32, F>, T>; // total weight, then weighted box + keypoints

fn decodeBox(i: u32) -> vec4f {
  let r = i * 18u;
  let centre = vec2f(rawBoxes[r], rawBoxes[r + 1u]) / SIZE + anchors[i];
  let half = vec2f(rawBoxes[r + 2u], rawBoxes[r + 3u]) / SIZE * 0.5;
  return vec4f(centre - half, centre + half);  // xmin, ymin, xmax, ymax
}

fn keypoint(i: u32, k: u32) -> vec2f {
  let r = i * 18u + 4u + 2u * k;
  return vec2f(rawBoxes[r], rawBoxes[r + 1u]) / SIZE + anchors[i];
}

fn intersectionOverUnion(a: vec4f, b: vec4f) -> f32 {
  let overlap = max(min(a.zw, b.zw) - max(a.xy, b.xy), vec2f(0.0));
  let intersection = overlap.x * overlap.y;
  let unionArea = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - intersection;
  return select(0.0, intersection / unionArea, intersection > 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
  // 1. Scores: sigmoid of the (clipped) logits.
  for (var i = t; i < N; i += T) {
    scores[i] = 1.0 / (1.0 + exp(-clamp(rawScores[i], -100.0, 100.0)));
  }
  workgroupBarrier();

  var count = 0u;
  for (var d = 0u; d < MAX_DETECTIONS; d++) {
    // 2a. Argmax over all unsuppressed anchors.
    var best = -1.0;
    var index = 0u;
    for (var i = t; i < N; i += T) {
      if (scores[i] > best) { best = scores[i]; index = i; }
    }
    bestScore[t] = best;
    bestIndex[t] = index;
    workgroupBarrier();
    for (var stride = T / 2u; stride > 0u; stride /= 2u) {
      if (t < stride && bestScore[t + stride] > bestScore[t]) {
        bestScore[t] = bestScore[t + stride];
        bestIndex[t] = bestIndex[t + stride];
      }
      workgroupBarrier();
    }
    // Every thread must take the same branch around barriers; workgroupUniformLoad
    // makes that provable to the compiler.
    let topScore = workgroupUniformLoad(&bestScore[0]);
    if (topScore < MIN_SCORE) { break; }
    let topIndex = bestIndex[0];
    let topBox = decodeBox(topIndex);

    // 2b. Each thread's share of the weighted average; suppress what it used.
    var acc: array<f32, F>;
    for (var i = t; i < N; i += T) {
      if (scores[i] >= MIN_SCORE) {
        let box = decodeBox(i);
        if (i == topIndex || intersectionOverUnion(box, topBox) > MIN_IOU) {
          let w = scores[i];
          acc[0] += w;
          acc[1] += w * box.x; acc[2] += w * box.y; acc[3] += w * box.z; acc[4] += w * box.w;
          for (var k = 0u; k < 7u; k++) {
            let p = keypoint(i, k);
            acc[5u + 2u * k] += w * p.x;
            acc[6u + 2u * k] += w * p.y;
          }
          scores[i] = -1.0;
        }
      }
    }
    sums[t] = acc;
    workgroupBarrier();

    // 2c. Sum the 64 partial sums.
    for (var stride = T / 2u; stride > 0u; stride /= 2u) {
      if (t < stride) {
        for (var j = 0u; j < F; j++) { sums[t][j] += sums[t + stride][j]; }
      }
      workgroupBarrier();
    }
    if (t == 0u) {
      result[1u + d * F] = topScore;
      for (var j = 1u; j < F; j++) { result[1u + d * F + j] = sums[0][j] / sums[0][0]; }
    }
    count += 1u;
    workgroupBarrier(); // bestScore / sums are reused by the next iteration
  }
  if (t == 0u) { result[0] = f32(count); }
}
`

export class GpuPalmDecoder {
  constructor(device, readback, maxDetections) {
    this.device = device
    this.readback = readback
    this.maxDetections = maxDetections
    this.anchors = device.createBuffer({ size: NUM_ANCHORS * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(this.anchors, 0, palmAnchors())
    this.resultBytes = (1 + maxDetections * DETECTION_FLOATS) * 4
    this.result = device.createBuffer({ size: this.resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: NMS_SHADER }),
        entryPoint: 'main',
        constants: { MAX_DETECTIONS: maxDetections },
      },
    })
  }

  // boxes / scores: the detector's output GPUBuffers, straight from LiteRT
  // (tensor.toGpuBuffer()). LiteRT hands back new buffers on every run, so the
  // bind group is per call.
  async run(boxes, scores) {
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [boxes, scores, this.anchors, this.result].map((buffer, binding) => ({ binding, resource: { buffer } })),
    })
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(1)
    pass.end()
    const [data] = await this.readback.read([{ buffer: this.result, bytes: this.resultBytes }], encoder)

    const detections = []
    for (let d = 0; d < data[0]; d++) {
      const o = 1 + d * DETECTION_FLOATS
      detections.push({ score: data[o], box: Array.from(data.subarray(o + 1, o + 5)), keypoints: data.slice(o + 5, o + 19) })
    }
    return detections
  }
}

// Detections are normalised to the letterboxed 192×192 input; map them back to
// the frame by undoing the black bars. (detection_letterbox_removal_calculator.cc)
export function removeLetterbox(detection, padding) {
  const sx = 1 / (1 - 2 * padding.x), sy = 1 / (1 - 2 * padding.y)
  const x = (v) => (v - padding.x) * sx
  const y = (v) => (v - padding.y) * sy
  const [xmin, ymin, xmax, ymax] = detection.box
  return {
    score: detection.score,
    box: [x(xmin), y(ymin), x(xmax), y(ymax)],
    keypoints: detection.keypoints.map((v, j) => (j % 2 ? y(v) : x(v))),
  }
}
