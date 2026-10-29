// The whole pipeline: VideoFrame in, hand landmarks + gesture out.
//
//   const recognizer = await HandGestureRecognizer.create()
//   const result = await recognizer.process(videoFrame)   // caller still owns the frame
//
//   ┌──────────── no hand being tracked ────────────┐
//   │ 1. letterbox the frame to 192×192   (WGSL)     │
//   │ 2. palm detector                    (LiteRT, WebGPU)
//   │    decode anchors + weighted NMS    (WGSL, or JS for comparison)
//   │    palm -> rotated hand rect        (JS, a few lines of trig)
//   └────────────────────────────────────────────────┘
//     3. rotated crop of the frame to 224×224  (WGSL, same shader as 1.)
//     4. landmark model                         (LiteRT, WebGPU)
//        read back 4 tiny outputs (~520 bytes), project to the frame
//     5. landmarks -> next frame's hand rect, so 1–2 are skipped while tracking
//        gesture embedder + classifier          (LiteRT WASM + 20 lines of JS)
//
// Every frame-sized piece of data stays in GPU buffers. The CPU only sees the
// final detections and landmarks, and only waits on the GPU once or twice per frame.

import * as litert from '@litertjs/core'
import { getDevice, Readback } from './gpu.js'
import { loadTaskBundle } from './task-bundle.js'
import { FrameWarp, letterbox, rectToMatrix } from './warp.js'
import { GpuPalmDecoder, decodeAndSuppress, palmAnchors, removeLetterbox, INPUT_SIZE as PALM_SIZE } from './palm.js'
import { palmToRect, landmarksToRect } from './rects.js'
import { projectLandmarks, projectWorldLandmarks, INPUT_SIZE as LANDMARK_SIZE } from './landmarks.js'
import { GestureClassifier } from './gestures.js'

const MIN_HAND_PRESENCE = 0.5 // min_hand_presence_confidence

let liteRtReady
function initLiteRt(device) {
  return (liteRtReady ??= (async () => {
    // Threads need cross-origin isolation (COOP/COEP headers, see vite.config.js).
    await litert.loadLiteRt(`${import.meta.env.BASE_URL}vendor/litert/`, { threads: self.crossOriginIsolated })
    // LiteRT runs on *our* device, so its tensors and our buffers are interchangeable.
    litert.setWebGpuDevice(device)
  })())
}

export class HandGestureRecognizer {
  static async create({
    taskUrl = `${import.meta.env.BASE_URL}models/gesture_recognizer.task`,
    classifierUrl = `${import.meta.env.BASE_URL}models/gesture_classifier.json`,
  } = {}) {
    const device = await getDevice()
    const [models] = await Promise.all([loadTaskBundle(taskUrl), initLiteRt(device)])
    const [palmDetector, landmarker, gestures] = await Promise.all([
      litert.loadAndCompile(models['hand_detector.tflite'], { accelerator: 'webgpu' }),
      litert.loadAndCompile(models['hand_landmarks_detector.tflite'], { accelerator: 'webgpu' }),
      GestureClassifier.create(litert, models['gesture_embedder.tflite'], classifierUrl),
    ])
    return new HandGestureRecognizer(device, palmDetector, landmarker, gestures)
  }

  constructor(device, palmDetector, landmarker, gestures) {
    this.device = device
    this.palmDetector = palmDetector
    this.landmarker = landmarker
    this.gestures = gestures
    this.readback = new Readback(device)

    // The two model inputs: GPU buffers our shaders write, wrapped (not
    // copied) as LiteRT tensors once, up front.
    this.palmInput = new FrameWarp(device, PALM_SIZE, { zeroBorder: true })
    this.cropInput = new FrameWarp(device, LANDMARK_SIZE, { zeroBorder: false })
    this.palmTensor = new litert.Tensor(this.palmInput.buffer, [1, PALM_SIZE, PALM_SIZE, 3], 'float32')
    this.cropTensor = new litert.Tensor(this.cropInput.buffer, [1, LANDMARK_SIZE, LANDMARK_SIZE, 3], 'float32')

    this.gpuDecoder = new GpuPalmDecoder(device, this.readback, 1)
    this.anchors = palmAnchors()
    this.trackedRect = null
  }

  // options.tracking: reuse the previous frame's landmarks instead of running
  //   the palm detector (video). Turn off for unrelated still images.
  // options.nms: 'gpu' (WGSL) or 'js' (read the raw detector output back).
  async process(frame, { tracking = true, nms = 'gpu' } = {}) {
    const width = frame.displayWidth, height = frame.displayHeight
    const timings = {}
    let t = performance.now()
    const lap = (name) => { const now = performance.now(); timings[name] = now - t; t = now }

    let rect = tracking ? this.trackedRect : null
    let palm = null
    if (!rect) {
      // 1–2. Find a palm in the whole frame.
      const { matrix, padding } = letterbox(width, height, PALM_SIZE)
      this.palmInput.run(frame, matrix)
      const detections = await this.detectPalms(nms)
      lap('palm')
      if (!detections.length) return this.result({ timings })
      palm = removeLetterbox(detections[0], padding)
      rect = palmToRect(palm, width, height)
    }

    // 3–4. Crop the hand, find its landmarks.
    this.cropInput.run(frame, rectToMatrix(rect, width, height, LANDMARK_SIZE))
    const outputs = await this.landmarker.run([this.cropTensor])
    const [raw, [presence], [rightScore], rawWorld] = await this.readback.read(
      outputs.map((tensor, i) => ({ buffer: tensor.toGpuBuffer(), bytes: [63, 1, 1, 63][i] * 4 })),
    )
    outputs.forEach((tensor) => tensor.delete())
    lap('landmarks')
    if (presence < MIN_HAND_PRESENCE) {
      this.trackedRect = null // lost it: look for a palm again next frame
      return this.result({ palm, rect, presence, timings })
    }
    const landmarks = projectLandmarks(raw, rect)
    const worldLandmarks = projectWorldLandmarks(rawWorld, rect)

    // 5. Where to look next frame, and what the hand is doing.
    this.trackedRect = landmarksToRect(landmarks, width, height)
    const gesture = await this.gestures.classify(landmarks, worldLandmarks, rightScore, width, height)
    lap('gesture')
    const handedness = rightScore > 0.5 ? { name: 'Right', score: rightScore } : { name: 'Left', score: 1 - rightScore }
    return this.result({ palm, rect, presence, hand: { landmarks, worldLandmarks, handedness, gesture }, timings })
  }

  async detectPalms(nms) {
    const [boxes, scores] = await this.palmDetector.run([this.palmTensor])
    try {
      if (nms === 'gpu') return await this.gpuDecoder.run(boxes.toGpuBuffer(), scores.toGpuBuffer())
      // JS path: read all 2016 × 19 raw values back, decode on the CPU.
      const [boxData, scoreData] = await this.readback.read([
        { buffer: boxes.toGpuBuffer(), bytes: 2016 * 18 * 4 },
        { buffer: scores.toGpuBuffer(), bytes: 2016 * 4 },
      ])
      return decodeAndSuppress(boxData, scoreData, this.anchors, 1)
    } finally {
      boxes.delete()
      scores.delete()
    }
  }

  result({ palm = null, rect = null, presence = 0, hand = null, timings }) {
    timings.total = Object.values(timings).reduce((a, b) => a + b, 0)
    return { detected: !!palm, palm, rect, presence, hand, timings }
  }

  // For visualising what the models see (tensor-view.js).
  get inputBuffers() {
    return { palm: this.palmInput.buffer, crop: this.cropInput.buffer }
  }

  reset() {
    this.trackedRect = null
  }
}
