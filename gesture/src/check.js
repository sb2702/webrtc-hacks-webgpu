// check.html: does our pipeline give the same answers as MediaPipe's?
//
// MediaPipe's GestureRecognizer runs the same four models from the same
// .task file, with its own C++ pre/post-processing. We compare against its CPU
// delegate (resizing done by OpenCV), and also measure how far its GPU
// delegate (resizing done by WebGL) is from its CPU delegate: that gap is what
// resampling alone does to the landmarks, so it's the yardstick for ours.
// Different gestures or handedness would be real bugs; small landmark
// differences are expected.
import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision'
import { HandGestureRecognizer } from './recognizer.js'
import { drawHand, drawRect } from './draw.js'

// Pass criteria. Landmark error is the mean distance between corresponding
// landmarks, as a fraction of the image's larger side.
const MAX_MEAN_LANDMARK_ERROR = 0.02
const MAX_NMS_DIFFERENCE = 1e-4

const status = document.querySelector('#status')
const tbody = document.querySelector('#results tbody')
const rows = []

const fileset = await FilesetResolver.forVisionTasks('/vendor/mediapipe')
const mediapipe = (delegate) =>
  GestureRecognizer.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: '/models/gesture_recognizer.task', delegate },
    runningMode: 'IMAGE',
    numHands: 2, // we track one hand; compare against whichever of MediaPipe's is the same hand
  })
const [ours, mediapipeCpu, mediapipeGpu] = await Promise.all([
  HandGestureRecognizer.create(),
  mediapipe('CPU'),
  mediapipe('GPU').catch((e) => (console.warn('MediaPipe GPU delegate unavailable:', e.message), null)),
])

async function check(name, bitmap) {
  const frame = new VideoFrame(bitmap, { timestamp: 0 })
  const gpu = await ours.process(frame, { tracking: false, nms: 'gpu' })
  const js = await ours.process(frame, { tracking: false, nms: 'js' })
  frame.close()
  const mp = sameHand(mediapipeCpu.recognize(bitmap), gpu.hand)
  const mpGpu = mediapipeGpu && sameHand(mediapipeGpu.recognize(bitmap), gpu.hand)

  const row = { name, ours: summarize(gpu.hand), mediapipe: null, pass: true, notes: [] }
  const fail = (note) => { row.pass = false; row.notes.push(note) }
  if (mp) {
    row.mediapipe = {
      gesture: { name: mp.gestures[0].categoryName, score: mp.gestures[0].score },
      handedness: { name: mp.handedness[0].categoryName, score: mp.handedness[0].score },
    }
  }

  if (!gpu.hand) fail('we found no hand')
  else if (!mp) fail('MediaPipe found no hand, or a different one')
  else {
    row.landmarkError = landmarkError(gpu.hand.landmarks, mp.landmarks, bitmap)
    // Mean offset (ours − MediaPipe) in pixels: a consistent bias would point at a coordinate convention bug.
    row.landmarkBias = ['x', 'y'].map((k) => mean(gpu.hand.landmarks.map((p, i) => (p[k] - mp.landmarks[i][k]) * (k === 'x' ? bitmap.width : bitmap.height))))
    row.worldError = mean(gpu.hand.worldLandmarks.map((p, i) => Math.hypot(p.x - mp.worldLandmarks[i].x, p.y - mp.worldLandmarks[i].y, p.z - mp.worldLandmarks[i].z)))
    if (mpGpu) row.mediapipeGpuVsCpu = landmarkError(mpGpu.landmarks, mp.landmarks, bitmap)
    if (row.landmarkError.mean > MAX_MEAN_LANDMARK_ERROR) fail('landmarks differ')
    if (row.ours.gesture.name !== row.mediapipe.gesture.name) fail('different gesture')
    if (row.ours.handedness.name !== row.mediapipe.handedness.name) fail('different handedness')
  }

  // The two NMS implementations should agree to float precision.
  if (!!gpu.palm !== !!js.palm) fail('GPU and JS NMS disagree on detection')
  else if (gpu.palm) {
    row.nmsDifference = Math.max(...gpu.palm.box.map((v, i) => Math.abs(v - js.palm.box[i])), ...Array.from(gpu.palm.keypoints, (v, i) => Math.abs(v - js.palm.keypoints[i])))
    if (row.nmsDifference > MAX_NMS_DIFFERENCE) fail('GPU and JS NMS differ')
  }

  rows.push(row)
  render(row, bitmap, gpu, mp)
}

function landmarkError(a, b, { width, height }) {
  const size = Math.max(width, height)
  const errors = a.map((p, i) => Math.hypot((p.x - b[i].x) * width, (p.y - b[i].y) * height) / size)
  return { mean: mean(errors), max: Math.max(...errors), pixels: mean(errors) * size }
}

// MediaPipe's hand nearest ours (by wrist position), flattened to a single
// hand, or null if it found none within 10% of the image of ours.
function sameHand(result, hand) {
  if (!hand) return null
  const wrist = hand.landmarks[0]
  let best = null, bestDistance = 0.1
  result.landmarks.forEach((landmarks, i) => {
    const distance = Math.hypot(landmarks[0].x - wrist.x, landmarks[0].y - wrist.y)
    if (distance < bestDistance) {
      bestDistance = distance
      best = { landmarks, worldLandmarks: result.worldLandmarks[i], handedness: result.handedness[i], gestures: result.gestures[i] }
    }
  })
  return best
}

function summarize(hand) {
  return hand && { gesture: { name: hand.gesture.name, score: hand.gesture.score }, handedness: hand.handedness }
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length
const pct = (v) => `${(v * 100).toFixed(2)}%`
const label = (x) => (x ? `${x.name} ${x.score.toFixed(2)}` : '—')

function render(row, bitmap, ourResult, mp) {
  const tr = document.createElement('tr')
  tr.className = row.pass ? 'pass' : 'fail'
  tr.innerHTML = `
    <td><a href="#img-${rows.length}">${row.name}</a></td>
    <td>${label(row.ours?.gesture)} / ${label(row.mediapipe?.gesture)}</td>
    <td>${label(row.ours?.handedness)} / ${label(row.mediapipe?.handedness)}</td>
    <td>${row.landmarkError ? `mean ${pct(row.landmarkError.mean)} (${row.landmarkError.pixels.toFixed(1)} px), max ${pct(row.landmarkError.max)}` : '—'}</td>
    <td>${row.mediapipeGpuVsCpu ? `mean ${pct(row.mediapipeGpuVsCpu.mean)} (${row.mediapipeGpuVsCpu.pixels.toFixed(1)} px)` : '—'}</td>
    <td>${row.worldError !== undefined ? `mean ${(row.worldError * 1000).toFixed(1)} mm` : '—'}</td>
    <td>${row.nmsDifference !== undefined ? row.nmsDifference.toExponential(1) : '—'}</td>
    <td>${row.pass ? 'PASS' : `FAIL: ${row.notes.join(', ')}`}</td>`
  tbody.append(tr)

  const figure = document.createElement('figure')
  figure.id = `img-${rows.length}`
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(bitmap, 0, 0)
  if (ourResult.rect) drawRect(ctx, ourResult.rect)
  if (mp) drawHand(ctx, mp.landmarks, { color: '#ff4081', radius: 6 })
  if (ourResult.hand) drawHand(ctx, ourResult.hand.landmarks, { radius: 3 })
  figure.append(canvas, Object.assign(document.createElement('figcaption'), { textContent: row.name }))
  document.querySelector('#images').append(figure)
}

// The images fetched by `npm run fetch-test-images`.
const list = await fetch('/test_images/index.json').then((r) => (r.ok ? r.json() : []), () => [])
if (!list.length) status.textContent = 'No test images: run `npm run fetch-test-images`, or add a camera frame.'
for (const file of list) {
  status.textContent = `Checking ${file}…`
  const blob = await (await fetch(`/test_images/${file}`)).blob()
  await check(file, await createImageBitmap(blob))
}
const summary = () => `${rows.filter((r) => r.pass).length} / ${rows.length} passed`
if (list.length) status.textContent = summary()
window.checkResults = { rows, pass: rows.length > 0 && rows.every((r) => r.pass) }

let video
document.querySelector('#camera').onclick = async () => {
  if (!video) {
    video = document.createElement('video')
    video.muted = true
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
    await video.play()
  }
  await check(`camera ${rows.length + 1}`, await createImageBitmap(video))
  status.textContent = summary()
}
