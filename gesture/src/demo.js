// index.html: camera (or a file) -> VideoFrames -> HandGestureRecognizer -> overlay.
//
// Frames come from a MediaStreamTrackProcessor, the same insertable-streams
// API a WebRTC app would use to process a camera or a remote track. Each
// VideoFrame is handed to the recognizer, drawn, then closed.
import { HandGestureRecognizer } from './recognizer.js'
import { TensorView } from './tensor-view.js'
import { drawHand, drawPalm, drawRect } from './draw.js'

const $ = (selector) => document.querySelector(selector)
const view = $('#view')
const ctx = view.getContext('2d')
const options = () => ({ nms: $('#nms').value, tracking: $('#tracking').checked })
$('#mirror').onchange = () => view.classList.toggle('mirror', $('#mirror').checked)
view.classList.toggle('mirror', $('#mirror').checked)

$('#gesture-score').textContent = 'Loading models…'
const recognizer = await HandGestureRecognizer.create()
const device = recognizer.device
const palmView = new TensorView(device, $('#palm-input'), recognizer.inputBuffers.palm, 192)
const cropView = new TensorView(device, $('#crop-input'), recognizer.inputBuffers.crop, 224)
$('#gesture-score').textContent = 'Start the camera, or open a file.'

let stopCurrent = () => {}

$('#camera').onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
  play(stream.getVideoTracks()[0], () => stream.getTracks().forEach((t) => t.stop()))
}

$('#file').onchange = async (e) => {
  const file = e.target.files[0]
  if (!file) return
  if (file.type.startsWith('image/')) return showImage(await createImageBitmap(file))
  // A video file becomes a MediaStreamTrack too, so it takes the same path as the camera.
  const video = Object.assign(document.createElement('video'), { src: URL.createObjectURL(file), muted: true, loop: true })
  await video.play()
  play(video.captureStream().getVideoTracks()[0], () => video.pause())
}

async function play(track, cleanup) {
  stopCurrent()
  recognizer.reset()
  let running = true
  // maxBufferSize 1: if we fall behind, drop old frames rather than queue them.
  const reader = new MediaStreamTrackProcessor({ track, maxBufferSize: 1 }).readable.getReader()
  stopCurrent = () => {
    running = false
    reader.cancel()
    track.stop()
    cleanup()
  }
  while (running) {
    const { value: frame, done } = await reader.read()
    if (done) break
    try {
      const result = await recognizer.process(frame, options())
      render(frame, result)
    } finally {
      frame.close()
    }
  }
}

async function showImage(bitmap) {
  stopCurrent()
  const frame = new VideoFrame(bitmap, { timestamp: 0 })
  const result = await recognizer.process(frame, { ...options(), tracking: false })
  render(frame, result)
  frame.close()
}

// ---- display -----------------------------------------------------------------

let lastFrameTime = 0, fps = 0
const smoothed = {}

function render(frame, result) {
  if (view.width !== frame.displayWidth || view.height !== frame.displayHeight) {
    view.width = frame.displayWidth
    view.height = frame.displayHeight
  }
  ctx.drawImage(frame, 0, 0)
  // Overlays and model-input previews only when there's a hand: a palm box or
  // crop that the landmark model rejected is just noise on screen.
  const hand = result.hand
  if (hand) {
    if (result.palm) drawPalm(ctx, result.palm)
    drawRect(ctx, result.rect)
    drawHand(ctx, hand.landmarks)
  }
  palmView.draw({ empty: !hand })
  cropView.draw({ empty: !hand })

  $('#gesture').textContent = hand ? hand.gesture.name.replace('_', ' ') : '—'
  $('#gesture-score').textContent = hand ? `score ${hand.gesture.score.toFixed(2)}` : 'no hand'
  // MediaPipe's handedness assumes a mirrored (selfie) image.
  $('#hand-info').textContent = hand ? `${hand.handedness.name} hand (${hand.handedness.score.toFixed(2)}), presence ${result.presence.toFixed(2)}` : ''

  const now = performance.now()
  if (lastFrameTime) fps = 0.9 * fps + 0.1 * (1000 / (now - lastFrameTime))
  lastFrameTime = now
  for (const [k, v] of Object.entries(result.timings)) smoothed[k] = smoothed[k] === undefined ? v : 0.9 * smoothed[k] + 0.1 * v
  const ms = (k) => (result.timings[k] !== undefined ? `${smoothed[k].toFixed(1)} ms` : '—')
  $('#stats').innerHTML = `
    <dt>Frame</dt><dd>${frame.displayWidth}×${frame.displayHeight}</dd>
    <dt>This frame</dt><dd>${result.detected ? 'palm detector' : result.rect ? 'tracking' : 'no hand'}</dd>
    <dt>Palm detection</dt><dd>${ms('palm')}</dd>
    <dt>Landmarks</dt><dd>${ms('landmarks')}</dd>
    <dt>Gesture</dt><dd>${ms('gesture')}</dd>
    <dt>Total</dt><dd>${ms('total')}</dd>
    <dt>Display rate</dt><dd>${fps.toFixed(0)} fps</dd>`
}
