# Hand gesture recognition on WebGPU, from just the model files

One of the demos for a webrtcHacks article on WebGPU for real-time video
([all demos](../README.md)).

A small, readable reference for building a real-time ML video pipeline in the
browser when all you have is a model, not a framework. It runs Google's
MediaPipe **gesture recognizer** models with [LiteRT.js](https://ai.google.dev/edge/litert/web),
and does everything around the models (the part MediaPipe would normally do
for you) by hand, on WebGPU:

```
VideoFrame ─▶ letterbox 192² ─▶ palm detector ─▶ decode + NMS ─▶ rotated crop 224² ─▶ landmark model ─▶ gesture
             (WGSL)            (LiteRT/WebGPU)   (WGSL)          (WGSL)               (LiteRT/WebGPU)   (WASM + JS)
                                                                    ▲                       │
                                                                    └── tracking: next frame's crop from these landmarks
```

- **Zero-copy input.** The `VideoFrame` is imported as a GPU texture and resized
  by a compute shader straight into the GPUBuffer that LiteRT uses as the
  model's input tensor. No canvas, no `getImageData`, no CPU copy.
- **Model output stays on the GPU.** The palm detector's raw output (2016
  candidate boxes, ~150 KB) is decoded and non-maximum-suppressed by one WGSL
  workgroup reading LiteRT's output buffers in place. Only the final detection
  (~80 bytes) is read back. A plain-JS version sits alongside it for
  comparison, and you can switch between them live.
- **Rotated hand crop**: the same warp shader as the resize, just a different
  matrix.
- **Tracking**: once a hand is found, each frame's crop comes from the previous
  frame's landmarks and the palm detector is skipped.

Everything is checked against MediaPipe's own `GestureRecognizer` running the
same `.task` file ([below](#is-it-right)).

**How it works, stage by stage:** [`docs/PIPELINE.md`](docs/PIPELINE.md)

## Run it

Requirements: Node 20+ and Chrome or Edge (WebGPU, plus `MediaStreamTrackProcessor` for the live demo).

```sh
cd gesture
npm install        # also copies LiteRT's and MediaPipe's WASM files into public/vendor
npm run dev        # http://localhost:5174
```

| Page | What it does |
|---|---|
| `/` | Live demo: camera, or a video / image file. Shows the landmarks, the crop the landmark model was given, both model inputs as images (drawn straight from the GPU tensors), and per-stage timings. Toggle GPU vs JS NMS and tracking. |
| `/check.html` | Runs a set of hand photos (and frames from your camera) through our pipeline and MediaPipe's, and compares them. |

Command-line check (headless Chrome; set `CHROME=/path/to/chrome` to use another build):

```sh
npm run fetch-test-images   # downloads 7 hand photos from MediaPipe's public test assets
npm test
```

On Linux, open the pages in a Chrome started with `--enable-features=Vulkan`,
or WebGPU may silently run on SwiftShader (a CPU emulation).

To try it on a phone or another machine, the page needs HTTPS (or localhost):

```sh
ALLOWED_HOSTS=your-tunnel.example.com npm run dev
ngrok http 5174 --url your-tunnel.example.com      # or cloudflared, etc.
```

## Code

The pipeline is about 870 lines including comments, in the order a frame goes through it:

| File | Stage |
|---|---|
| [`src/recognizer.js`](src/recognizer.js) | The whole pipeline: `await recognizer.process(videoFrame)` |
| [`src/task-bundle.js`](src/task-bundle.js) | Reads the four `.tflite` models out of MediaPipe's `.task` file (it's a zip) |
| [`src/warp.js`](src/warp.js) | Stages 1 and 3: `VideoFrame` → model input tensor (letterbox resize, rotated crop) in WGSL |
| [`src/palm.js`](src/palm.js) | Stage 2: SSD anchors, box decoding, weighted NMS, in JS and in WGSL |
| [`src/rects.js`](src/rects.js) | Palm detection → hand crop, landmarks → next frame's crop (rotation maths) |
| [`src/landmarks.js`](src/landmarks.js) | Stage 4: landmark model output → points on the frame |
| [`src/gestures.js`](src/gestures.js) | Stage 5: gesture embedder (LiteRT WASM) + classifier (plain JS) |
| [`src/gpu.js`](src/gpu.js) | Device setup, batched readback |
| [`src/demo.js`](src/demo.js), [`src/draw.js`](src/draw.js), [`src/tensor-view.js`](src/tensor-view.js) | Live demo, overlays, drawing GPU tensors to a canvas |
| [`src/check.js`](src/check.js) | The comparison against MediaPipe |
| [`tools/export_classifier.py`](tools/export_classifier.py) | Exports the 8-gesture classifier's weights to JSON |

Using it from your own code:

```js
import { HandGestureRecognizer } from './src/recognizer.js'

const recognizer = await HandGestureRecognizer.create()
const result = await recognizer.process(videoFrame) // you still own (and close) the frame
if (result.hand) {
  result.hand.gesture       // { name: 'Thumb_Up', score: 0.93, all: [...] }
  result.hand.landmarks     // 21 × { x, y, z }, x / y normalised to the frame
  result.hand.handedness    // { name: 'Left' | 'Right', score }, assuming a mirrored (selfie) image
}
```

It tracks one hand. Gestures: `None`, `Closed_Fist`, `Open_Palm`,
`Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

## Is it right?

Our pipeline re-implements MediaPipe's C++ pre/post-processing, so the test is
whether it gives MediaPipe's answers. `check.html` runs each image through:

- ours, with the WGSL NMS and with the JS NMS;
- MediaPipe's `GestureRecognizer` (tasks-vision 1.1.1), with the same
  `.task` file, on its CPU delegate (the reference) and its GPU delegate.

On MediaPipe's 7 test photos (Chrome 151, Linux):

| | Result |
|---|---|
| Gesture | same as MediaPipe on 7 / 7 |
| Handedness | same on 7 / 7 |
| WGSL vs JS NMS | identical to ~1e-7 (float rounding) |
| Landmarks, ours vs MediaPipe CPU | mean distance 0.4–1.3% of the image size (2–5 px on ~400 px photos) |
| Landmarks, MediaPipe GPU vs MediaPipe CPU | 0.3–0.7% |
| Gesture score | within 0.07 of MediaPipe's |

The remaining landmark difference is honest-to-goodness unexplained. Things
ruled out: colour handling (a solid-colour `VideoFrame` comes through the
external texture with exact RGB values), border handling, a second tracking
pass, and resampling sensitivity (rescaling the input by ±3% moves *our*
landmarks by ≤0.24%). It doesn't change any result. `npm test` fails if a
gesture or handedness differs or mean landmark error exceeds 2%. Use
"Add a camera frame" on `/check.html` to compare on your own hands.

## Gotchas found along the way

These cost time and aren't written down elsewhere (as of LiteRT.js 2.5.3):

- **LiteRT's WebGPU output buffers are bigger than the tensor.** A `[1, 2016, 18]`
  output comes back in a buffer sized for 20 channels, a `[1, 2016, 1]` one
  for 4. The data at the front is plain row-major; the rest is padding. So
  shaders can read them directly, but don't use `buffer.size` as the element count.
- **LiteRT returns new output GPUBuffers on every `run()`.** Bind groups that
  reference them must be created per call.
- **The gesture embedder doesn't compile on LiteRT's WebGPU backend**, and the
  classifier's dynamic batch dimension (`[-1, 128]`) isn't accepted on either
  backend. Both are tiny, so: WASM for one, 20 lines of JS for the other.
- **MediaPipe `.task` files are zips with padding before each entry**, so walk
  the central directory, not the local headers.
- **The landmark model's handedness output is P(right)**, not P(left): MediaPipe
  maps index 0 to "Right" in its graph config, not in the model metadata.

## Models and licences

`public/models/gesture_recognizer.task` is Google's MediaPipe Gesture
Recognizer, unmodified, licensed under the Apache License 2.0 (see
[MediaPipe's gesture recognizer docs](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)
for the model card):

```
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
sha256 97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482
```

`public/models/gesture_classifier.json` is derived from it by
`tools/export_classifier.py` (which also checks the exported formula against
the TFLite interpreter). The pre/post-processing is a port of MediaPipe's
calculators (Apache 2.0); source files are named in the comments.

The test images are MediaPipe's and are downloaded, not committed.

Code: MIT (see [`LICENSE`](../LICENSE)).
