# The pipeline, stage by stage

What it takes to go from four `.tflite` files to a working gesture recognizer
on WebGPU, and why each piece is built the way it is. Code references are to
`src/`; MediaPipe source files are named where we port their behaviour.

- [0. What's in the box](#0-whats-in-the-box)
- [1. Zero-copy input: VideoFrame → tensor](#1-zero-copy-input-videoframe--tensor)
- [2. Palm detection: decoding the output, and non-maximum suppression](#2-palm-detection-decoding-the-output-and-non-maximum-suppression)
- [3. The rotated hand crop](#3-the-rotated-hand-crop)
- [4. Landmarks, and reading back only what you need](#4-landmarks-and-reading-back-only-what-you-need)
- [5. Tracking, and the gesture itself](#5-tracking-and-the-gesture-itself)
- [Where the time goes](#where-the-time-goes)

## 0. What's in the box

MediaPipe ships the gesture recognizer as one `gesture_recognizer.task` file.
It's a zip (no compression) holding two more zips holding four models:

| Model | Input | Output |
|---|---|---|
| `hand_detector.tflite` | 192×192 RGB, [0, 1] | 2016 candidate boxes × (box + 7 palm keypoints), 2016 scores |
| `hand_landmarks_detector.tflite` | 224×224 RGB, [0, 1] | 21 landmarks (x, y, z), hand presence, handedness, 21 world landmarks |
| `gesture_embedder.tflite` | landmarks, world landmarks, handedness | 128 numbers |
| `canned_gesture_classifier.tflite` | 128 numbers | 8 gesture probabilities |

`task-bundle.js` pulls them out in ~30 lines; LiteRT.js accepts the bytes
directly. That's all the model files tell you, though. Everything between the
models (how to resize, what the 2016 boxes mean, how to crop the hand, how to
map the landmarks back) lives in MediaPipe's C++ graph, not in the `.tflite`
files. Using "just the model" means re-creating that part, which is most of
this repo.

Both image models run on LiteRT's WebGPU backend, on a `GPUDevice` we create
and hand to LiteRT (`litert.setWebGpuDevice(device)`). Sharing the device is
the key enabler: a `GPUBuffer` our shader writes can be wrapped as a LiteRT
tensor, and a LiteRT output tensor can be bound to our shader, with no copies.

## 1. Zero-copy input: VideoFrame → tensor

`warp.js`, `recognizer.js`

A camera frame arrives as a `VideoFrame` (here from a
`MediaStreamTrackProcessor`, as in any insertable-streams WebRTC pipeline).
The palm detector wants a 192×192 float RGB tensor. The usual web approach is
`drawImage` into a canvas, `getImageData`, a JS loop to convert bytes to
floats, and an upload. That's a GPU→CPU readback, a CPU pass over every pixel,
and a CPU→GPU upload, every frame.

Instead:

```js
// once
const input = device.createBuffer({ size: 192 * 192 * 3 * 4, usage: STORAGE | ... })
const inputTensor = new litert.Tensor(input, [1, 192, 192, 3], 'float32')  // wraps, doesn't copy

// per frame
const texture = device.importExternalTexture({ source: videoFrame })      // the frame's own GPU memory
// compute shader: for each tensor pixel, sample the texture, write 3 floats into `input`
const [boxes, scores] = await palmDetector.run([inputTensor])
```

`importExternalTexture` gives a shader access to the frame where it already
is (including any YUV → RGB conversion) without an extra copy on most
platforms. The compute shader in `warp.js` does the resize, the letterbox
(black bars, so the hand isn't stretched), the colour conversion to [0, 1]
floats and the HWC layout, all in one pass, writing straight into the buffer
LiteRT reads.

The shader is generic: each tensor pixel (u, v) is mapped to a frame position
by a 2×3 affine matrix. For the letterbox it's a scale and an offset
(`letterbox()`); for the hand crop in stage 3 it adds a rotation
(`rectToMatrix()`). Same shader, different matrix.

Ordering needs no synchronisation code: our dispatch and LiteRT's inference go
to the same `device.queue`, and the GPU runs queue submissions in order.

## 2. Palm detection: decoding the output, and non-maximum suppression

`palm.js`

### What the detector actually outputs

The palm detector is an SSD-style detector. It doesn't say "there is a hand
at (x, y)". It has 2016 fixed **anchors**, reference points laid on two grids
over the input (24×24 cells × 2 anchors, plus 12×12 cells × 6), and for every
anchor it outputs:

- a **score** (a logit; sigmoid gives a probability that a palm is centred here);
- a **box**: offset of the palm centre from the anchor, plus width and height;
- **7 keypoints** (wrist, finger bases, …), also as offsets from the anchor.

Decoding is arithmetic: `centre = anchor + offset / 192`, etc. The anchor grid
isn't in the model file; `palmAnchors()` regenerates it from the parameters
in MediaPipe's `hand_detector_graph.cc`.

### Non-maximum suppression (NMS)

A real hand doesn't light up one anchor. It lights up a dozen neighbouring
ones, each predicting nearly the same box. NMS turns that cluster into one
detection:

1. Throw away candidates below the score threshold (0.5).
2. Take the highest-scoring candidate.
3. Find every remaining candidate whose box overlaps it by more than 30%,
   measured as intersection over union (IoU): the area of overlap divided by
   the area covered by either box.
4. Emit one detection for that group, remove the group, go to 2.

MediaPipe uses **weighted** NMS: in step 4 the emitted box and keypoints are
the score-weighted average of the whole group, not just the winner's. That
averages out the noise between anchors, so the box jitters less from frame to
frame.

`decodeAndSuppress()` is the straightforward JS version (~40 lines). To run it,
the CPU first needs the detector's raw output: 2016 × 19 floats, about 150 KB,
read back from the GPU every frame we detect. 99% of it is thrown away.

### NMS in WGSL

`GpuPalmDecoder` does the same thing in one compute dispatch of one 64-thread
workgroup, binding LiteRT's output buffers directly:

1. Each thread computes the sigmoid scores of every 64th anchor into
   workgroup (shared) memory.
2. **Argmax by reduction.** Each thread finds the best score among its
   anchors. Then a tree reduction: threads 0–31 compare themselves with
   32–63, then 0–15 with 16–31, … six steps later thread 0 holds the winner.
   `workgroupBarrier()` between steps keeps everyone in lockstep.
3. Each thread checks its anchors against the winner's box, sums
   score-weighted boxes and keypoints for the overlapping ones, and marks
   them suppressed.
4. A second tree reduction adds up the 64 partial sums; thread 0 divides by
   the total weight and writes the detection.
5. Repeat from 2 for as many detections as wanted (one, here).

Then 80 bytes are read back instead of 150 KB. `check.html` confirms the two
versions agree to ~1e-7. In the demo you can switch between them live.

WGSL details worth knowing: workgroup memory is limited (16 KB guaranteed),
which is why the workgroup is 64 threads rather than 256 (64 × 19 partial
sums + 2016 scores fit). Also, a barrier must be reached by all threads, so the "stop
if the best score is below threshold" branch reads the score through
`workgroupUniformLoad()`, which tells the compiler every thread sees the same
value.

Is 2016 anchors worth a GPU kernel? Not for arithmetic; a CPU does this in
microseconds. The win is not having to wait for and copy the whole output
tensor, and it's the general pattern: reduce on the GPU, read back the answer.

## 3. The rotated hand crop

`rects.js`, `warp.js`

The landmark model was trained on hands that are centred, upright (fingers
up) and filling the frame. So we cut that out of the full-resolution frame:

1. **Rotation.** From the palm keypoints: the angle of the line from the wrist
   (keypoint 0) to the base of the middle finger (keypoint 2), relative to
   straight up. It's computed in pixels, not normalised coordinates, or the
   angle would be wrong on non-square frames.
2. **Size and position.** A palm box only covers the palm, so it's scaled up
   2.6× and shifted half its height towards the fingers (along the rotated
   axis), then made square.
3. **Crop.** `rectToMatrix()` turns the rotated square into the warp matrix,
   and the stage 1 shader samples the *original* frame (not the 192×192
   detector input) into a 224×224 tensor.

```
tensor pixel (u, v)
  → offset from crop centre:   (u/224 − ½, v/224 − ½) × crop size in pixels
  → rotated by the crop angle:  [cos −sin; sin cos] · offset
  → plus the crop centre       = where to sample the frame
```

The palm detector input is low resolution and letterboxed; the crop is taken
from the full frame, so the landmark model gets as much detail as the camera
provides. Both are one dispatch with no CPU involvement beyond computing six
matrix numbers.

The demo draws the crop (dashed square, solid edge = "up") and shows the
224×224 tensor itself: `tensor-view.js` renders the GPU buffer to a canvas
with a fragment shader, so even the debug view doesn't read anything back.

## 4. Landmarks, and reading back only what you need

`landmarks.js`, `gpu.js`

The landmark model outputs four small tensors: 21 landmarks in crop pixels, a
hand-presence probability, a handedness probability, and 21 "world" landmarks
in metres. Together that's 128 floats, which the CPU genuinely needs (to draw them,
classify the gesture and track the hand).

`Readback.read()` copies all four GPU buffers into one staging buffer and maps
it once. The cost of a readback is mostly the round trip (submit, wait for the
GPU to finish, map), not the bytes, so one map of 512 bytes beats four.

Mapping the landmarks back to the frame is the crop in reverse: divide by
224, un-rotate around the centre, scale to the crop size, add the crop
position (`projectLandmarks()`).

If hand presence is below 0.5, there's no hand in the crop: report nothing
and run the palm detector on the next frame.

## 5. Tracking, and the gesture itself

`rects.js`, `gestures.js`

**Tracking.** Palm detection is the expensive, whole-frame stage. Once there
are landmarks, the next frame's crop can come from them instead:
`landmarksToRect()` takes the palm and lower finger joints (fingertips move
too much), finds their bounding box along the hand's own axis, and grows it
2×. Next frame, stages 1–2 are skipped entirely and we go straight to the crop.
The palm detector only runs again if the landmark model reports the hand is
gone. Untick "Tracking" in the demo to see the difference.

**Gesture.** The landmarks are normalised (wrist at the origin, hand scaled
to size 1), so position and distance from the camera don't matter. The
embedder turns landmarks + world landmarks + handedness into 128 numbers. The
classifier maps those to 8 gestures with one scale/shift, ReLU and dense layer.

These two models see 127 floats, not pixels, so there's no reason to use the
GPU: the embedder runs on LiteRT's WASM backend (a fraction of a millisecond), and the classifier
is 20 lines of JS (`classifyEmbedding()`), with its 1k weights exported from
the `.tflite` by `tools/export_classifier.py`. Knowing when *not* to use the GPU is
part of the picture.

## Where the time goes

The demo shows per-stage times. Each one runs from when we start the stage
until its result is back on the CPU, so it includes the GPU work and the wait.
On frames where the hand is tracked, it's one crop + one landmark inference +
one 512-byte readback + the gesture models; the palm detector only runs when
a hand appears.

| Stage | Runs | Work |
|---|---|---|
| Palm detection | only when not tracking | letterbox dispatch, detector, NMS dispatch, 80-byte readback |
| Landmarks | every frame with a hand | crop dispatch, landmark model, 512-byte readback |
| Gesture | every frame with a hand | embedder (WASM) + classifier (JS) |

Two GPU→CPU waits on a detection frame, one on a tracking frame. Numbers
depend heavily on the GPU; measure on your own hardware with the demo.
