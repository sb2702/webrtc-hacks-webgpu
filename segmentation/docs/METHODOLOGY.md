# Methodology

What the benchmark measures, the decisions behind it, and what it can't tell
you. Organised around the questions people ask when they see the numbers.

## The workload

- **Model**: MediaPipe Selfie Segmenter (`selfie_segmenter.tflite`, the
  official `float16/1` file — see the README for URL and hash). MobileNetV3-style
  encoder, 3-stage decoder, ~106k parameters. Input 256×256 RGB in [0, 1],
  stretched from the frame (not letterboxed, as MediaPipe does it); output a
  256×256 foreground probability.
- **Effect**: replace the background with solid green: `mix(green, frame, mask)`
  with the mask bilinearly upsampled to frame size.
- **Clip**: `web/public/media/woman_on_video_call.mp4` — 8 s, 192 frames, 24 fps,
  H.264 High, 1280×720 — and a 640×360 re-encode of it
  (`ffmpeg -vf scale=640:360:flags=lanczos -c:v libx264 -profile:v high -crf 20 -g 48`).

## What exactly is timed?

`web/src/bench.js`. One **run** is one pass over the clip:

```
cached EncodedVideoChunks → VideoDecoder → pipeline.process(frame) → VideoEncoder (H.264) → MP4 muxer
└──────────────────────────────── timed: first chunk in → muxer finalised ────────────────────────────────┘
```

All stages are Streams (`webcodecs-utils`' `VideoDecodeStream`,
`VideoProcessStream`, `VideoEncodeStream`) with their default backpressure,
so decoding, processing and encoding overlap like they would in a real
transcoder. **FPS = 192 frames / wall-clock time of the run.**

Not timed:

- **Demuxing.** The file is demuxed once; every run replays the same cached
  chunks.
- **Pipeline creation** — downloading the model, compiling shaders or the
  WebNN graph, LiteRT / MediaPipe initialisation. Reported separately as
  `initMs`. (It varies a lot: ~3 ms for our WebGPU port, 0.2–5 s for LiteRT
  and MediaPipe.)
- **The warm-up run.** Each pipeline does one untimed pass first (shader
  compilation on first use, encoder start-up, GPU clocks ramping). Its FPS is
  in the JSON as `warmupFps`.

Each pipeline then does N timed runs (default 5). We report the **median**,
plus min and max; every individual run is in the JSON report.

## Why include the encoder? Isn't that measuring the encoder?

Two reasons:

1. **It's the real workload.** An effect in a WebRTC sender or an export
   pipeline feeds an encoder; measuring the effect in isolation hides
   integration costs (getting a `VideoFrame` back out of a canvas, readbacks,
   synchronisation).
2. **It's an honest completion signal.** GPU APIs are asynchronous — "process()
   returned" means work was submitted, not done. The encoder must read every
   output frame's pixels, so the GPU can't fall behind without the pipeline
   stalling. That forces completion without inserting an artificial
   `await gpu idle` after every frame, which would serialise the pipeline and
   measure latency instead of throughput.

The cost is a **ceiling**: the `passthrough` pipeline (decode → encode, no
effect) measures it. At 720p on the M4 it was ~425 fps, close enough to the
fastest pipelines to compress their differences — so the main numbers are at
**360p**, where the ceiling is ~1,150 fps (2× the fastest pipeline). The model
always runs at 256×256, so lower resolution shrinks decode / encode /
compositing cost, not inference.

`--mode process` is also available: decode → process → wait for the output's
pixels (a 1×1 `drawImage` + `getImageData` on the output frame), one frame at
a time, no encoder. That measures per-frame latency; it's in the notes but not
the headline numbers.

## Is the comparison fair?

Same everywhere:

- The **same `.tflite` file** is loaded by LiteRT.js and MediaPipe. Our ports
  (WebGPU / WebGL / WebNN) use weights exported from it by
  `tools/export_weights.py` (fp16 → fp32, layouts unchanged) — regenerating
  them gives byte-identical files.
- The **same frames, same encoder settings** (H.264 via WebCodecs, codec
  string and bitrate from `webcodecs-utils`' `getCodecString` / `getBitrate`,
  keyframe every 60 frames, browser-chosen hardware acceleration — recorded in
  each JSON report).
- The **same effect** and the same `process(VideoFrame) → VideoFrame` contract.

Different, deliberately — each pipeline gets the best I/O its runtime allows:

| Pipeline | Pre-processing (resize to 256²) | Tensor path | Composite |
|---|---|---|---|
| Ours — WebGPU | our WebGPU compute shader | stays in GPU buffers; one command buffer per frame | our WebGPU render pass |
| Ours — WebGL2 | our WebGL shader | stays in textures | our WebGL draw |
| LiteRT — WebGPU | our WebGPU shader, into a buffer LiteRT wraps (shared `GPUDevice`) | zero-copy in and out | our WebGPU |
| LiteRT — WASM | our WebGPU shader | **read back** to CPU, mask **uploaded** | our WebGPU |
| WebNN (all) | our WebGPU shader | **read back** to CPU, mask **uploaded** (see below) | our WebGPU |
| MediaPipe — GPU | **MediaPipe's own** | stays in its WebGL context; mask returned at **frame resolution** | our WebGL shader, in MediaPipe's context |
| MediaPipe — CPU | **MediaPipe's own** | mask returned as a frame-resolution `Float32Array`, uploaded | our WebGPU |

- **WebNN reads back through the CPU** because WebNN ↔ WebGPU buffer sharing
  doesn't work for this model yet: Chrome 156 on macOS exposes
  `createExportableTensor` / `exportToGPU`, but exportable tensors are
  float16-only and shape-limited (`[1,256,256,1]` and `[1,256,256,3]` are
  rejected). The `webnn-interop` pipeline implements the zero-copy path and is
  off by default until browsers allow it.
- **MediaPipe's pre/post-processing can't be swapped out**, and it upsamples
  the mask to the frame's full resolution, which costs more than our 256×256
  mask. That is part of what you get by using MediaPipe, so we count it.

Precision:

| Pipeline | Precision |
|---|---|
| Ours — WebGPU / WebGL | fp32 activations and weights |
| LiteRT — WebGPU | fp32 by default (an fp16 variant is included) |
| LiteRT — WASM, MediaPipe — CPU | XNNPACK's choice (fp32 by default) |
| WebNN | fp32 and fp16 variants, chosen when building the graph |
| MediaPipe — GPU | not exposed by the API |

Threads: LiteRT loads its multi-threaded WASM build (the dev server sends
COOP/COEP headers so `SharedArrayBuffer` is available — check
`crossOriginIsolated` in the report). MediaPipe's CPU delegate and WebNN's CPU
path use whatever their runtimes do. Everything runs on the page's main
thread; no pipeline uses a worker.

## How do we know every pipeline produces the right output?

Two checks, and we didn't trust a timing until both passed:

1. **`/test.html` (and `npm test`)** feeds a fixed 256×256 input to our WebGPU,
   WebGL and WebNN ports and compares the raw mask with the TFLite
   interpreter's output for the same input (`tools/selfie_numpy.py` writes the
   goldens). fp32 ports match to ≤1e-5. WebNN fp16 differs by ~1e-4 on average
   (~0.04–0.07 max, at mask edges) — expected for half precision.
2. **Check outputs** on `/bench.html` (`npm run bench -- --check`) runs every
   pipeline end-to-end on one decoded frame and reports the share of green
   pixels and the per-pixel agreement with our WebGPU pipeline. On the M4 all
   pipelines agreed on ≥99.5% of pixels — no flipped, inverted or empty
   masks. Thumbnails are shown for a visual check.

That's how the WebNN GPU bug on Linux / Intel (below) was caught: it was fast
and wrong.

## Did you tune your own port to win?

No, if anything the opposite. Our WebGPU and WebGL ports are deliberately
naive: one shader per layer type, one dispatch / draw per layer (92 / 102 per
frame), no layer fusion, no fp16, no tiling or shared-memory convolution. The
two backends share the graph code, the weight packing and the algorithms, so
WebGPU vs WebGL compares the APIs, not the effort.

There is exactly one optimisation, and it was a bug fix. The first version
computed the global average pool (10 per frame) with one thread per channel
group looping over up to 128×128 pixels. `/profile.html` (GPU timestamp
queries around every dispatch) showed it taking **81% of GPU time** (5.8 of
7.1 ms on the M4). Both backends were then given the best reduction their API
allows:

- WebGPU: one 256-thread workgroup per channel group with a shared-memory tree
  reduction → 0.24 ms.
- WebGL: no shared memory, so two passes (row means, then the mean of the
  column).

| M4, 720p transcode | before fix | after |
|---|---|---|
| Ours — WebGPU | 150 fps | 345 fps |
| Ours — WebGL2 | 85 fps | 145 fps |

That difference — workgroup shared memory exists in WebGPU and not in WebGL —
is part of the point of the article, which is why both versions are reported.

Run `/profile.html` on your machine to see where our WebGPU port spends its
time. LiteRT.js's WebGPU delegate matching our naive port on the M4 doesn't
mean it can't be beaten; on Intel Iris Xe it was 1.6× faster than ours.

## Which chip actually ran it? (WebNN)

WebNN takes a device hint (`cpu`, `gpu`, `npu`) but nothing in the API reports
where a graph actually ran. We attributed it externally with
`/sustained.html`: each pipeline runs flat out for 20 s with 12 s idle gaps and
a coloured banner, while a system monitor (`macmon` on macOS) watches CPU,
GPU and ANE (Apple Neural Engine) activity.

Observed on the M4, Chrome 156:

| WebNN graph | FPS (360p) | ANE active |
|---|---|---|
| `npu` hint, fp16 | 550 | yes |
| `gpu` hint, fp16 | 551 | yes |
| `gpu` hint, fp32 | 206 | no |
| `cpu`, fp32 | 266 | no |

- The `gpu` and `npu` hints produce bit-identical output and the same timing:
  in Chrome on macOS, **the hint doesn't choose the hardware; precision does**.
  Core ML puts fp16 graphs on the Neural Engine; the ANE can't run fp32.
- The GPU is also busy during the fp16 windows: our WebGPU resize and
  composite run every frame, and Core ML may keep parts of the graph on the
  GPU. These can't be separated from inside the browser.

## Known limitations

- **One machine, one browser version.** The published numbers are from an
  Apple M4 MacBook on Chrome 156. Ranking changes on other hardware: on an
  Intel Iris Xe laptop (Linux, Chrome 154) our naive WebGPU port was *slower*
  than our WebGL port (68 vs 116 fps), LiteRT's WebGPU delegate ran at 112,
  and WebNN's GPU path produced an incorrect mask. Please run it yourself.
- **Background load matters.** During the original runs the MacBook also
  picked up unrelated background jobs for a while; affected numbers dropped by
  up to 40% (e.g. 550 → 332 fps) and were thrown away. Close other apps, plug
  in power, check the run-to-run spread in the report, and prefer
  `--order interleaved` so drift hits every pipeline equally rather than
  whichever ran last. (The published M4 numbers used `sequential` order; most
  pipelines' runs were within ~3% of each other, LiteRT — WASM and WebNN —
  GPU fp32 within ~10%.)
- **Short clip.** 192 frames is ~0.2–2 s per run. Five runs give tight spreads
  on a quiet machine, but sustained thermal behaviour (laptops throttling over
  minutes) isn't captured — `/sustained.html` is closer to that.
- **GPU clocks.** Apple and laptop GPUs ramp clocks with load; the first timed
  runs after a warm-up can still differ. Look at `runs` in the JSON, not only
  the median.
- **Headless browsers are not used for timings.** On Linux, Chrome's WebGPU
  silently falls back to SwiftShader (a CPU emulation) unless Vulkan is
  enabled (`--enable-features=Vulkan`), and headless additionally needs
  `--use-angle=vulkan` — which removes WebGL2 and makes
  `new VideoFrame(webgpuCanvas)` read back blank. `npm run bench` always
  launches a visible window without that flag; `npm test` (pure compute) runs
  headless. Always check the GPU line at the top of `/bench.html` (or
  `environment.webgpu` in the JSON) — `swiftshader` means the numbers are CPU
  emulation.
- **WebNN is behind a flag** and its behaviour is changing between Chrome
  releases (device selection, interop, op coverage).
- **Pinned library versions**: `@litertjs/core` 2.5.3,
  `@mediapipe/tasks-vision` 1.1.1, `webcodecs-utils` 0.2.6 (exact versions in
  `package-lock.json`). Newer releases may be faster or slower.
- **Not explained** (reported as observed):
  - WebNN fp32 on the GPU was faster at 720p (278 fps) than at 360p (219),
    consistently.
  - LiteRT's fp16 option made no difference in transcode, though it was ~20%
    faster per frame in `process` mode.
  - Our WebGPU port and both LiteRT WebGPU variants cluster at 340–370 fps on
    the M4. It isn't the shared output path — WebNN fp16 uses the same one and
    reaches 548.

## Reporting results

A results file should come from `npm run bench` (or Download JSON on
`/bench.html`) without edits. It contains the config (mode, order, runs,
resolution, encoder settings), the environment (date, user agent, OS version,
CPU count, GPU adapter and features, cross-origin isolation, WebNN
availability) and every individual run. Say whether the machine was on
battery and what else was running.
