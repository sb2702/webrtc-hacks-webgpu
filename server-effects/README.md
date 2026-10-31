# Server-side video effects with WebGPU (Dawn + Node)

One of the demos for a webrtcHacks article on WebGPU for real-time video
([all demos](../README.md)).

The browser background-replacement effect from [`../segmentation`](../segmentation),
running on a server: the **same** WGSL shaders and the **same** selfie
segmentation network code, executed by [Dawn](https://dawn.googlesource.com/dawn)
(Chrome's WebGPU implementation) through the [`webgpu`](https://www.npmjs.com/package/webgpu)
npm package, with ffmpeg doing decode and encode.

```
ffmpeg decode ─▶ RGBA bytes ─▶ writeTexture ─▶ [ resize → selfie network → composite ] ─▶ copyTextureToBuffer ─▶ RGBA bytes ─▶ ffmpeg encode
                                    (CPU → GPU)        one GPU submit, browser shaders           (GPU → CPU, mapAsync)
```

## Run it

Requirements: Node 20+, `ffmpeg` and `ffprobe` on the PATH, and a GPU with
Metal (macOS), Vulkan (Linux) or D3D12 (Windows). The `webgpu` package ships
prebuilt Dawn binaries for all three, so there's nothing to compile.

It imports the network and shaders from `../segmentation`, so clone the whole repo:

```sh
cd server-effects
npm install
npm run demo        # the sample clip → output.mp4 with a green background

node effect.mjs input.mp4 output.mp4 --color 1e90ff               # any background colour
node effect.mjs input.mp4 output.mp4 --encoder h264_videotoolbox  # hardware encode on a Mac
node effect.mjs input.mp4 output.mp4 --passthrough                # no effect: the decode/encode ceiling
```

The input can be anything ffmpeg reads: a file, or a live RTMP / SRT / RTSP
stream.

## What's shared with the browser, and what isn't

| | Browser (`segmentation/`) | Server (here) |
|---|---|---|
| Selfie network (`selfie/model.js`, `selfie/webgpu.js`) | ✓ | same files, imported |
| Resize-to-tensor and composite shaders (`frames/webgpu.js`) | `texture_external` variant | same shaders, `texture_2d` variant |
| WebGPU constants (`GPUBufferUsage`, …) | `window` | `Object.assign(globalThis, globals)` from `webgpu` |
| Frame in | `VideoFrame` → `importExternalTexture` (no copy) | RGBA bytes → `queue.writeTexture` (upload) |
| Frame out | canvas → `new VideoFrame(canvas)` (no copy) | `copyTextureToBuffer` + `mapAsync` (readback) |
| Decode / encode | WebCodecs | ffmpeg over pipes |

Dawn has no `VideoFrame`, `importExternalTexture` or canvas, so each frame
costs an upload and a readback. To hide that latency,
[`src/effect.mjs`](src/effect.mjs) keeps up to 3 frames in flight, each with
its own staging buffer: frame N+1 is uploaded and processed while frame N is
still being read back.

The changes needed in the browser code were small: `frames/webgpu.js` now
exports its shaders, and `selfie/weights.js` gained `parseWeights()` so the
weights can be read from disk instead of `fetch`ed.

## Code

| File | |
|---|---|
| [`effect.mjs`](effect.mjs) | CLI: probe, decode, process with frames in flight, encode |
| [`src/gpu.mjs`](src/gpu.mjs) | Dawn device setup, WebGPU globals |
| [`src/effect.mjs`](src/effect.mjs) | `ServerEffect`: upload, one submit per frame (browser shaders + network), readback |
| [`src/ffmpeg.mjs`](src/ffmpeg.mjs) | ffmpeg decode to / encode from raw RGBA pipes |

## Does it work?

Checked on two machines (2026-10-31): the output matches the browser
pipeline's (person kept, background replaced). One run each, over the
192-frame sample clip, wall clock from first decoded frame to encoder exit.

| Machine | Backend | Clip | Effect | Passthrough |
|---|---|---|---|---|
| Apple M4 MacBook, macOS 26.5 | Metal | 360p, libx264 | 743 fps | 1,042 fps |
| | | 720p, libx264 | 289 fps | 331 fps |
| | | 720p, VideoToolbox | 318 fps | 315 fps |
| Linux desktop, RTX 5090 | Vulkan | 360p, libx264 | 839 fps | 1,017 fps |
| | | 720p, libx264 | 156 fps | 210 fps |

These are smoke-test numbers, not a benchmark: single runs, and the Mac was
also running its normal background workload. They show the shape:

- **The CPU↔GPU copies aren't the bottleneck at these sizes.** Passthrough
  still uploads and reads back every frame, and runs at 1,000+ fps at 360p.
- **At 720p the encoder is the ceiling.** With VideoToolbox on the Mac, the
  effect runs as fast as passthrough does.

The Linux 720p numbers are lower than the Mac's because the CPU side
(decode + libx264) is slower there; its passthrough is lower too.

## Licence

Code: MIT (see [`LICENSE`](../LICENSE)). The model is MediaPipe's selfie
segmenter (Apache 2.0); see [`../segmentation`](../segmentation#model).
