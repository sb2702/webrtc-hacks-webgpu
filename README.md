# WebGPU for real-time video: demos and benchmarks

Companion code for a [webrtcHacks](https://webrtchacks.com) article on using
WebGPU (and its neighbours WebGL, WebNN and WASM) for ML effects on live video
in the browser.

Each folder is a project with its own README and dependencies: `cd` into it
and `npm install`. The browser demos run with `npm run dev`; `server-effects/`
is a Node CLI and reuses code from `segmentation/`.

| Folder | What it is | Highlights |
|---|---|---|
| [`segmentation/`](segmentation/) | MediaPipe's selfie segmenter (background replacement) on every browser backend: our own WebGPU, WebGL2 and WebNN ports, LiteRT.js (WebGPU and WASM) and MediaPipe Tasks (GPU and CPU), benchmarked as a real WebCodecs decode → effect → encode transcode | Results on an Apple M4, including WebNN on the Neural Engine; [methodology](segmentation/docs/METHODOLOGY.md) written for skeptics; correctness tests against TFLite |
| [`gesture/`](gesture/) | Hand gesture recognition from just MediaPipe's model files: LiteRT.js for inference, with all the pre/post-processing done by hand on WebGPU | Zero-copy `VideoFrame` → tensor, non-maximum suppression in a WGSL workgroup, rotated hand crop, tracking; [stage-by-stage walkthrough](gesture/docs/PIPELINE.md); checked against MediaPipe's own recognizer |
| [`server-effects/`](server-effects/) | The same background-replacement shaders and network as `segmentation/`, run server-side in Node on Dawn (the `webgpu` npm package), with ffmpeg for decode/encode | Browser WebGPU code reused as-is (two small exports added); frames in flight to hide upload/readback; tested on Metal (M4) and Vulkan |

## Requirements

- Node 20+ (and ffmpeg for `server-effects/`)
- Chrome or Edge for the full set of demos (WebGPU, WebCodecs,
  `MediaStreamTrackProcessor`). WebNN needs Chrome's WebNN flag
  (`chrome://flags`, search "WebNN").
- On Linux, start Chrome with `--enable-features=Vulkan`, or WebGPU may
  silently fall back to a CPU emulation (SwiftShader).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The MediaPipe models are Google's, under
Apache 2.0, unmodified; see each folder's README for sources and checksums,
and for the sample video's terms.
