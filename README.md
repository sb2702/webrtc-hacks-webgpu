# Selfie segmentation in the browser: WebGPU vs WebGL vs WebNN vs LiteRT.js vs MediaPipe

The companion repo for a webrtcHacks article on WebGPU for real-time video.

It takes one small, widely deployed model — Google's MediaPipe **selfie
segmenter** — and runs the same background-replacement effect through every
way a web app can run it today, on the same frames, with the same
pre/post-processing where possible:

| Pipeline | Model runs on | Notes |
|---|---|---|
| **Ours — WebGPU** | hand-written WGSL compute shaders | ~350 lines + a 70-line model graph, no ML framework |
| **Ours — WebGL2** | hand-written GLSL fragment shaders | same graph, same weights |
| **Ours — WebNN** | the browser's `MLGraphBuilder` → OS ML stack (Core ML, DirectML, …) | CPU / GPU / NPU hints, fp32 / fp16 |
| **LiteRT.js — WebGPU** | Google's TFLite web runtime, WebGPU delegate | zero-copy GPU buffers in and out |
| **LiteRT.js — WASM** | LiteRT's XNNPACK CPU build | WebGPU pre/post, CPU round trip |
| **MediaPipe — GPU** | MediaPipe Tasks `ImageSegmenter`, GPU delegate (WebGL) | MediaPipe's own pre/post-processing |
| **MediaPipe — CPU** | MediaPipe Tasks `ImageSegmenter`, CPU delegate (WASM) | |
| Passthrough | — | decode → encode with no effect: the ceiling |

Every pipeline has the same interface — `process(VideoFrame) → VideoFrame` —
so the benchmark is a real WebCodecs transcode (decode → effect → H.264 encode
→ mux), and the live demo is the WebRTC insertable-streams pattern
(`MediaStreamTrackProcessor` → effect → `MediaStreamTrackGenerator`).

**Results:** [`results/`](results/) · **How it's measured and what it doesn't
tell you:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)

## Headline numbers (Apple M4 MacBook, Chrome 156)

360p and 720p transcode, 192 frames, median of 5 runs after a warm-up. Full
table, run-to-run spread and caveats in
[`results/2026-10-28-apple-m4.md`](results/2026-10-28-apple-m4.md).

| Pipeline | 360p FPS | 720p FPS |
|---|---|---|
| Passthrough (decode → encode only) | 1,158 | 425 |
| WebNN, fp16 (runs on the Neural Engine) | 548 | 426 |
| Ours — WebGPU | 358 | 369 |
| LiteRT.js — WebGPU | 359 | 343 |
| LiteRT.js — WASM | 334 | 286 |
| WebNN — CPU | 266 | 240 |
| WebNN — GPU, fp32 | 219 | 278 |
| Ours — WebGL2 | 153 | 145 |
| MediaPipe — GPU (WebGL) | 147 | 136 |
| MediaPipe — CPU (WASM) | 143 | 103 |

One machine, one browser, one model. Run it on yours: numbers on other
hardware differ a lot (see the Intel notes in the results file).

## Run it

Requirements: Node 20+, a Chromium-based browser (Chrome 113+ for WebGPU;
WebNN needs Chrome with the WebNN flag — `chrome://flags`, search "WebNN").

```sh
git clone <this repo> && cd <this repo>
npm install          # also copies the LiteRT / MediaPipe / web-demuxer WASM into web/public/vendor
npm run dev          # http://localhost:5173
```

Pages:

| URL | What it does |
|---|---|
| `/` | Live demo: camera or the sample video → any pipeline → `<video>` |
| `/test.html` | Correctness: our WebGPU / WebGL / WebNN ports vs the TFLite reference output |
| `/bench.html` | The benchmark. Run **Check outputs** first, then **Run benchmark**, then **Download JSON** |
| `/sustained.html` | Long labelled load windows per pipeline, to attribute work to CPU / GPU / NPU with a system monitor |
| `/profile.html` | GPU time per layer type for our WebGPU network (timestamp queries) |

Or from the command line, which launches your local Chrome (headed — see
methodology for why) and saves a JSON report to `results/`:

```sh
npm test                                      # correctness, headless
npm run bench -- --check                      # sanity-check every pipeline's output
npm run bench                                 # 360p transcode, 5 runs, all pipelines
npm run bench -- --sample 720p --order interleaved --runs 7
npm run bench -- --pipelines webgpu,webgl --mode process
```

The runners use your installed Google Chrome (set `CHROME=/path/to/chrome` for
another build) and enable WebNN for you. On Linux they also enable Vulkan;
when opening the pages in your own Chrome on Linux, start it with
`--enable-features=Vulkan` and check the GPU line on `/bench.html` — if it says
`swiftshader`, WebGPU is running on a CPU emulation and the numbers are
meaningless. Don't add `--use-angle=vulkan` to a visible browser: it disables
WebGL2 and breaks WebGPU canvas capture.

### Testing on another device (phone, laptop)

The pages need a secure context (HTTPS or localhost) for WebGPU / WebCodecs.
Expose the dev server through a tunnel and allow its hostname:

```sh
ALLOWED_HOSTS=your-tunnel.example.com npm run dev
ngrok http 5173 --url your-tunnel.example.com   # or cloudflared, etc.
```

### Submitting results

Run `npm run bench` (or Download JSON from `/bench.html`) on a quiet, plugged-in
machine and open a PR adding the file to `results/`. The JSON records the
browser, OS, GPU adapter, every individual run and the exact encoder settings.

## Repository layout

```
web/src/selfie/model.js     the network, written once against an 8-op backend interface
web/src/selfie/webgpu.js    WebGPU backend: one WGSL compute shader per layer type
web/src/selfie/webgl.js     WebGL2 backend: one GLSL fragment shader per layer type
web/src/selfie/webnn.js     WebNN backend: MLGraphBuilder ops, fp32 or fp16
web/src/selfie/weights.js   weight loading + vec4/mat4 packing for the shader backends
web/src/frames/             VideoFrame → model input, mask → VideoFrame (WebGPU and WebGL)
web/src/pipelines.js        every pipeline as process(VideoFrame) → VideoFrame
web/src/bench.js            the benchmark harness (webcodecs-utils streams)
web/public/models/          selfie_segmenter.tflite + the weights exported from it
web/public/media/           sample clip, 720p and 360p
web/public/test_data/       golden input / TFLite output for the correctness test
tools/tflite_graph.py       minimal .tflite reader: prints the graph with wiring and options
tools/export_weights.py     .tflite → selfie_segmenter.{json,bin} (fp32)
tools/selfie_numpy.py       the network in plain numpy, checked against the TFLite interpreter; writes goldens
tools/bench.mjs, browser_test.mjs   command-line runners (Playwright + your Chrome)
results/                    benchmark reports
docs/METHODOLOGY.md         what is measured, fairness decisions, known limitations
```

### Re-deriving the weights and goldens

Everything under `web/public/models/*.{json,bin}` and `web/public/test_data/`
is generated from the `.tflite` file and is reproducible byte-for-byte:

```sh
pip install -r requirements.txt
python3 tools/tflite_graph.py web/public/models/selfie_segmenter.tflite      # inspect the graph
python3 tools/export_weights.py web/public/models/selfie_segmenter.tflite web/public/models/selfie_segmenter
python3 tools/selfie_numpy.py web/public/test_data/frame.png                 # also checks numpy vs TFLite
```

## Model

`web/public/models/selfie_segmenter.tflite` is Google's MediaPipe Selfie
Segmenter (square, 256×256), unmodified:

```
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite
sha256 191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b
```

See the [MediaPipe image segmentation docs](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
for its model card and license.

## License

Code: MIT (see `LICENSE`). The model and the sample video are not covered by
the MIT license — see above for the model's source.
