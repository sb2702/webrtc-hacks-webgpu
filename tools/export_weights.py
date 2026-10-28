"""Export the MediaPipe selfie segmenter's weights to a flat fp32 blob.

  python3 tools/export_weights.py models/selfie_segmenter.tflite web/models/selfie_segmenter

writes <out>.bin (all weights, fp32, concatenated) and <out>.json, a manifest
keyed by Keras layer name:

  { "conv2d_5": { "kind": "conv", "shape": [72, 1, 1, 16], "w": <offset>, "b": <offset>, ... } }

Weights are kept in TFLite's native layouts — conv: [out, kh, kw, in],
depthwise: [1, kh, kw, channels], transposed conv: [out, kh, kw, in]. Each
backend repacks into whatever layout its shaders want at load time.
"""
import json, sys
import numpy as np
from tflite_graph import Graph

KINDS = {"CONV_2D": "conv", "DEPTHWISE_CONV_2D": "depthwise",
         "Convolution2DTransposeBias": "conv_transpose"}

def main(src, out):
    g = Graph(src)
    blob, manifest = [], {}
    offset = 0  # in floats
    def push(a):
        nonlocal offset
        a = np.ascontiguousarray(a, np.float32).ravel()
        blob.append(a); off = offset; offset += a.size
        return off
    for op in g.ops():
        if op["kind"] not in KINDS: continue
        name = g.name(op["outputs"][0])
        w, b = g.const(op["inputs"][1]), g.const(op["inputs"][2])
        manifest[name] = dict(kind=KINDS[op["kind"]], shape=list(w.shape),
                              w=push(w), b=push(b), **{k: v for k, v in op["opts"].items() if k in ("pad", "stride")})
    np.concatenate(blob).tofile(out + ".bin")
    with open(out + ".json", "w") as f:
        f.write("{\n" + ",\n".join(f' "{k}": {json.dumps(v)}' for k, v in manifest.items()) + "\n}\n")
    print(f"{len(manifest)} layers, {offset} params, {offset * 4 / 1024:.0f} KiB -> {out}.bin")

if __name__ == "__main__":
    main(*sys.argv[1:3])
