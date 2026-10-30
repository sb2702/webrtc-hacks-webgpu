"""The selfie segmenter, re-implemented in plain numpy from the exported weights.

This is the spec the WebGPU / WebGL ports follow: same layer names, same op
order. `python3 selfie_numpy.py <image>` checks it against the TFLite
interpreter and writes golden input/output tensors for the browser tests.
"""
import json, sys
import numpy as np

class Weights:
    def __init__(self, path):
        self.m = json.load(open(path + ".json"))
        self.blob = np.fromfile(path + ".bin", np.float32)
    def __getitem__(self, name):
        l = self.m[name]
        n = int(np.prod(l["shape"]))
        return self.blob[l["w"]:l["w"] + n].reshape(l["shape"]), self.blob[l["b"]:l["b"] + l["shape"][0 if l["kind"] != "depthwise" else 3]], l

def act(x, a):
    if a == "relu": return np.maximum(x, 0)
    if a == "hswish": return x * np.clip(x + 3, 0, 6) / 6
    if a == "sigmoid": return 1 / (1 + np.exp(-x))
    return x

def same_pad(n, k, s):
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2

def conv(x, W, name, a="none"):            # x: [H, W, C]
    w, b, l = W[name]                       # w: [O, kh, kw, I]
    O, kh, kw, _ = w.shape
    s = l["stride"][0]
    H, Wd, _ = x.shape
    oh, pt = same_pad(H, kh, s); ow, pl = same_pad(Wd, kw, s)
    xp = np.pad(x, ((pt, kh), (pl, kw), (0, 0)))
    y = np.zeros((oh, ow, O), np.float32) + b
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[ky:ky + oh * s:s, kx:kx + ow * s:s]
            y += patch @ w[:, ky, kx, :].T
    return act(y, a)

def depthwise(x, W, name, a="none"):
    w, b, l = W[name]                       # w: [1, kh, kw, C]
    _, kh, kw, C = w.shape
    s = l["stride"][0]
    H, Wd, _ = x.shape
    oh, pt = same_pad(H, kh, s); ow, pl = same_pad(Wd, kw, s)
    xp = np.pad(x, ((pt, kh), (pl, kw), (0, 0)))
    y = np.zeros((oh, ow, C), np.float32) + b
    for ky in range(kh):
        for kx in range(kw):
            y += xp[ky:ky + oh * s:s, kx:kx + ow * s:s] * w[0, ky, kx]
    return act(y, a)

def conv_transpose_2x2(x, W, name):         # kernel 2, stride 2, VALID: each input pixel paints a 2x2 block
    w, b, _ = W[name]                       # w: [O, 2, 2, I]
    H, Wd, _ = x.shape
    y = np.zeros((H * 2, Wd * 2, w.shape[0]), np.float32)
    for dy in range(2):
        for dx in range(2):
            y[dy::2, dx::2] = x @ w[:, dy, dx, :].T + b
    return y

def pool(x): return x.mean(axis=(0, 1), keepdims=True)

def upsample2x(x):                          # bilinear, half-pixel centers (TF align_corners=False)
    H, Wd, _ = x.shape
    def idx(n):
        src = np.clip((np.arange(n * 2) + 0.5) / 2 - 0.5, 0, n - 1)
        i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, n - 1)
        return i0, i1, (src - i0)[:, None]
    y0, y1, fy = idx(H); x0, x1, fx = idx(Wd)
    rows = x[y0] * (1 - fy[:, :, None]) + x[y1] * fy[:, :, None]
    return rows[:, x0] * (1 - fx[None]) + rows[:, x1] * fx[None]

def squeeze_excite(x, W, a, b):             # channel attention: x * sigmoid(fc(relu(fc(mean(x)))))
    return x * conv(conv(pool(x), W, a, "relu"), W, b, "sigmoid")

def inverted_residual(x, W, expand, dw, project, se=None, a="relu", residual=False):
    y = depthwise(conv(x, W, expand, a), W, dw, a)
    if se: y = squeeze_excite(y, W, *se)
    y = conv(y, W, project)
    return x + y if residual else y

def decoder_block(x, skip, W, proj, se, conv1, dw):
    up = conv(upsample2x(x), W, proj)
    att = conv(conv(pool(skip + up), W, se[0], "relu"), W, se[1], "sigmoid")
    y = conv(skip * att + up, W, conv1, "relu")
    return y + depthwise(y, W, dw, "relu")

def selfie(img, W):                         # img: [256, 256, 3] RGB in [0, 1]
    s1 = conv(img, W, "conv2d", "hswish")                                   # 128² x16
    x = conv(s1, W, "conv2d_1", "relu")
    x = depthwise(x, W, "depthwise_conv2d", "relu")
    x = squeeze_excite(x, W, "conv2d_2", "conv2d_3")
    s2 = conv(x, W, "conv2d_4")                                             # 64² x16
    x = inverted_residual(s2, W, "conv2d_5", "depthwise_conv2d_1", "conv2d_6")          # 32² x24
    s3 = inverted_residual(x, W, "conv2d_7", "depthwise_conv2d_2", "conv2d_8", residual=True)
    x = inverted_residual(s3, W, "conv2d_9", "depthwise_conv2d_3", "conv2d_12", ("conv2d_10", "conv2d_11"), "hswish")  # 16² x32
    for i in range(4):
        e, d, a, b, p = (f"conv2d_{13 + 4 * i}", f"depthwise_conv2d_{4 + i}",
                         f"conv2d_{14 + 4 * i}", f"conv2d_{15 + 4 * i}", f"conv2d_{16 + 4 * i}")
        x = inverted_residual(x, W, e, d, p, (a, b), "hswish", residual=True)
    # LR-ASPP style head: features gated by a global-context branch
    x = conv(x, W, "conv2d_29", "relu") * conv(pool(x), W, "conv2d_30", "sigmoid")  # 16² x128
    x = decoder_block(x, s3, W, "conv2d_31", ("conv2d_32", "conv2d_33"), "conv2d_34", "depthwise_conv2d_8")  # 32²
    x = decoder_block(x, s2, W, "conv2d_35", ("conv2d_36", "conv2d_37"), "conv2d_38", "depthwise_conv2d_9")  # 64²
    x = decoder_block(x, s1, W, "conv2d_39", ("conv2d_40", "conv2d_41"), "conv2d_42", "depthwise_conv2d_10") # 128²
    return act(conv_transpose_2x2(x, W, "segment"), "sigmoid")              # 256² x1

if __name__ == "__main__":
    from PIL import Image
    import os
    img_path = sys.argv[1]
    root = os.path.join(os.path.dirname(__file__), "..")
    img = np.asarray(Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR), np.float32) / 255
    W = Weights(f"{root}/web/public/models/selfie_segmenter")
    ours = selfie(img, W)[..., 0]

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=f"{root}/web/public/models/selfie_segmenter.tflite"); it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], img[None])
    it.invoke()
    ref = it.get_tensor(it.get_output_details()[0]["index"])[0, ..., 0]
    print(f"max |numpy - tflite| = {np.abs(ours - ref).max():.2e}   mask mean {ref.mean():.3f}")

    out = f"{root}/web/public/test_data"; os.makedirs(out, exist_ok=True)
    img.astype(np.float32).tofile(f"{out}/input_256.f32")
    ref.astype(np.float32).tofile(f"{out}/mask_256.f32")
    Image.fromarray((ref * 255).astype(np.uint8)).save(f"{out}/mask_tflite.png")
