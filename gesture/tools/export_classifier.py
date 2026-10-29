"""Exports the canned gesture classifier's weights to JSON so the browser can run it in plain JS.

The classifier is tiny (embedding -> scale/shift -> ReLU -> dense -> softmax) and
declares a dynamic batch size ([-1, 128]) that LiteRT.js won't accept, so we
evaluate it ourselves (src/gestures.js). Reproduce public/models/gesture_classifier.json with:

    pip install tensorflow
    python3 tools/export_classifier.py public/models/gesture_recognizer.task public/models/gesture_classifier.json
"""
import io, json, sys, zipfile
import numpy as np
import tensorflow as tf

task, out = sys.argv[1], sys.argv[2]
inner = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(task).read('hand_gesture_recognizer.task')))
model = inner.read('canned_gesture_classifier.tflite')
labels = zipfile.ZipFile(io.BytesIO(model)).read('labels.txt').decode().split()  # metadata zip appended to the .tflite

interpreter = tf.lite.Interpreter(model_content=model)
interpreter.resize_tensor_input(0, [1, 128])
interpreter.allocate_tensors()
consts = {t['name']: interpreter.get_tensor(t['index']) for t in interpreter.get_tensor_details()
          if 'batchnorm/mul' in t['name'] and t['shape'].tolist() == [128] or 'batchnorm/sub' in t['name']
          or t['name'].endswith('MatMul') or 'BiasAdd/ReadVariableOp' in t['name']}
pick = lambda key: next(v for k, v in consts.items() if key(k))
scale = pick(lambda k: k.endswith('batchnorm/mul'))
shift = pick(lambda k: k.endswith('batchnorm/sub'))
weights = pick(lambda k: k.endswith('MatMul'))
bias = pick(lambda k: 'BiasAdd' in k)
assert scale.shape == (128,) and shift.shape == (128,) and weights.shape == (8, 128) and bias.shape == (8,)

# Check our formula against the TFLite interpreter on random embeddings.
for _ in range(5):
    x = np.random.randn(1, 128).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x); interpreter.invoke()
    logits = weights @ np.maximum(x[0] * scale + shift, 0) + bias
    ours = np.exp(logits - logits.max()); ours /= ours.sum()
    assert np.abs(ours - interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]).max() < 1e-5

r = lambda a: [round(float(v), 7) for v in a.ravel()]
json.dump({'labels': labels, 'scale': r(scale), 'shift': r(shift), 'weights': r(weights), 'bias': r(bias)}, open(out, 'w'))
print(f'{out}: {labels}')
