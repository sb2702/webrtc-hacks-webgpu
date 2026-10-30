"""Minimal TFLite flatbuffer reader: ops, tensor wiring, options, weights.

Loads TF's generated schema module straight from its file so we don't pay the
full `import tensorflow` cost (only `flatbuffers` is needed).
"""
import importlib.util, sys
import numpy as np

def _load_schema():
    import site, glob
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        hits = glob.glob(f"{sp}/tensorflow/lite/python/schema_py_generated.py")
        if hits:
            spec = importlib.util.spec_from_file_location("tfl_schema", hits[0])
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
    raise ImportError("schema_py_generated.py not found (pip install tensorflow)")

S = _load_schema()
BUILTIN = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
PADDING = {S.Padding.SAME: "SAME", S.Padding.VALID: "VALID"}
ACT = {v: k for k, v in vars(S.ActivationFunctionType).items() if not k.startswith("_")}
DTYPE = {S.TensorType.FLOAT32: np.float32, S.TensorType.FLOAT16: np.float16,
         S.TensorType.INT32: np.int32}

class Graph:
    def __init__(self, path):
        buf = open(path, "rb").read()
        self.model = S.Model.GetRootAsModel(buf, 0)
        self.sg = self.model.Subgraphs(0)
        self.tensors = [self.sg.Tensors(i) for i in range(self.sg.TensorsLength())]
        self.inputs = [self.sg.Inputs(i) for i in range(self.sg.InputsLength())]
        self.outputs = [self.sg.Outputs(i) for i in range(self.sg.OutputsLength())]

    def name(self, t): return self.tensors[t].Name().decode()
    def shape(self, t): return self.tensors[t].ShapeAsNumpy().tolist()

    def raw(self, t):
        """Constant data of tensor t, or None for activations."""
        tensor = self.tensors[t]
        b = self.model.Buffers(tensor.Buffer())
        if b.DataLength() == 0: return None
        return np.frombuffer(b.DataAsNumpy().tobytes(), DTYPE[tensor.Type()]).reshape(self.shape(t))

    def ops(self):
        out = []
        for i in range(self.sg.OperatorsLength()):
            op = self.sg.Operators(i)
            code = self.model.OperatorCodes(op.OpcodeIndex())
            b = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            kind = code.CustomCode().decode() if b == S.BuiltinOperator.CUSTOM else BUILTIN[b]
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            outs = [op.Outputs(j) for j in range(op.OutputsLength())]
            out.append(dict(kind=kind, inputs=ins, outputs=outs, opts=self._opts(op, kind)))
        return out

    def _opts(self, op, kind):
        t = op.BuiltinOptions()
        if t is None: return {}
        def wrap(cls):
            o = cls(); o.Init(t.Bytes, t.Pos); return o
        if kind == "CONV_2D":
            o = wrap(S.Conv2DOptions)
            return dict(pad=PADDING[o.Padding()], stride=(o.StrideH(), o.StrideW()),
                        dilation=(o.DilationHFactor(), o.DilationWFactor()), act=ACT[o.FusedActivationFunction()])
        if kind == "DEPTHWISE_CONV_2D":
            o = wrap(S.DepthwiseConv2DOptions)
            return dict(pad=PADDING[o.Padding()], stride=(o.StrideH(), o.StrideW()),
                        mult=o.DepthMultiplier(), act=ACT[o.FusedActivationFunction()])
        if kind == "AVERAGE_POOL_2D":
            o = wrap(S.Pool2DOptions)
            return dict(pad=PADDING[o.Padding()], stride=(o.StrideH(), o.StrideW()),
                        filter=(o.FilterHeight(), o.FilterWidth()), act=ACT[o.FusedActivationFunction()])
        if kind in ("ADD", "MUL"):
            o = wrap(S.AddOptions if kind == "ADD" else S.MulOptions)
            return dict(act=ACT[o.FusedActivationFunction()])
        if kind == "RESIZE_BILINEAR":
            o = wrap(S.ResizeBilinearOptions)
            return dict(align_corners=bool(o.AlignCorners()), half_pixel=bool(o.HalfPixelCenters()))
        return {}

    def const(self, t):
        """Weights, following a DEQUANTIZE (fp16 → fp32) producer if there is one."""
        d = self.raw(t)
        if d is not None: return d.astype(np.float32)
        for op in self.ops():
            if op["outputs"] == [t] and op["kind"] == "DEQUANTIZE":
                return self.raw(op["inputs"][0]).astype(np.float32)
        return None

if __name__ == "__main__":
    g = Graph(sys.argv[1])
    print("inputs", [(g.name(t), g.shape(t)) for t in g.inputs])
    print("outputs", [(g.name(t), g.shape(t)) for t in g.outputs])
    for i, op in enumerate(g.ops()):
        if op["kind"] == "DEQUANTIZE": continue
        acts = [t for t in op["inputs"] if t >= 0 and g.const(t) is None]
        print(f"{i:3d} {op['kind']:<26} in={acts} out={op['outputs']} "
              f"{g.shape(op['outputs'][0])} {op['opts']}")
