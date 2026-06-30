"""task335 v10: all 30-elem band vectors in float16 (60B vs 120B).
g1 (f32, required) -> colmax/rowmax (f32 ReduceMax) -> cast fp16 -> vector logic.
3 path tensors (bool) + g1 dominate; vectors squeezed."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b20_task335.onnx"
H = W = 30
nodes, inits = [], []
f16 = np.float16


def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name


def n(op, i, o, **kw):
    if isinstance(o, str): o = [o]
    if isinstance(i, str): i = [i]
    nodes.append(helper.make_node(op, i, o, **kw)); return o[0]


C("w_col", np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1))
C("ax2", np.array([2], np.int64))
C("ax3", np.array([3], np.int64))
C("ax2s", np.array(2, np.int64))
C("ax3s", np.array(3, np.int64))
C("three16", np.array(3.0, f16))
C("nine16", np.array(9.0, f16))
C("two5_16", np.array(2.5, f16))
C("zero16", np.array(0.0, f16))
e4 = np.zeros((1, 10, 1, 1), np.float32); e4[0, 4, 0, 0] = 1.0
C("e4", e4)

n("Conv", ["input", "w_col"], "g1")          # f32

n("ReduceMax", ["g1", "ax2"], "colmax32", keepdims=1)
n("ReduceMax", ["g1", "ax3"], "rowmax32", keepdims=1)
n("Cast", ["colmax32"], "colmax", to=TensorProto.FLOAT16)   # [1,1,1,30] f16
n("Cast", ["rowmax32"], "rowmax", to=TensorProto.FLOAT16)   # [1,1,30,1] f16

n("Equal", ["colmax", "nine16"], "col8_b")
n("Equal", ["rowmax", "three16"], "row2_b")
n("Equal", ["rowmax", "nine16"], "row8_b")
n("Equal", ["colmax", "three16"], "col2_b")
n("Greater", ["colmax", "two5_16"], "colmark_b")
n("Greater", ["rowmax", "two5_16"], "rowmark_b")

n("Cast", ["rowmark_b"], "rowmark", to=TensorProto.FLOAT16)
n("Cast", ["colmark_b"], "colmark", to=TensorProto.FLOAT16)
n("CumSum", ["rowmark", "ax2s"], "rfwd")
n("CumSum", ["rowmark", "ax2s"], "rrev", reverse=1)
n("Mul", ["rfwd", "rrev"], "rband_f")
n("Greater", ["rband_f", "zero16"], "rband_b")
n("CumSum", ["colmark", "ax3s"], "cfwd")
n("CumSum", ["colmark", "ax3s"], "crev", reverse=1)
n("Mul", ["cfwd", "crev"], "cband_f")
n("Greater", ["cband_f", "zero16"], "cband_b")

n("Not", ["row8_b"], "not_row8")
n("And", ["rband_b", "not_row8"], "rowband")
n("Not", ["col2_b"], "not_col2")
n("And", ["cband_b", "not_col2"], "colband")

n("And", ["col8_b", "rowband"], "vert_b")
n("And", ["row2_b", "colband"], "horiz_b")
n("Or", ["vert_b", "horiz_b"], "new4_b")

n("Where", ["new4_b", "e4", "input"], "output")

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
