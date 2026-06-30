"""task335 v12: drop the colsum/rowsum fp16 casts (compare f32 sums directly).
Mark vectors cast bool->fp16 only for CumSum. Path = 3 bool masks (2700B floor)."""
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


Wcol = np.zeros((1, 10, H, 1), np.float32)
for ch in range(10):
    Wcol[0, ch, :, 0] = ch
C("Wcol", Wcol)
Wrow = np.zeros((1, 10, 1, W), np.float32)
for ch in range(10):
    Wrow[0, ch, 0, :] = ch
C("Wrow", Wrow)

C("ax3s", np.array(3, np.int64))
C("ax2s", np.array(2, np.int64))
C("eightf", np.array(8.0, np.float32))
C("twof", np.array(2.0, np.float32))
C("halff", np.array(0.5, np.float32))
C("zero16", np.array(0.0, f16))
e4 = np.zeros((1, 10, 1, 1), np.float32); e4[0, 4, 0, 0] = 1.0
C("e4", e4)

n("Conv", ["input", "Wcol"], "colsum")       # [1,1,1,30] f32
n("Conv", ["input", "Wrow"], "rowsum")       # [1,1,30,1] f32

n("Equal", ["colsum", "eightf"], "col8_b")
n("Equal", ["rowsum", "twof"], "row2_b")
n("Equal", ["rowsum", "eightf"], "row8_b")
n("Equal", ["colsum", "twof"], "col2_b")
n("Greater", ["colsum", "halff"], "colmark_b")
n("Greater", ["rowsum", "halff"], "rowmark_b")

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
