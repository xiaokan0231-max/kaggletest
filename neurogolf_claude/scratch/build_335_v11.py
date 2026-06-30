"""task335 v11: ELIMINATE the 30x30 g1 grid. Each row/column has at most one
nonzero cell (the single 8 or single 2), so a full-height/width Conv that SUMS
color values collapses directly to a 30-elem vector:
  colsum = Conv(input, Wcol[1,10,30,1])  -> [1,1,1,30]: 8 in 8-col, 2 in 2-col, 0 else
  rowsum = Conv(input, Wrow[1,10,1,30])  -> [1,1,30,1]
No interior [1,1,30,30] f32 tensor at all. Output via Where(new4, e4, input).
Only large interior tensors: the 3 bool path masks (900B each)."""
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


# Conv weights: color value at each channel, spanning a full column / row.
Wcol = np.zeros((1, 10, H, 1), np.float32)
for ch in range(10):
    Wcol[0, ch, :, 0] = ch          # sum over rows -> column color-sum
C("Wcol", Wcol)
Wrow = np.zeros((1, 10, 1, W), np.float32)
for ch in range(10):
    Wrow[0, ch, 0, :] = ch          # sum over cols -> row color-sum
C("Wrow", Wrow)

C("ax3s", np.array(3, np.int64))
C("ax2s", np.array(2, np.int64))
C("eight16", np.array(8.0, f16))
C("two16", np.array(2.0, f16))
C("half16", np.array(0.5, f16))
C("zero16", np.array(0.0, f16))
e4 = np.zeros((1, 10, 1, 1), np.float32); e4[0, 4, 0, 0] = 1.0
C("e4", e4)

# collapse to row/col color-sum vectors (no 30x30 grid)
n("Conv", ["input", "Wcol"], "colsum32")     # [1,1,1,30] f32
n("Conv", ["input", "Wrow"], "rowsum32")     # [1,1,30,1] f32
n("Cast", ["colsum32"], "colsum", to=TensorProto.FLOAT16)
n("Cast", ["rowsum32"], "rowsum", to=TensorProto.FLOAT16)

# presence
n("Equal", ["colsum", "eight16"], "col8_b")    # [1,1,1,30]
n("Equal", ["rowsum", "two16"], "row2_b")      # [1,1,30,1]
n("Equal", ["rowsum", "eight16"], "row8_b")
n("Equal", ["colsum", "two16"], "col2_b")
n("Greater", ["colsum", "half16"], "colmark_b")
n("Greater", ["rowsum", "half16"], "rowmark_b")

# bands via CumSum (fp16)
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

# exclude endpoint row/col so whole path is empty cells
n("Not", ["row8_b"], "not_row8")
n("And", ["rband_b", "not_row8"], "rowband")
n("Not", ["col2_b"], "not_col2")
n("And", ["cband_b", "not_col2"], "colband")

# L-path (3 bool 30x30 tensors)
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
