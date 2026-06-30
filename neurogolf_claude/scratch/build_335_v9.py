"""task335 v9: shrink bands to exclude the 8-row and 2-col endpoints so the
whole L-path is guaranteed empty -> drop the empty mask AND new4-And (both 900B).
new4 = vert OR horiz directly. Endpoint exclusion done on 30-elem vectors (tiny).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b20_task335.onnx"
H = W = 30
nodes, inits = [], []


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
C("threef", np.array(3.0, np.float32))
C("ninef", np.array(9.0, np.float32))
C("two5f", np.array(2.5, np.float32))
C("zerof", np.array(0.0, np.float32))
e4 = np.zeros((1, 10, 1, 1), np.float32); e4[0, 4, 0, 0] = 1.0
C("e4", e4)

n("Conv", ["input", "w_col"], "g1")          # 8->9,2->3,0->1,outside->0

n("ReduceMax", ["g1", "ax2"], "colmax", keepdims=1)   # [1,1,1,30]
n("ReduceMax", ["g1", "ax3"], "rowmax", keepdims=1)   # [1,1,30,1]

# presence
n("Equal", ["colmax", "ninef"], "col8_b")    # 8's column
n("Equal", ["rowmax", "threef"], "row2_b")   # 2's row
n("Equal", ["rowmax", "ninef"], "row8_b")    # 8's row (to exclude)
n("Equal", ["colmax", "threef"], "col2_b")   # 2's col (to exclude)
n("Greater", ["colmax", "two5f"], "colmark_b")
n("Greater", ["rowmax", "two5f"], "rowmark_b")

# bands
n("Cast", ["rowmark_b"], "rowmark", to=TensorProto.FLOAT)
n("Cast", ["colmark_b"], "colmark", to=TensorProto.FLOAT)
n("CumSum", ["rowmark", "ax2s"], "rfwd")
n("CumSum", ["rowmark", "ax2s"], "rrev", reverse=1)
n("Mul", ["rfwd", "rrev"], "rband_f")
n("Greater", ["rband_f", "zerof"], "rband_b")
n("CumSum", ["colmark", "ax3s"], "cfwd")
n("CumSum", ["colmark", "ax3s"], "crev", reverse=1)
n("Mul", ["cfwd", "crev"], "cband_f")
n("Greater", ["cband_f", "zerof"], "cband_b")

# shrink: vertical band excludes the 8-row; horizontal band excludes the 2-col
n("Not", ["row8_b"], "not_row8")
n("And", ["rband_b", "not_row8"], "rowband")   # [1,1,30,1]
n("Not", ["col2_b"], "not_col2")
n("And", ["cband_b", "not_col2"], "colband")   # [1,1,1,30]

# segments -> path (all empty by construction)
n("And", ["col8_b", "rowband"], "vert_b")    # [1,1,30,30]
n("And", ["row2_b", "colband"], "horiz_b")
n("Or", ["vert_b", "horiz_b"], "new4_b")     # path-empty cells

n("Where", ["new4_b", "e4", "input"], "output")

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
