"""task335 v8: derive ALL presence vectors from g1 via ReduceMax (axis2/axis3)
-> [1,1,1,30]/[1,1,30,1] (30 elems, 120B) instead of [1,10,..] (1200B).
g1 in color+1 domain: 8-col max=9, 2-col max=3, empty=1, outside=0 (distinct,
since 8 and 2 never share a row or column). Output via Where(new4, e4, input)."""
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
C("onef", np.array(1.0, np.float32))
C("threef", np.array(3.0, np.float32))      # 2+1
C("ninef", np.array(9.0, np.float32))       # 8+1
C("two5f", np.array(2.5, np.float32))       # threshold for "marked" (>=3)
e4 = np.zeros((1, 10, 1, 1), np.float32); e4[0, 4, 0, 0] = 1.0
C("e4", e4)

n("Conv", ["input", "w_col"], "g1")          # f32; 8->9, 2->3, 0->1, outside->0

# column / row max color+1
n("ReduceMax", ["g1", "ax2"], "colmax", keepdims=1)   # [1,1,1,30]
n("ReduceMax", ["g1", "ax3"], "rowmax", keepdims=1)   # [1,1,30,1]

# presence: col8 = colmax==9 ; row2 = rowmax==3
n("Equal", ["colmax", "ninef"], "col8_b")    # [1,1,1,30] bool
n("Equal", ["rowmax", "threef"], "row2_b")   # [1,1,30,1] bool
# marks (8 or 2): max+1 >= 3
n("Greater", ["colmax", "two5f"], "colmark_b")   # [1,1,1,30] bool
n("Greater", ["rowmax", "two5f"], "rowmark_b")   # [1,1,30,1] bool

# bands via CumSum on float marks
n("Cast", ["rowmark_b"], "rowmark", to=TensorProto.FLOAT)
n("Cast", ["colmark_b"], "colmark", to=TensorProto.FLOAT)
n("CumSum", ["rowmark", "ax2s"], "rfwd")
n("CumSum", ["rowmark", "ax2s"], "rrev", reverse=1)
n("Mul", ["rfwd", "rrev"], "rband_f")        # >0 inside band
C("zerof", np.array(0.0, np.float32))
n("Greater", ["rband_f", "zerof"], "rowband")   # [1,1,30,1] bool
n("CumSum", ["colmark", "ax3s"], "cfwd")
n("CumSum", ["colmark", "ax3s"], "crev", reverse=1)
n("Mul", ["cfwd", "crev"], "cband_f")
n("Greater", ["cband_f", "zerof"], "colband")   # [1,1,1,30] bool

# segments
n("And", ["col8_b", "rowband"], "vert_b")    # [1,1,30,30]
n("And", ["row2_b", "colband"], "horiz_b")
n("Or", ["vert_b", "horiz_b"], "seg_b")

# empty = g1==1
n("Equal", ["g1", "onef"], "empty_b")
n("And", ["seg_b", "empty_b"], "new4_b")

n("Where", ["new4_b", "e4", "input"], "output")

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
