"""task335 v6: replace triangular-MatMul cummax with CumSum (drops 1800 params
and all the reshape tensors). Bands computed in-place on [1,1,30,1]/[1,1,1,30]."""
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
C("ax1", np.array([1], np.int64))
C("ax2", np.array([2], np.int64))
C("ax3", np.array([3], np.int64))
C("ax2s", np.array(2, np.int64))   # scalar axis for CumSum (rows)
C("ax3s", np.array(3, np.int64))   # scalar axis for CumSum (cols)
C("zerof", np.array(0.0, np.float32))
C("one_u8", np.array(1, np.uint8))
C("five_u8", np.array(5, np.uint8))
C("colors_p1", np.arange(1, 11, dtype=np.uint8).reshape(1, 10, 1, 1))
C("ch8_s", np.array([8], np.int64)); C("ch8_e", np.array([9], np.int64))
C("ch2_s", np.array([2], np.int64)); C("ch2_e", np.array([3], np.int64))

n("Conv", ["input", "w_col"], "g1")

n("ReduceMax", ["input", "ax2"], "pmax_col")         # [1,10,1,30]
n("ReduceMax", ["input", "ax3"], "pmax_row")         # [1,10,30,1]
n("Slice", ["pmax_col", "ch8_s", "ch8_e", "ax1"], "col8")   # [1,1,1,30]
n("Slice", ["pmax_col", "ch2_s", "ch2_e", "ax1"], "col2")
n("Slice", ["pmax_row", "ch8_s", "ch8_e", "ax1"], "row8")   # [1,1,30,1]
n("Slice", ["pmax_row", "ch2_s", "ch2_e", "ax1"], "row2")
n("Max", ["row8", "row2"], "rowmark")                # [1,1,30,1]
n("Max", ["col8", "col2"], "colmark")                # [1,1,1,30]

# row band via CumSum along rows (axis2)
n("CumSum", ["rowmark", "ax2s"], "rfwd")
n("CumSum", ["rowmark", "ax2s"], "rrev", reverse=1)
n("Greater", ["rfwd", "zerof"], "rf_b")
n("Greater", ["rrev", "zerof"], "rr_b")
n("And", ["rf_b", "rr_b"], "rowband")                # [1,1,30,1] bool

# col band via CumSum along cols (axis3)
n("CumSum", ["colmark", "ax3s"], "cfwd")
n("CumSum", ["colmark", "ax3s"], "crev", reverse=1)
n("Greater", ["cfwd", "zerof"], "cf_b")
n("Greater", ["crev", "zerof"], "cr_b")
n("And", ["cf_b", "cr_b"], "colband")                # [1,1,1,30] bool

n("Greater", ["col8", "zerof"], "col8_b")
n("Greater", ["row2", "zerof"], "row2_b")
n("And", ["col8_b", "rowband"], "vert_b")            # [1,1,30,30]
n("And", ["row2_b", "colband"], "horiz_b")
n("Or", ["vert_b", "horiz_b"], "seg_b")

n("Cast", ["g1"], "g1u8", to=TensorProto.UINT8)
n("Equal", ["g1u8", "one_u8"], "empty_b")
n("And", ["seg_b", "empty_b"], "new4_b")
n("Where", ["new4_b", "five_u8", "g1u8"], "gmp1")
n("Equal", ["gmp1", "colors_p1"], "output")

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
