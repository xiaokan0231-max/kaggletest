"""task335 v7: pattern (b) Where(new4, e4_onehot, input) as the graph output.
Eliminates g1u8/gmp1/sentinel. Out-of-grid handled free (input is all-zero there).
g1 kept only to derive the empty-cell mask (original color 0)."""
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
C("ax2s", np.array(2, np.int64))
C("ax3s", np.array(3, np.int64))
C("zerof", np.array(0.0, np.float32))
C("onef", np.array(1.0, np.float32))
# color-4 one-hot constant [1,10,1,1]
e4 = np.zeros((1, 10, 1, 1), np.float32); e4[0, 4, 0, 0] = 1.0
C("e4", e4)
C("ch8_s", np.array([8], np.int64)); C("ch8_e", np.array([9], np.int64))
C("ch2_s", np.array([2], np.int64)); C("ch2_e", np.array([3], np.int64))

n("Conv", ["input", "w_col"], "g1")                  # f32, interior=color+1, ext 0

n("ReduceMax", ["input", "ax2"], "pmax_col")
n("ReduceMax", ["input", "ax3"], "pmax_row")
n("Slice", ["pmax_col", "ch8_s", "ch8_e", "ax1"], "col8")
n("Slice", ["pmax_col", "ch2_s", "ch2_e", "ax1"], "col2")
n("Slice", ["pmax_row", "ch8_s", "ch8_e", "ax1"], "row8")
n("Slice", ["pmax_row", "ch2_s", "ch2_e", "ax1"], "row2")
n("Max", ["row8", "row2"], "rowmark")
n("Max", ["col8", "col2"], "colmark")

n("CumSum", ["rowmark", "ax2s"], "rfwd")
n("CumSum", ["rowmark", "ax2s"], "rrev", reverse=1)
n("Greater", ["rfwd", "zerof"], "rf_b")
n("Greater", ["rrev", "zerof"], "rr_b")
n("And", ["rf_b", "rr_b"], "rowband")                # [1,1,30,1] bool

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

# empty = original color 0  <=> g1 == 1
n("Equal", ["g1", "onef"], "empty_b")
n("And", ["seg_b", "empty_b"], "new4_b")

# pattern (b): keep input one-hot, overwrite path-empty cells with color-4 one-hot
n("Where", ["new4_b", "e4", "input"], "output")      # [1,10,30,30] f32 = output

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
