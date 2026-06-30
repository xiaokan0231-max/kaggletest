"""Build compact ONNX for task335.

Rule (verified on all 266 train+test+arc-gen examples):
  Grid has one cell of color 8 at (r8,c8) and one cell of color 2 at (r2,c2)
  on background 0. Draw an L-shaped path of color 4 connecting them:
    - vertical segment: column c8, rows inclusive [min(r8,r2),max(r8,r2)]
    - horizontal segment: row r2, cols inclusive [min(c8,c2),max(c8,c2)]
  Only cells that were 0 become 4; the 8 and 2 cells keep their color.

ONNX algorithm (no labeling, no Loop):
  g = single color grid [1,1,30,30] via Conv weight [[0..9]].
  is8=(g==8), is2=(g==2). Reduce to row/col presence vectors (tiny 30-elem).
  Band-between via triangular MatMul cummax. Compose vertical+horizontal seg.
  gm = where(seg & g==0, 4, g); set out-of-grid cells to sentinel 255.
  Output one-hot bool [1,10,30,30] = Equal(gm, colors[1,10,1,1]) = graph output
  (excluded from memory).

All interior tensors are uint8/bool [1,1,30,30] (<=900B) or 30-elem vectors.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b20_task335.onnx"
H = W = 30

nodes, inits = [], []


def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name))
    return name


def n(op, i, o, **kw):
    if isinstance(o, str):
        o = [o]
    if isinstance(i, str):
        i = [i]
    nodes.append(helper.make_node(op, i, o, **kw))
    return o[0]


# ---------- constants ----------
# color collapse weight [1,10,1,1] = [0..9]
w_col = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
C("w_col", w_col)
C("ax1", np.array([1], np.int64))
C("ax2", np.array([2], np.int64))
C("ax3", np.array([3], np.int64))
# triangular lower (incl diag) [30,30] for cummax-down along rows:
#   tri_d[i,j] = 1 if j<=i  -> max over j<=i. We use MatMul with max via... but MatMul is sum.
# Instead implement cummax with running prefix-OR on boolean using MatMul as
# "any j<=i with mark"  = sign(tri @ mark). We work with 0/1 marks so sum>0 == any.
tri_d = np.tril(np.ones((H, H), np.float32))   # [i,j]=1 if j<=i
tri_u = np.triu(np.ones((H, H), np.float32))   # [i,j]=1 if j>=i
C("tri_d", tri_d)
C("tri_u", tri_u)
tri_dW = np.tril(np.ones((W, W), np.float32))
tri_uW = np.triu(np.ones((W, W), np.float32))
C("tri_dW", tri_dW)
C("tri_uW", tri_uW)
C("c8f", np.array(8.0, np.float32))
C("c2f", np.array(2.0, np.float32))
C("c0f", np.array(0.0, np.float32))
C("c4u", np.array(4, np.uint8))
C("c255u", np.array(255, np.uint8))
C("zerof", np.array(0.0, np.float32))
# colors 0..9 as uint8 [1,10,1,1] for final one-hot Equal
colors_u = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
C("colors_u", colors_u)

# ---------- color grid + active ----------
n("Conv", ["input", "w_col"], "g")                          # [1,1,30,30] f32 color
n("ReduceSum", ["input", "ax1"], "active", keepdims=1)      # 1 inside real grid

# ---------- masks ----------
n("Equal", ["g", "c8f"], "is8")                             # bool [1,1,30,30]
n("Equal", ["g", "c2f"], "is2")
n("Cast", ["is8"], "is8f", to=TensorProto.FLOAT)
n("Cast", ["is2"], "is2f", to=TensorProto.FLOAT)

# row/col presence vectors
# row_has8 [1,1,30,1] = max over cols(axis3); col_has8 [1,1,1,30]=max over rows(axis2)
n("ReduceMax", ["is8f", "ax3"], "row8", keepdims=1)         # [1,1,30,1]
n("ReduceMax", ["is8f", "ax2"], "col8", keepdims=1)         # [1,1,1,30]
n("ReduceMax", ["is2f", "ax3"], "row2", keepdims=1)
n("ReduceMax", ["is2f", "ax2"], "col2", keepdims=1)

# row band between the two marked rows: rowmark=max(row8,row2) [1,1,30,1]
n("Max", ["row8", "row2"], "rowmark")
n("Max", ["col8", "col2"], "colmark")

# cummax-down rows: for each i, any mark at j<=i. rowmark is [1,1,30,1].
# squeeze to [30] not needed; use MatMul on last two dims won't align. Reshape.
# rowmark -> [30,1]; tri_d[30,30] @ [30,1] -> [30,1] sum; >0 == any.
C("rs_r", np.array([H, 1], np.int64))
n("Reshape", ["rowmark", "rs_r"], "rowmark2")               # [30,1]
n("MatMul", ["tri_d", "rowmark2"], "rd_sum")                # [30,1]
n("MatMul", ["tri_u", "rowmark2"], "ru_sum")                # [30,1]
n("Greater", ["rd_sum", "zerof"], "rd_b")
n("Greater", ["ru_sum", "zerof"], "ru_b")
n("And", ["rd_b", "ru_b"], "rowband_b")                     # [30,1] bool
n("Cast", ["rowband_b"], "rowband", to=TensorProto.FLOAT)
# reshape rowband back to [1,1,30,1]
C("rs_r4", np.array([1, 1, H, 1], np.int64))
n("Reshape", ["rowband", "rs_r4"], "rowband4")

# col band between marked cols: colmark [1,1,1,30] -> [30,1] via reshape, tri on W
C("rs_c", np.array([W, 1], np.int64))
n("Reshape", ["colmark", "rs_c"], "colmark2")               # [30,1]
n("MatMul", ["tri_dW", "colmark2"], "cd_sum")
n("MatMul", ["tri_uW", "colmark2"], "cu_sum")
n("Greater", ["cd_sum", "zerof"], "cd_b")
n("Greater", ["cu_sum", "zerof"], "cu_b")
n("And", ["cd_b", "cu_b"], "colband_b")
n("Cast", ["colband_b"], "colband", to=TensorProto.FLOAT)
C("rs_c4", np.array([1, 1, 1, W], np.int64))
n("Reshape", ["colband", "rs_c4"], "colband4")              # [1,1,1,30]

# vertical seg = col8 (col==c8, [1,1,1,30]) * rowband4 ([1,1,30,1]) -> [1,1,30,30]
n("Mul", ["col8", "rowband4"], "vert")
# horizontal seg = row2 ([1,1,30,1]) * colband4 ([1,1,1,30]) -> [1,1,30,30]
n("Mul", ["row2", "colband4"], "horiz")
n("Max", ["vert", "horiz"], "seg")                          # f32 0/1 [1,1,30,30]

# new4 = seg>0 AND g==0
n("Greater", ["seg", "zerof"], "seg_b")
n("Equal", ["g", "c0f"], "g0_b")
n("And", ["seg_b", "g0_b"], "new4_b")                       # bool

# build output color grid as uint8: gm = where(new4, 4, g_uint8)
n("Cast", ["g"], "g_u8", to=TensorProto.UINT8)
n("Where", ["new4_b", "c4u", "g_u8"], "gm0")               # [1,1,30,30] u8
# mask out-of-grid: where active==0 -> 255 sentinel
n("Greater", ["active", "zerof"], "active_b")
n("Where", ["active_b", "gm0", "c255u"], "gm")             # u8

# final one-hot via Equal against colors 0..9 -> graph output (excluded)
n("Equal", ["gm", "colors_u"], "output")                   # bool [1,10,30,30]

# ---------- graph ----------
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
