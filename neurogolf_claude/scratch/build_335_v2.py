"""Build compact ONNX for task335 - v2 (presence-vector + sentinel collapse).

Rule: one 8 at (r8,c8), one 2 at (r2,c2). Draw L of 4s: vertical column c8 rows
[min..max] and horizontal row r2 cols [min..max], only on background-0 cells.

Memory tricks vs v1:
 * Conv weight [1..10] -> g1: interior=color+1, OUT-OF-GRID=0 (free 'active').
   colbase = Cast(g1,u8)-1 : exterior underflows to 255 = sentinel, free.
 * Presence vectors via ReduceMax(input) over a spatial axis -> [1,10,1,30]/[1,10,30,1]
   then Slice channels 2/8 -> 30-elem vectors. No 30x30 is8/is2 tensors.
 * Band via triangular MatMul cummax on 30-elem vectors.
 * Only large interior 30x30 tensors: g1(f32, required), and a few u8/bool masks.
 * Output one-hot bool = Equal(gm_u8, colors) = graph output (excluded).
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


# constants
C("w_col", np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1))  # color+1
C("ax2", np.array([2], np.int64))
C("ax3", np.array([3], np.int64))
C("tri_d", np.tril(np.ones((H, H), np.float32)))
C("tri_u", np.triu(np.ones((H, H), np.float32)))
C("zerof", np.array(0.0, np.float32))
C("one_u8", np.array(1, np.uint8))
C("four_u8", np.array(4, np.uint8))
C("g1_is1", np.array(1.0, np.float32))   # g1==1 means original color 0
C("colors_u", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
# slice helpers for channel selection
C("ch8_s", np.array([8], np.int64))
C("ch8_e", np.array([9], np.int64))
C("ch2_s", np.array([2], np.int64))
C("ch2_e", np.array([3], np.int64))
C("ax1", np.array([1], np.int64))

# ---- collapse ----
n("Conv", ["input", "w_col"], "g1")                  # f32 [1,1,30,30], ext=0

# ---- presence vectors ----
n("ReduceMax", ["input", "ax2"], "pmax_col")         # [1,10,1,30] color x col present
n("ReduceMax", ["input", "ax3"], "pmax_row")         # [1,10,30,1] color x row present
# channel slices -> tiny vectors
n("Slice", ["pmax_col", "ch8_s", "ch8_e", "ax1"], "col8")   # [1,1,1,30]
n("Slice", ["pmax_col", "ch2_s", "ch2_e", "ax1"], "col2")
n("Slice", ["pmax_row", "ch8_s", "ch8_e", "ax1"], "row8")   # [1,1,30,1]
n("Slice", ["pmax_row", "ch2_s", "ch2_e", "ax1"], "row2")
n("Max", ["row8", "row2"], "rowmark")                # [1,1,30,1]
n("Max", ["col8", "col2"], "colmark")                # [1,1,1,30]

# ---- bands via triangular cummax ----
C("rs30_1", np.array([H, 1], np.int64))
n("Reshape", ["rowmark", "rs30_1"], "rowmark2")      # [30,1]
n("MatMul", ["tri_d", "rowmark2"], "rd")
n("MatMul", ["tri_u", "rowmark2"], "ru")
n("Greater", ["rd", "zerof"], "rd_b")
n("Greater", ["ru", "zerof"], "ru_b")
n("And", ["rd_b", "ru_b"], "rband_b")                # [30,1] bool
n("Cast", ["rband_b"], "rband", to=TensorProto.FLOAT)
C("rs_r4", np.array([1, 1, H, 1], np.int64))
n("Reshape", ["rband", "rs_r4"], "rowband")          # [1,1,30,1]

n("Reshape", ["colmark", "rs30_1"], "colmark2")
n("MatMul", ["tri_d", "colmark2"], "cd")
n("MatMul", ["tri_u", "colmark2"], "cu")
n("Greater", ["cd", "zerof"], "cd_b")
n("Greater", ["cu", "zerof"], "cu_b")
n("And", ["cd_b", "cu_b"], "cband_b")
n("Cast", ["cband_b"], "cband", to=TensorProto.FLOAT)
C("rs_c4", np.array([1, 1, 1, W], np.int64))
n("Reshape", ["cband", "rs_c4"], "colband")          # [1,1,1,30]

# ---- segment mask (f32 broadcast products, combined) ----
n("Mul", ["col8", "rowband"], "vert")                # [1,1,30,30]
n("Mul", ["row2", "colband"], "horiz")               # [1,1,30,30]
n("Add", ["vert", "horiz"], "seg")                   # >0 where path

# ---- new4 = seg>0 AND original color 0 (g1==1) ----
n("Greater", ["seg", "zerof"], "seg_b")
n("Equal", ["g1", "g1_is1"], "g0_b")
n("And", ["seg_b", "g0_b"], "new4_b")                # bool [1,1,30,30]

# ---- gm uint8: colbase = Cast(g1)-1 (ext->255), then where new4 ->4 ----
n("Cast", ["g1"], "g1u8", to=TensorProto.UINT8)
n("Sub", ["g1u8", "one_u8"], "colbase")              # u8 ; ext 0-1=255
n("Where", ["new4_b", "four_u8", "colbase"], "gm")   # u8

# ---- output one-hot = graph output ----
n("Equal", ["gm", "colors_u"], "output")             # bool [1,10,30,30]

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
