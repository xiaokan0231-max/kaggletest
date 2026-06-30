"""Compact D4-symmetrization ONNX for task020 (STATIC SHAPES ONLY).

Rule (verified 0/262): single 5x5 object; output = D4 symmetrization of it about its own
center (no orbit conflicts -> max over the 8 transforms).  All content lives in rows/cols
1..8 -> an 8x8 frame.

To keep the interior tensors tiny AND fully static:
  1. Conv collapse one-hot -> color grid g [1,1,30,30] f32 (3600B floor).
  2. Slice the 8x8 frame (rows/cols 1..8); occupancy -> r0',c0' (top/left in the frame).
  3. Align the object to the top-left corner with shift permutation matmuls
     (ga = Su @ g @ Sl), then take a STATIC [0:5,0:5] slice -> 5x5 object.
  4. D4-symmetrize the fixed 5x5 with uint8 Gather flips + Transpose (no MatMul, 25B each):
     Shv = max(sub, UD, LR, UDLR); sym = max(Shv, Shv^T).
  5. Pad 5x5 -> 8x8, shift back to original position (SuT @ pad @ SlT), pad 8x8 -> 10x10
     (background color 0) then 10x10 -> 30x30 (sentinel 255); Equal(gm, colors) -> output.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as h
from onnx import numpy_helper as nh

N = 8
CROP = 1

nodes = []
inits = []


def C(name, arr):
    inits.append(nh.from_array(np.asarray(arr), name))
    return name


# ---- constants ----
C("w_color", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
C("c_colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
C("s_crop", np.array([CROP, CROP], dtype=np.int64))
C("e_crop", np.array([CROP + N, CROP + N], dtype=np.int64))
C("ax23", np.array([2, 3], dtype=np.int64))
C("sq01", np.array([0, 1], dtype=np.int64))
C("idx", np.arange(N, dtype=np.float16))
C("zero_h", np.array(0.0, dtype=np.float16))
C("ax0_i", np.array(0, dtype=np.int64))
C("ax0_red", np.array([0], dtype=np.int64))
C("ax1_red", np.array([1], dtype=np.int64))
C("u1", np.array([1], dtype=np.int64))
C("u0", np.array([0], dtype=np.int64))
C("s_55", np.array([0, 0], dtype=np.int64))     # static slice start [0:5,0:5]
C("e_55", np.array([5, 5], dtype=np.int64))
C("sl_ax01", np.array([0, 1], dtype=np.int64))
C("rev5", np.array([4, 3, 2, 1, 0], dtype=np.int64))   # reverse a length-5 axis
C("pads_5to8", np.array([0, 0, 3, 3], dtype=np.int64))  # pad 5x5 -> 8x8 (ends 3,3)
C("pad0u8", np.array(0, dtype=np.uint8))
_gback = 10 - CROP - N
C("pads_grid", np.array([0, 0, CROP, CROP, 0, 0, _gback, _gback], dtype=np.int64))
C("pads_out", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64))
C("sent255", np.array(255, dtype=np.uint8))

# ---- crop the 8x8 frame from the input FIRST (cheaper than a full 30x30 Conv) ----
# Slice input [1,10,30,30] -> [1,10,8,8] (640 elems = 2560B) then Conv-collapse -> [1,1,8,8].
# This beats the 3600B full-grid Conv: 2560B(slice)+256B(conv) < 3600B.
nodes.append(h.make_node("Slice", ["input", "s_crop", "e_crop", "ax23"], ["in8"]))  # [1,10,8,8] f32
nodes.append(h.make_node("Conv", ["in8", "w_color"], ["g8"]))                    # [1,1,8,8] f32
nodes.append(h.make_node("Cast", ["g8"], ["g8h"], to=TP.FLOAT16))
nodes.append(h.make_node("Squeeze", ["g8h", "sq01"], ["g2d"]))                    # [8,8] f16

# ---- occupancy -> r0', c0' ----
nodes.append(h.make_node("ReduceMax", ["g2d", "ax1_red"], ["rmax"], keepdims=0))
nodes.append(h.make_node("ReduceMax", ["g2d", "ax0_red"], ["cmax"], keepdims=0))
nodes.append(h.make_node("CumSum", ["rmax", "ax0_i"], ["cr"]))
nodes.append(h.make_node("Equal", ["cr", "zero_h"], ["cr0_b"]))
nodes.append(h.make_node("Cast", ["cr0_b"], ["cr0_f"], to=TP.FLOAT16))
nodes.append(h.make_node("ReduceSum", ["cr0_f", "ax0_red"], ["r0"], keepdims=0))
nodes.append(h.make_node("CumSum", ["cmax", "ax0_i"], ["cc"]))
nodes.append(h.make_node("Equal", ["cc", "zero_h"], ["cc0_b"]))
nodes.append(h.make_node("Cast", ["cc0_b"], ["cc0_f"], to=TP.FLOAT16))
nodes.append(h.make_node("ReduceSum", ["cc0_f", "ax0_red"], ["c0"], keepdims=0))

# ---- alignment shift perms ----
# idx grids
nodes.append(h.make_node("Unsqueeze", ["idx", "u1"], ["idx_col"]))   # [N,1] = i
nodes.append(h.make_node("Unsqueeze", ["idx", "u0"], ["idx_row"]))   # [1,N] = j
# Su[k,m] = 1 iff m == k + r0   ->  (idx_row == idx_col + r0)
nodes.append(h.make_node("Add", ["idx_col", "r0"], ["icpr0"]))       # k + r0
nodes.append(h.make_node("Equal", ["idx_row", "icpr0"], ["Su_b"]))
nodes.append(h.make_node("Cast", ["Su_b"], ["Su"], to=TP.FLOAT16))
# Sl[m,k] = 1 iff m == k + c0   ->  (idx_col == idx_row + c0)
nodes.append(h.make_node("Add", ["idx_row", "c0"], ["irpc0"]))       # k + c0  (broadcast row)
nodes.append(h.make_node("Equal", ["idx_col", "irpc0"], ["Sl_b"]))
nodes.append(h.make_node("Cast", ["Sl_b"], ["Sl"], to=TP.FLOAT16))
# inverse (shift back) perms = transposes of Su, Sl, built directly from formulas.
# SuT[m,k] = 1 iff m == k + r0  (m=row, k=col)  -> (idx_col == idx_row + r0)
nodes.append(h.make_node("Add", ["idx_row", "r0"], ["irpr0"]))
nodes.append(h.make_node("Equal", ["idx_col", "irpr0"], ["SuT_b"]))
nodes.append(h.make_node("Cast", ["SuT_b"], ["SuT"], to=TP.FLOAT16))
# SlT = transpose of Sl[m,k]=(m==k+c0) -> SlT[a,b]=1 iff b==a+c0 -> (idx_row == idx_col + c0)
nodes.append(h.make_node("Add", ["idx_col", "c0"], ["icpc0"]))   # a + c0  ([N,1])
nodes.append(h.make_node("Equal", ["idx_row", "icpc0"], ["SlT_b"]))  # [1,N]==[N,1] -> (b==a+c0)
nodes.append(h.make_node("Cast", ["SlT_b"], ["SlT"], to=TP.FLOAT16))

# ---- align object to top-left corner ----
nodes.append(h.make_node("MatMul", ["Su", "g2d"], ["gu"]))     # rows shifted up
nodes.append(h.make_node("MatMul", ["gu", "Sl"], ["ga"]))      # cols shifted left  [8,8] f16
nodes.append(h.make_node("Slice", ["ga", "s_55", "e_55", "sl_ax01"], ["sub_h"]))  # [5,5] f16
nodes.append(h.make_node("Cast", ["sub_h"], ["sub"], to=TP.UINT8))  # [5,5] u8

# ---- D4 symmetrize fixed 5x5 (uint8, Gather flips) ----
nodes.append(h.make_node("Gather", ["sub", "rev5"], ["ud"], axis=0))   # UD flip
nodes.append(h.make_node("Gather", ["sub", "rev5"], ["lr"], axis=1))   # LR flip
nodes.append(h.make_node("Gather", ["ud", "rev5"], ["udlr"], axis=1))  # both
nodes.append(h.make_node("Max", ["sub", "ud", "lr", "udlr"], ["Shv5"]))
nodes.append(h.make_node("Transpose", ["Shv5"], ["Shv5T"], perm=[1, 0]))
nodes.append(h.make_node("Max", ["Shv5", "Shv5T"], ["sym5"]))          # [5,5] u8

# ---- place back ----
nodes.append(h.make_node("Pad", ["sym5", "pads_5to8", "pad0u8"], ["sym8"]))  # [8,8] u8
nodes.append(h.make_node("Cast", ["sym8"], ["sym8h"], to=TP.FLOAT16))        # need float for matmul
nodes.append(h.make_node("MatMul", ["SuT", "sym8h"], ["bu"]))   # shift rows back down
nodes.append(h.make_node("MatMul", ["bu", "SlT"], ["back"]))    # shift cols back right [8,8] f16
nodes.append(h.make_node("Cast", ["back"], ["back_u8"], to=TP.UINT8))
nodes.append(h.make_node("Unsqueeze", ["back_u8", "sq01"], ["back4"]))       # [1,1,8,8] u8
nodes.append(h.make_node("Pad", ["back4", "pads_grid", "pad0u8"], ["gm10"]))  # [1,1,10,10]
nodes.append(h.make_node("Pad", ["gm10", "pads_out", "sent255"], ["gm30"]))   # [1,1,30,30]
nodes.append(h.make_node("Equal", ["gm30", "c_colors"], ["output"]))

# drop the dead None entries
nodes = [n for n in nodes if n is not None]

graph = h.make_graph(
    nodes, "task020_align5",
    [h.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
    [h.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])],
    inits,
)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/p1a_task020.onnx")
print("saved, nodes:", len(model.graph.node), "inits:", len(inits))
