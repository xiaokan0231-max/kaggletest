"""task020 compact D4-symmetrization, DIRECT reflection-matmul variant (static shapes).

Same rule as build_020.py.  Pipeline:
  * Slice input [1,10,30,30] -> [1,10,8,8] (frame rows/cols 1..8), Conv-collapse -> [1,1,8,8] f32.
    (slicing the input FIRST then collapsing is cheaper than a full 30x30 Conv: 2560+256 < 3600.)
  * occupancy -> r0',c0'.
  * reflection perms Pr=(i+j==2r0'+4), Pc=(i+j==2c0'+4); transpose-shift Srow=(i-j==r0'-c0').
  * Shv = max(g, Pr@g, g@Pc, Pr@g@Pc); result = max(Shv, Srow@Shv^T@Srow).
  * pad 8x8 -> 10x10 (bg 0) -> 30x30 (sentinel 255); Equal(gm, colors) -> output.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as h
from onnx import numpy_helper as nh

N = 8
CROP = 1
nodes, inits = [], []


def C(name, arr):
    inits.append(nh.from_array(np.asarray(arr), name)); return name


C("w_color", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
C("c_colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
C("s_crop", np.array([CROP, CROP], dtype=np.int64))
C("e_crop", np.array([CROP + N, CROP + N], dtype=np.int64))
C("ax23", np.array([2, 3], dtype=np.int64))
C("sq01", np.array([0, 1], dtype=np.int64))
C("idx", np.arange(N, dtype=np.float16))
C("four", np.array(4.0, dtype=np.float16))
C("two", np.array(2.0, dtype=np.float16))
C("zero_h", np.array(0.0, dtype=np.float16))
C("ax0_i", np.array(0, dtype=np.int64))
C("ax0_red", np.array([0], dtype=np.int64))
C("ax1_red", np.array([1], dtype=np.int64))
C("u1", np.array([1], dtype=np.int64))
C("u0", np.array([0], dtype=np.int64))
_gback = 10 - CROP - N
C("pads_grid", np.array([0, 0, CROP, CROP, 0, 0, _gback, _gback], dtype=np.int64))
C("pad0u8", np.array(0, dtype=np.uint8))
C("pads_out", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64))
C("sent255", np.array(255, dtype=np.uint8))

nodes.append(h.make_node("Slice", ["input", "s_crop", "e_crop", "ax23"], ["in8"]))
nodes.append(h.make_node("Conv", ["in8", "w_color"], ["g8"]))
nodes.append(h.make_node("Cast", ["g8"], ["g8h"], to=TP.FLOAT16))
nodes.append(h.make_node("Squeeze", ["g8h", "sq01"], ["g2d"]))

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

nodes.append(h.make_node("Mul", ["r0", "two"], ["r0x2"]))
nodes.append(h.make_node("Add", ["r0x2", "four"], ["Sr"]))
nodes.append(h.make_node("Mul", ["c0", "two"], ["c0x2"]))
nodes.append(h.make_node("Add", ["c0x2", "four"], ["Sc"]))
nodes.append(h.make_node("Unsqueeze", ["idx", "u1"], ["idx_col"]))
nodes.append(h.make_node("Unsqueeze", ["idx", "u0"], ["idx_row"]))
nodes.append(h.make_node("Add", ["idx_col", "idx_row"], ["ipj"]))
nodes.append(h.make_node("Equal", ["ipj", "Sr"], ["Pr_b"]))
nodes.append(h.make_node("Cast", ["Pr_b"], ["Pr"], to=TP.FLOAT16))
nodes.append(h.make_node("Equal", ["ipj", "Sc"], ["Pc_b"]))
nodes.append(h.make_node("Cast", ["Pc_b"], ["Pc"], to=TP.FLOAT16))
nodes.append(h.make_node("Sub", ["idx_col", "idx_row"], ["imj"]))
nodes.append(h.make_node("Sub", ["r0", "c0"], ["delta"]))
nodes.append(h.make_node("Equal", ["imj", "delta"], ["Srow_b"]))
nodes.append(h.make_node("Cast", ["Srow_b"], ["Srow"], to=TP.FLOAT16))

nodes.append(h.make_node("MatMul", ["Pr", "g2d"], ["Rg"]))
nodes.append(h.make_node("MatMul", ["g2d", "Pc"], ["Cg"]))
nodes.append(h.make_node("MatMul", ["Rg", "Pc"], ["RCg"]))
nodes.append(h.make_node("Max", ["g2d", "Rg", "Cg", "RCg"], ["Shv"]))
nodes.append(h.make_node("Transpose", ["Shv"], ["ShvT"], perm=[1, 0]))
nodes.append(h.make_node("MatMul", ["Srow", "ShvT"], ["tmpT"]))
nodes.append(h.make_node("MatMul", ["tmpT", "Srow"], ["Thv"]))
nodes.append(h.make_node("Max", ["Shv", "Thv"], ["gm2d"]))

nodes.append(h.make_node("Cast", ["gm2d"], ["gm2d_u8"], to=TP.UINT8))
nodes.append(h.make_node("Unsqueeze", ["gm2d_u8", "sq01"], ["gm4"]))
nodes.append(h.make_node("Pad", ["gm4", "pads_grid", "pad0u8"], ["gm10"]))
nodes.append(h.make_node("Pad", ["gm10", "pads_out", "sent255"], ["gm30"]))
nodes.append(h.make_node("Equal", ["gm30", "c_colors"], ["output"]))

graph = h.make_graph(nodes, "task020_direct",
                     [h.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
                     [h.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])], inits)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/p1b_task020.onnx")
print("saved", len(model.graph.node), "nodes", len(inits), "inits")
