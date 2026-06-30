"""Minimal ONNX for task398 (anti-diagonal value spread). opset 10.

Rule: input 1x5 row, k=#nonzero, N=5k. Output NxN; along anti-diagonal d=r+c the
value is vals[d-N+1] (else background 0); cells outside NxN are empty (trim away).

Build: output = Gather(Rpad[1,10,8], J[30,30], axis=2) -> [1,10,30,30] == "output".
Rpad columns (axis=2): 0=bg(channel0), 1..5=input row cols0..4, 6=bg, 7=all-zero.
Inner index Jin = clip(r+c+2-N, 0, 6); outside NxN -> 7 (zero col) via Max markers.
All 30x30 arithmetic in float16; one Cast to int32 for the gather index.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H = W = 30
C = 10
inits = []


def add_init(name, arr):
    inits.append(numpy_helper.from_array(arr, name))
    return name


# const S2[r,c] = r+c+2 (float16) -> Jin = clip(S2-N, 0, 6)
S2 = (np.add.outer(np.arange(H), np.arange(W)) + 2).astype(np.float16)
add_init("S2", S2)
add_init("rc_f", np.arange(H, dtype=np.float16).reshape(H, 1))   # [30,1] f16
add_init("cc_f", np.arange(W, dtype=np.float16).reshape(1, W))   # [1,30] f16
add_init("five_h", np.array(5, dtype=np.float16))
add_init("seven_h", np.array(7, dtype=np.float16))
add_init("zero_h", np.array(0, dtype=np.float16))

nodes = []

# input row -> [1,10,1,5] -> reshape [1,10,5]
add_init("s_starts", np.array([0, 0], dtype=np.int64))
add_init("s_ends", np.array([1, 5], dtype=np.int64))
add_init("s_axes", np.array([2, 3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["input", "s_starts", "s_ends", "s_axes"], ["row4"]))
add_init("shp_1_10_5", np.array([1, 10, 5], dtype=np.int64))
nodes.append(helper.make_node("Reshape", ["row4", "shp_1_10_5"], ["row"]))

# k = sum of foreground one-hots (channels 1..9), as f16
add_init("k_starts", np.array([1], dtype=np.int64))
add_init("k_ends", np.array([10], dtype=np.int64))
add_init("k_axes", np.array([1], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["row", "k_starts", "k_ends", "k_axes"], ["rowfg"]))
nodes.append(helper.make_node("ReduceSum", ["rowfg"], ["k32"], keepdims=0))   # scalar f32
nodes.append(helper.make_node("Cast", ["k32"], ["kh"], to=TensorProto.FLOAT16))
nodes.append(helper.make_node("Mul", ["kh", "five_h"], ["Nh"]))               # scalar f16 = N

# Jin = clip(S2 - N, 0, 6)
nodes.append(helper.make_node("Sub", ["S2", "Nh"], ["M"]))                     # [30,30] f16
nodes.append(helper.make_node("Clip", ["M"], ["Jin"], min=0.0, max=6.0))        # [30,30] f16

# outside markers: 7 where r>=N (resp c>=N) else 0  (small [30,1]/[1,30])
nodes.append(helper.make_node("Less", ["rc_f", "Nh"], ["r_lt"]))   # [30,1] bool
nodes.append(helper.make_node("Less", ["cc_f", "Nh"], ["c_lt"]))   # [1,30] bool
nodes.append(helper.make_node("Where", ["r_lt", "zero_h", "seven_h"], ["rN"]))  # [30,1] f16
nodes.append(helper.make_node("Where", ["c_lt", "zero_h", "seven_h"], ["cN"]))  # [1,30] f16

# J = cast(max(Jin, rN, cN)) -> int32
nodes.append(helper.make_node("Max", ["Jin", "rN", "cN"], ["Jf"]))              # [30,30] f16
nodes.append(helper.make_node("Cast", ["Jf"], ["J"], to=TensorProto.INT32))     # [30,30] int32

# Rpad [1,10,8] = concat(bg, row[1,10,5], bg, zero) along axis=2
bg = np.zeros((1, C, 1), dtype=np.float32); bg[0, 0, 0] = 1.0
add_init("bg_low", bg)
add_init("bg_high", bg.copy())
add_init("zero_col", np.zeros((1, C, 1), dtype=np.float32))
nodes.append(helper.make_node("Concat", ["bg_low", "row", "bg_high", "zero_col"], ["Rpad"], axis=2))

# Gather -> output [1,10,30,30]
nodes.append(helper.make_node("Gather", ["Rpad", "J"], ["output"], axis=2))

x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "task398", [x], [y], inits)
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b4_task398.onnx")
print("saved nodes", len(nodes), "inits", len(inits))
