"""Build minimal ONNX for task175 - design B (int8, 30x30 self-kill, Equal=output free).

output[r,c] = ((D-1 + blk[r,c]) mod P) + 1   for r,c in 21x21 ; padding -> empty.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

H = W = 30
N = 21

blk = np.zeros((H, W), dtype=np.int8)
for i in range(N):
    for j in range(N):
        a = min(i, j); b = max(i, j); dd = b - a
        if dd == 0:
            bb = 0
        else:
            w6 = a + 1
            bb = -1 if dd <= w6 else (dd - w6 - 1) // (a + 2)
        blk[i, j] = bb

# offset added AFTER mod: 1 inside grid, 101 in padding (so it never matches a real color)
offset = np.full((H, W), 101, dtype=np.int8)
offset[:N, :N] = 1

# channel value table for one-hot: channel 0 = sentinel (never matches), 1..9 = colors
arange = np.array([-100, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int8).reshape(1, 10, 1, 1)

nodes, inits, vis = [], [], []

def add_init(name, arr, tp):
    inits.append(helper.make_tensor(name, tp, arr.shape, arr.flatten().tolist()))

def add_vi(name, shape, tp):
    vis.append(helper.make_tensor_value_info(name, tp, shape))

I8, I64, F32, BOOL = TensorProto.INT8, TensorProto.INT64, TensorProto.FLOAT, TensorProto.BOOL

add_init("blk", blk.reshape(1, 1, H, W), I8)
add_init("offset", offset.reshape(1, 1, H, W), I8)
add_init("arange", arange, I8)
add_init("rng10f", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
add_init("ax1", np.array([1], dtype=np.int64), I64)
add_init("c_starts", np.array([0, 0], dtype=np.int64), I64)
add_init("c_ends", np.array([1, 1], dtype=np.int64), I64)
add_init("c_axes", np.array([2, 3], dtype=np.int64), I64)

# corner = input[:,:,0:1,0:1] -> [1,10,1,1]
nodes.append(helper.make_node("Slice", ["input", "c_starts", "c_ends", "c_axes"], ["corner"]))
add_vi("corner", [1, 10, 1, 1], F32)
# present = ReduceMax(input, axes=2,3) -> [1,10,1,1]
nodes.append(helper.make_node("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1))
add_vi("present", [1, 10, 1, 1], F32)
# Df = ReduceSum(corner*rng10f, axis1)  ; Pf = ReduceMax(present*rng10f, axis1)
nodes.append(helper.make_node("Mul", ["corner", "rng10f"], ["cw"])); add_vi("cw", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceSum", ["cw", "ax1"], ["Df"], keepdims=1)); add_vi("Df", [1, 1, 1, 1], F32)
nodes.append(helper.make_node("Mul", ["present", "rng10f"], ["pw"])); add_vi("pw", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceMax", ["pw"], ["Pf"], axes=[1], keepdims=1)); add_vi("Pf", [1, 1, 1, 1], F32)
# s = D-1 (as int8) ; P (as int8)
nodes.append(helper.make_node("Cast", ["Df"], ["Di8"], to=I8)); add_vi("Di8", [1, 1, 1, 1], I8)
nodes.append(helper.make_node("Cast", ["Pf"], ["P"], to=I8)); add_vi("P", [1, 1, 1, 1], I8)
add_init("one8", np.array([1], dtype=np.int8), I8)
nodes.append(helper.make_node("Sub", ["Di8", "one8"], ["s"])); add_vi("s", [1, 1, 1, 1], I8)
# t = blk + s  ; m = Mod(t,P) ; col = m + offset
nodes.append(helper.make_node("Add", ["blk", "s"], ["t"])); add_vi("t", [1, 1, H, W], I8)
nodes.append(helper.make_node("Mod", ["t", "P"], ["m"])); add_vi("m", [1, 1, H, W], I8)
nodes.append(helper.make_node("Add", ["m", "offset"], ["col"])); add_vi("col", [1, 1, H, W], I8)
# output = Equal(col, arange) -> [1,10,30,30] bool (excluded)
nodes.append(helper.make_node("Equal", ["col", "arange"], ["output"]))

graph = helper.make_graph(
    nodes, "task175",
    [helper.make_tensor_value_info("input", F32, [1, 10, H, W])],
    [helper.make_tensor_value_info("output", BOOL, [1, 10, H, W])],
    inits, value_info=vis)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b4_task175.onnx")
print("saved")
