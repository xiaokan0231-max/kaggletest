"""Build minimal ONNX for task175.

Rule: output[r,c] = ((D-1 + blk[r,c]) mod P) + 1   for r,c in 21x21
  D = corner color (main-diagonal color) = argmax over channels of input[:, :, 0,0]
  P = max present color = max color channel that appears anywhere
  blk = fixed 21x21 offset matrix (function of position only)
Padding region (r or c >= 21) -> empty (all channels 0).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

H = W = 30
N = 21

# ---- build blk constant (30x30) ----
blk = np.zeros((H, W), dtype=np.int64)
for i in range(N):
    for j in range(N):
        a = min(i, j); b = max(i, j); dd = b - a
        if dd == 0:
            bb = 0
        else:
            w6 = a + 1
            if dd <= w6:
                bb = -1
            else:
                k = dd - w6 - 1
                bb = k // (a + 2)
        blk[i, j] = bb
# padding region stays 0 but will be masked out

# region mask: 1 in 21x21, 0 else
region = np.zeros((H, W), dtype=np.int64)
region[:N, :N] = 1

# arange for one-hot compare; channel 0 sentinel so it never matches
arange = np.array([255, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64).reshape(1, 10, 1, 1)

nodes = []
inits = []
vis = []  # value_info


def add_init(name, arr, tp=TensorProto.INT64):
    inits.append(helper.make_tensor(name, tp, arr.shape, arr.flatten().tolist()))


def add_vi(name, shape, tp=TensorProto.INT64):
    vis.append(helper.make_tensor_value_info(name, tp, shape))


# constants
add_init("blk", blk.reshape(1, 1, H, W))
add_init("region", region.reshape(1, 1, H, W))
add_init("arange", arange)
add_init("rng10", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1))
add_init("one_i", np.array([1], dtype=np.int64))
add_init("ax1", np.array([1], dtype=np.int64))

# input is float32 [1,10,30,30]
# 1) corner = input[:, :, 0:1, 0:1]  -> [1,10,1,1] float
add_init("c_starts", np.array([0, 0], dtype=np.int64))
add_init("c_ends", np.array([1, 1], dtype=np.int64))
add_init("c_axes", np.array([2, 3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["input", "c_starts", "c_ends", "c_axes"], ["corner"]))
add_vi("corner", [1, 10, 1, 1], TensorProto.FLOAT)

# 2) present = ReduceMax(input, axes=[2,3]) -> [1,10,1,1] float
nodes.append(helper.make_node("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1))
add_vi("present", [1, 10, 1, 1], TensorProto.FLOAT)

# cast rng10 to float for weighting
nodes.append(helper.make_node("Cast", ["rng10"], ["rng10f"], to=TensorProto.FLOAT))
add_vi("rng10f", [1, 10, 1, 1], TensorProto.FLOAT)

# D = ReduceSum(corner * rng10f, axis=1)  -> [1,1,1,1] float
nodes.append(helper.make_node("Mul", ["corner", "rng10f"], ["cw"]))
add_vi("cw", [1, 10, 1, 1], TensorProto.FLOAT)
nodes.append(helper.make_node("ReduceSum", ["cw", "ax1"], ["Df"], keepdims=1))
add_vi("Df", [1, 1, 1, 1], TensorProto.FLOAT)

# P = ReduceMax(present * rng10f, axis=1) -> [1,1,1,1] float
nodes.append(helper.make_node("Mul", ["present", "rng10f"], ["pw"]))
add_vi("pw", [1, 10, 1, 1], TensorProto.FLOAT)
nodes.append(helper.make_node("ReduceMax", ["pw"], ["Pf"], axes=[1], keepdims=1))
add_vi("Pf", [1, 1, 1, 1], TensorProto.FLOAT)

# cast D,P to int64
nodes.append(helper.make_node("Cast", ["Df"], ["D"], to=TensorProto.INT64))
add_vi("D", [1, 1, 1, 1])
nodes.append(helper.make_node("Cast", ["Pf"], ["P"], to=TensorProto.INT64))
add_vi("P", [1, 1, 1, 1])

# s = D - 1
nodes.append(helper.make_node("Sub", ["D", "one_i"], ["s"]))
add_vi("s", [1, 1, 1, 1])

# t = blk + s  -> [1,1,30,30]
nodes.append(helper.make_node("Add", ["blk", "s"], ["t"]))
add_vi("t", [1, 1, H, W])

# m = Mod(t, P)   (python-style mod; for negatives need fmod=0 -> use Mod with default = sign of divisor)
nodes.append(helper.make_node("Mod", ["t", "P"], ["m"]))
add_vi("m", [1, 1, H, W])

# col = (m + 1) * region
nodes.append(helper.make_node("Add", ["m", "one_i"], ["mp1"]))
add_vi("mp1", [1, 1, H, W])
nodes.append(helper.make_node("Mul", ["mp1", "region"], ["col"]))
add_vi("col", [1, 1, H, W])

# output = Equal(col, arange) -> [1,10,30,30] bool, name "output"
nodes.append(helper.make_node("Equal", ["col", "arange"], ["output"]))

graph = helper.make_graph(
    nodes, "task175",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, H, W])],
    inits, value_info=vis,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b4_task175.onnx")
print("saved /tmp/b4_task175.onnx")
