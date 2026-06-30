"""task175 - design C: crop-21 arithmetic, pad index to 30x30, Equal=output (free).

color[r,c] = ((D-1 + blk[r,c]) mod P) + 1, 21x21 ; padding -> empty.
Compare padded index m against arange_idx = [sentinel, 0,1,..,8] (channel k <-> index k-1).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

N = 21
H = W = 30
PADR = H - N  # 9

blk = np.zeros((N, N), dtype=np.int32)
for i in range(N):
    for j in range(N):
        a = min(i, j); b = max(i, j); dd = b - a
        if dd == 0:
            bb = 0
        else:
            w6 = a + 1
            bb = -1 if dd <= w6 else (dd - w6 - 1) // (a + 2)
        blk[i, j] = bb

# channel value table: channel0 sentinel; channel k(1..9) <-> index k-1
arange_idx = np.array([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8).reshape(1, 10, 1, 1)

I8, I32, I64, F32, BOOL = (TensorProto.INT8, TensorProto.INT32, TensorProto.INT64,
                           TensorProto.FLOAT, TensorProto.BOOL)
nodes, inits, vis = [], [], []

def add_init(name, arr, tp):
    inits.append(helper.make_tensor(name, tp, arr.shape, arr.flatten().tolist()))

def add_vi(name, shape, tp):
    vis.append(helper.make_tensor_value_info(name, tp, shape))

add_init("blk", blk.reshape(1, 1, N, N), I32)
add_init("arange", arange_idx, I8)
add_init("rng10f", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
add_init("ax1", np.array([1], dtype=np.int64), I64)
add_init("one32", np.array([1], dtype=np.int32), I32)
add_init("c_starts", np.array([0, 0], dtype=np.int64), I64)
add_init("c_ends", np.array([1, 1], dtype=np.int64), I64)
add_init("c_axes", np.array([2, 3], dtype=np.int64), I64)
# pad amounts: [b,c,h_begin,w_begin, b,c,h_end,w_end]
add_init("pads", np.array([0, 0, 0, 0, 0, 0, PADR, PADR], dtype=np.int64), I64)
add_init("padval", np.array(-50, dtype=np.int8), I8)

# corner color D and period P (float)
nodes.append(helper.make_node("Slice", ["input", "c_starts", "c_ends", "c_axes"], ["corner"]))
add_vi("corner", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1))
add_vi("present", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("Mul", ["corner", "rng10f"], ["cw"])); add_vi("cw", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceSum", ["cw", "ax1"], ["Df"], keepdims=1)); add_vi("Df", [1, 1, 1, 1], F32)
nodes.append(helper.make_node("Mul", ["present", "rng10f"], ["pw"])); add_vi("pw", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceMax", ["pw"], ["Pf"], axes=[1], keepdims=1)); add_vi("Pf", [1, 1, 1, 1], F32)
# to int32
nodes.append(helper.make_node("Cast", ["Df"], ["D32"], to=I32)); add_vi("D32", [1, 1, 1, 1], I32)
nodes.append(helper.make_node("Sub", ["D32", "one32"], ["s"])); add_vi("s", [1, 1, 1, 1], I32)
# t = blk + s  (int32 21x21)
nodes.append(helper.make_node("Add", ["blk", "s"], ["t"])); add_vi("t", [1, 1, N, N], I32)
# cast to int8 then Mod by P (int8)
nodes.append(helper.make_node("Cast", ["t"], ["t8"], to=I8)); add_vi("t8", [1, 1, N, N], I8)
nodes.append(helper.make_node("Cast", ["Pf"], ["P8"], to=I8)); add_vi("P8", [1, 1, 1, 1], I8)
nodes.append(helper.make_node("Mod", ["t8", "P8"], ["m"])); add_vi("m", [1, 1, N, N], I8)
# pad index to 30x30 with sentinel -100
nodes.append(helper.make_node("Pad", ["m", "pads", "padval"], ["mp"], mode="constant"))
add_vi("mp", [1, 1, H, W], I8)
# output one-hot
nodes.append(helper.make_node("Equal", ["mp", "arange"], ["output"]))

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
