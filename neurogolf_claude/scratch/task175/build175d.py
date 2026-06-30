"""task175 - design D: Gather a runtime-built 12-entry color LUT by a constant blk-index map.

color = ((blk + s) mod P) + 1, s=D-1.  blk in -1..10  => idx = blk+1 in 0..11.
LUT[v] = ((v-1+s) mod P)+1 for v in 0..11. colors = Gather(LUT, blkidx). Pad, one-hot.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

N = 21
H = W = 30
PADR = H - N

blk = np.zeros((N, N), dtype=np.int64)
for i in range(N):
    for j in range(N):
        a = min(i, j); b = max(i, j); dd = b - a
        if dd == 0:
            bb = 0
        else:
            w6 = a + 1
            bb = -1 if dd <= w6 else (dd - w6 - 1) // (a + 2)
        blk[i, j] = bb
blkidx = (blk + 1).astype(np.int64)  # 0..11 indices into LUT

# v-1 for v in 0..11  -> precompute as const (vm1[v] = v-1)
vm1 = (np.arange(12) - 1).astype(np.int32)

# channel value table: channel k(1..9)<->color k ; channel0 sentinel
arange = np.array([-100, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int8).reshape(1, 10, 1, 1)

I8, I32, I64, F32, BOOL = (TensorProto.INT8, TensorProto.INT32, TensorProto.INT64,
                           TensorProto.FLOAT, TensorProto.BOOL)
nodes, inits, vis = [], [], []

def add_init(name, arr, tp):
    inits.append(helper.make_tensor(name, tp, list(arr.shape), arr.flatten().tolist()))

def add_vi(name, shape, tp):
    vis.append(helper.make_tensor_value_info(name, tp, shape))

add_init("blkidx", blkidx.reshape(1, 1, N, N), I64)
add_init("vm1", vm1, I32)                       # [12]
add_init("arange", arange, I8)
add_init("rng10f", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
add_init("ax1", np.array([1], dtype=np.int64), I64)
add_init("one32", np.array([1], dtype=np.int32), I32)
add_init("c_starts", np.array([0, 0], dtype=np.int64), I64)
add_init("c_ends", np.array([1, 1], dtype=np.int64), I64)
add_init("c_axes", np.array([2, 3], dtype=np.int64), I64)
add_init("pads", np.array([0, 0, 0, 0, 0, 0, PADR, PADR], dtype=np.int64), I64)
add_init("padval", np.array(-50, dtype=np.int8), I8)

# D (corner) and P (max present)
nodes.append(helper.make_node("Slice", ["input", "c_starts", "c_ends", "c_axes"], ["corner"]))
add_vi("corner", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1))
add_vi("present", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("Mul", ["corner", "rng10f"], ["cw"])); add_vi("cw", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceSum", ["cw", "ax1"], ["Df"], keepdims=1)); add_vi("Df", [1, 1, 1, 1], F32)
nodes.append(helper.make_node("Mul", ["present", "rng10f"], ["pw"])); add_vi("pw", [1, 10, 1, 1], F32)
nodes.append(helper.make_node("ReduceMax", ["pw"], ["Pf"], axes=[1], keepdims=1)); add_vi("Pf", [1, 1, 1, 1], F32)
nodes.append(helper.make_node("Cast", ["Df"], ["D32"], to=I32)); add_vi("D32", [1, 1, 1, 1], I32)
nodes.append(helper.make_node("Sub", ["D32", "one32"], ["s"])); add_vi("s", [1, 1, 1, 1], I32)
# reshape s to scalar for broadcast with vm1[12]
add_init("shp1", np.array([1], dtype=np.int64), I64)
nodes.append(helper.make_node("Reshape", ["s", "shp1"], ["s1"])); add_vi("s1", [1], I32)
# build LUT[12] = ((vm1 + s) mod P) + 1
nodes.append(helper.make_node("Add", ["vm1", "s1"], ["lv"])); add_vi("lv", [12], I32)   # vm1+s
nodes.append(helper.make_node("Cast", ["lv"], ["lv8"], to=I8)); add_vi("lv8", [12], I8)
nodes.append(helper.make_node("Cast", ["Pf"], ["P8"], to=I8)); add_vi("P8", [1, 1, 1, 1], I8)
add_init("P8shp", np.array([1], dtype=np.int64), I64)
nodes.append(helper.make_node("Reshape", ["P8", "P8shp"], ["P8a"])); add_vi("P8a", [1], I8)
nodes.append(helper.make_node("Mod", ["lv8", "P8a"], ["lm"])); add_vi("lm", [12], I8)    # (vm1+s) mod P
# +1 via Mod-free: need Add but int8 unsupported -> do +1 in the arange compare instead.
# So LUT stores INDEX (0..P-1); compare colors against arange_idx (channel k <-> index k-1)
# colors = Gather(lm, blkidx)
nodes.append(helper.make_node("Gather", ["lm", "blkidx"], ["g"], axis=0)); add_vi("g", [1, 1, N, N], I8)
# pad index map to 30x30 with sentinel
nodes.append(helper.make_node("Pad", ["g", "pads", "padval"], ["gp"], mode="constant"))
add_vi("gp", [1, 1, H, W], I8)
# one-hot: channel k(1..9) <-> index k-1 ; channel0 sentinel
arange_idx = np.array([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8).reshape(1, 10, 1, 1)
# replace arange init
for it in inits:
    if it.name == "arange":
        inits.remove(it); break
add_init("arange", arange_idx, I8)
nodes.append(helper.make_node("Equal", ["gp", "arange"], ["output"]))

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
