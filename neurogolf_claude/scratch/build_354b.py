"""task354 v3: Conv full input -> f32 30x30, cast uint8, slice to 10x10. Route B."""
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as h

N_ITER = 9
W = 10
nodes = []
inits = []
def const(name, arr):
    inits.append(onnx.numpy_helper.from_array(arr, name))

# 1. Conv full input -> g_full f32 [1,1,30,30]
wconv = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
const("convw", wconv)
nodes.append(h.make_node("Conv", ["input", "convw"], ["g_full"]))
nodes.append(h.make_node("Cast", ["g_full"], ["g8full"], to=TP.UINT8))
# slice to 10x10
const("cs", np.array([0, 0], dtype=np.int64))
const("ce", np.array([W, W], dtype=np.int64))
const("ca", np.array([2, 3], dtype=np.int64))
nodes.append(h.make_node("Slice", ["g8full", "cs", "ce", "ca"], ["g8"]))

# gray + seed
const("five8", np.array(5, dtype=np.uint8))
nodes.append(h.make_node("Equal", ["g8", "five8"], ["gray_b"]))
nodes.append(h.make_node("Cast", ["gray_b"], ["gray8"], to=TP.UINT8))
const("mrs", np.array([0], dtype=np.int64))
const("mre", np.array([1], dtype=np.int64))
const("mra", np.array([2], dtype=np.int64))
nodes.append(h.make_node("Slice", ["g8", "mrs", "mre", "mra"], ["marker_row"]))
nodes.append(h.make_node("Mul", ["gray8", "marker_row"], ["cur0"]))

cur = "cur0"
for it in range(N_ITER):
    nodes.append(h.make_node("MaxPool", [cur], [f"mp{it}"],
                             kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1]))
    nxt = f"cur{it+1}"
    nodes.append(h.make_node("Mul", ["gray8", f"mp{it}"], [nxt]))
    cur = nxt

nodes.append(h.make_node("Where", ["gray_b", cur, "g8"], ["recol_u8"]))
const("pad30", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64))
const("sentinel", np.array(255, dtype=np.uint8))
nodes.append(h.make_node("Pad", ["recol_u8", "pad30", "sentinel"], ["gm"], mode="constant"))
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
const("colors", colors)
nodes.append(h.make_node("Equal", ["gm", "colors"], ["output"]))

inp_vi = h.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
out_vi = h.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])
graph = h.make_graph(nodes, "task354", [inp_vi], [out_vi], inits)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b12_task354b.onnx")
print("saved nodes", len(nodes))
