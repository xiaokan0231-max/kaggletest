"""task150: width-aware horizontal flip (fliplr within occupied WxW block).
Build a compact ONNX. Strategy:
  - Conv collapse one-hot -> color grid g [1,1,30,30] f32 (irreducible 3600B input cost).
  - Conv with ones -> present count grid; but present per-cell = sum of channels = 1 inside grid.
  - W = max row-sum of present.
  - reversal matrix M[30,30], M[k,c]=1 iff k+c==W-1; flipped = g (matmul over width) -> reversed rows.
  - gm = where(present, flipped, 255 sentinel) as color grid.
  - output one-hot = Equal(gm, colors 0..9) -> bool [1,10,30,30] = graph OUTPUT (excluded).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H = W = 30
N = 10


def const(name, arr):
    return numpy_helper.from_array(arr, name=name)


nodes = []
inits = []

# weight to collapse one-hot to color grid: [1,10,1,1] = [0..9]
w_color = np.arange(N, dtype=np.float32).reshape(1, N, 1, 1)
inits.append(const("w_color", w_color))
nodes.append(helper.make_node("Conv", ["input", "w_color"], ["g"], name="g"))  # [1,1,30,30] f32

# present: conv with all ones [1,10,1,1] -> per-cell sum of channels = 1 inside grid, 0 outside
w_ones = np.ones((1, N, 1, 1), dtype=np.float32)
inits.append(const("w_ones", w_ones))
nodes.append(helper.make_node("Conv", ["input", "w_ones"], ["pres"], name="pres"))  # [1,1,30,30] f32

# W = max row-sum of present. ReduceSum over width (axis=3) -> [1,1,30,1]; ReduceMax over all -> scalar
inits.append(const("axes3", np.array([3], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["pres", "axes3"], ["rowsum"], name="rowsum", keepdims=0))  # [1,1,30]
nodes.append(helper.make_node("ReduceMax", ["rowsum"], ["Wf"], name="Wf", keepdims=0))  # scalar f32

# Build M[30,30]: need k+c grid compared to W-1.
# Kgrid[k,c]=k+c using a const KC [30,30]; M = Equal(KC, W-1)
KC = (np.arange(W)[:, None] + np.arange(W)[None, :]).astype(np.float32)  # [30,30]
inits.append(const("KC", KC))
inits.append(const("one_f", np.array(1.0, dtype=np.float32)))
nodes.append(helper.make_node("Sub", ["Wf", "one_f"], ["Wm1"], name="Wm1"))  # scalar
nodes.append(helper.make_node("Equal", ["KC", "Wm1"], ["Mb"], name="Mb"))  # bool [30,30]
nodes.append(helper.make_node("Cast", ["Mb"], ["M"], name="M", to=TensorProto.FLOAT))  # [30,30] f32

# flipped = g @ M over width. g is [1,1,30,30]; matmul last dim: out[...,i,c]=sum_k g[...,i,k] M[k,c]
nodes.append(helper.make_node("MatMul", ["g", "M"], ["flipped"], name="flipped"))  # [1,1,30,30] f32

# gm = where(pres==1, flipped, 255)
inits.append(const("c255", np.array(255.0, dtype=np.float32)))
inits.append(const("half", np.array(0.5, dtype=np.float32)))
# pres>0.5 -> bool present mask
nodes.append(helper.make_node("Greater", ["pres", "half"], ["pmask"], name="pmask"))  # bool [1,1,30,30]
nodes.append(helper.make_node("Where", ["pmask", "flipped", "c255"], ["gm"], name="gm"))  # [1,1,30,30] f32

# output one-hot: Equal(gm, colors[1,10,1,1]) broadcast -> [1,10,30,30] bool == OUTPUT
colors = np.arange(N, dtype=np.float32).reshape(1, N, 1, 1)
inits.append(const("colors", colors))
nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"], name="output"))  # bool [1,10,30,30]

graph = helper.make_graph(
    nodes, "task150",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, N, H, W])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, N, H, W])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b17_task150.onnx")
print("saved")
