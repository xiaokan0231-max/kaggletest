"""Build compact ONNX for task279 (cropped interior to NxN).
Rule: color-1 shapes enclosing a hole -> 8; else stay 1. bg=9.
Interior tensors uint8/bool cropped to [1,1,N,N] to shrink memory.
Content is <=16x16 (top-left); N=20 gives a free border for flood seeding + margin.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

EXT_ITERS = 12
WALL_ITERS = 3
N = 17            # cropped interior size (content<=16, +border)
H = W = 30

nodes, inits = [], []
def init(name, arr): inits.append(numpy_helper.from_array(arr, name))

# g = Conv(input, [0..9]) -> color grid f32 [1,1,30,30]
init("Wcollapse", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Conv", ["input", "Wcollapse"], ["g30"], name="collapse", kernel_shape=[1, 1]))

# crop g to top-left NxN -> g [1,1,N,N]
init("st0", np.array([0, 0], np.int64))
init("en_n", np.array([N, N], np.int64))
init("ax23", np.array([2, 3], np.int64))
nodes.append(helper.make_node("Slice", ["g30", "st0", "en_n", "ax23"], ["g"], name="crop_g"))

# wall_b=(g==1); free_b=not wall; uint8 versions
init("one_f", np.array([1.0], np.float32))
nodes.append(helper.make_node("Equal", ["g", "one_f"], ["wall_b"], name="wall_eq"))
nodes.append(helper.make_node("Not", ["wall_b"], ["free_b"], name="free_not"))
nodes.append(helper.make_node("Cast", ["wall_b"], ["wall_u"], name="wall_u_cast", to=TensorProto.UINT8))
nodes.append(helper.make_node("Cast", ["free_b"], ["free_u"], name="free_u_cast", to=TensorProto.UINT8))

# seed reach = free_u * bordermask
border = np.zeros((1, 1, N, N), np.uint8)
border[0, 0, 0, :] = 1; border[0, 0, -1, :] = 1; border[0, 0, :, 0] = 1; border[0, 0, :, -1] = 1
init("bordermask", border)
nodes.append(helper.make_node("Min", ["free_u", "bordermask"], ["reach0"], name="seed"))

# exterior flood (8-conn box): reach = Min(MaxPool3x3(reach), free_u)
prev = "reach0"
for k in range(EXT_ITERS):
    dil = f"ed{k}"
    nodes.append(helper.make_node("MaxPool", [prev], [dil], name=f"em{k}",
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    nxt = f"er{k}"
    nodes.append(helper.make_node("Min", [dil, "free_u"], [nxt], name=f"ea{k}"))
    prev = nxt
reach = prev

# hole_b = free_b AND NOT reach ; hole_u
nodes.append(helper.make_node("Cast", [reach], ["reach_b"], name="reach_b_cast", to=TensorProto.BOOL))
nodes.append(helper.make_node("Not", ["reach_b"], ["nreach_b"], name="nreach_not"))
nodes.append(helper.make_node("And", ["free_b", "nreach_b"], ["hole_b"], name="hole_and"))
nodes.append(helper.make_node("Cast", ["hole_b"], ["hole_u"], name="hole_u_cast", to=TensorProto.UINT8))

def cross_dilate(src, tag):
    h = f"{tag}_h"; v = f"{tag}_v"; o = f"{tag}_c"
    nodes.append(helper.make_node("MaxPool", [src], [h], name=f"{tag}_mh",
                                  kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1]))
    nodes.append(helper.make_node("MaxPool", [src], [v], name=f"{tag}_mv",
                                  kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1]))
    nodes.append(helper.make_node("Max", [h, v], [o], name=f"{tag}_mx"))
    return o

# seed flag = wall_u AND box_dilate(hole_u)  (box catches diagonal ring cells in one step)
nodes.append(helper.make_node("MaxPool", ["hole_u"], ["hole_box"], name="hole_box_mp",
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
nodes.append(helper.make_node("Min", ["hole_box", "wall_u"], ["flag0"], name="flag_seed"))
prev = "flag0"
for k in range(WALL_ITERS):
    cd = cross_dilate(prev, f"w{k}")
    nxt = f"flag{k+1}"
    nodes.append(helper.make_node("Min", [cd, "wall_u"], [nxt], name=f"wa{k}"))
    prev = nxt
flag_u = prev
nodes.append(helper.make_node("Cast", [flag_u], ["flag_b"], name="flag_b_cast", to=TensorProto.BOOL))

# pad_b=(g==0); cg = Where(pad,255, Where(flag,8, Where(wall,1,9)))
init("zero_f", np.array([0.0], np.float32))
nodes.append(helper.make_node("Equal", ["g", "zero_f"], ["pad_b"], name="pad_eq"))
init("c255", np.array([255], np.uint8)); init("c8", np.array([8], np.uint8))
init("c1", np.array([1], np.uint8)); init("c9", np.array([9], np.uint8))
nodes.append(helper.make_node("Where", ["wall_b", "c1", "c9"], ["cg1"], name="w_wall"))
nodes.append(helper.make_node("Where", ["flag_b", "c8", "cg1"], ["cg2"], name="w_flag"))
nodes.append(helper.make_node("Where", ["pad_b", "c255", "cg2"], ["cgN"], name="w_pad"))

# Pad cgN [1,1,N,N] back to [1,1,30,30] with sentinel 255 (out-of-content -> no color)
init("pads", np.array([0, 0, 0, 0, 0, 0, H - N, W - N], np.int64))
init("c255s", np.array(255, np.uint8))
nodes.append(helper.make_node("Pad", ["cgN", "pads", "c255s"], ["cg"], name="pad_back", mode="constant"))

# output one-hot = Equal(cg, colors[1,10,1,1]) -> bool [1,10,30,30] (graph output)
init("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Equal", ["cg", "colors"], ["output"], name="onehot"))

graph = helper.make_graph(
    nodes, "task279",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, "/tmp/p1a_task279.onnx")
print("saved. nodes:", len(nodes), "inits:", len(inits))
