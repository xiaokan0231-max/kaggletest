import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

nodes = []
inits = []
def add_const(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))

# Collapse to color grid g30 f32 [1,1,30,30] (irreducible 3600)
W_collapse = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
add_const("Wcol", W_collapse)
nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"], kernel_shape=[1, 1]))

# six on full grid -> bool, then slice to 15x15 (avoid f32 g)
add_const("c6", np.array(6.0, dtype=np.float32))
nodes.append(helper.make_node("Equal", ["g30", "c6"], ["six30"]))   # bool [1,1,30,30]
add_const("starts", np.array([0,0], dtype=np.int64))
add_const("ends",   np.array([15,15], dtype=np.int64))
add_const("axes",   np.array([2,3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["six30","starts","ends","axes"], ["six_b"]))  # bool 15x15
nodes.append(helper.make_node("Cast", ["six_b"], ["six_f"], to=TensorProto.FLOAT16))

# 4 directional prefix convs (fp16). Threshold each to bool immediately.
ones_h = np.ones((1, 1, 1, 15), dtype=np.float16)
ones_v = np.ones((1, 1, 15, 1), dtype=np.float16)
add_const("onesH", ones_h)
add_const("onesV", ones_v)
nodes.append(helper.make_node("Conv", ["six_f", "onesH"], ["left_sum"],  kernel_shape=[1,15], pads=[0,14,0,0]))
nodes.append(helper.make_node("Conv", ["six_f", "onesH"], ["right_sum"], kernel_shape=[1,15], pads=[0,0,0,14]))
nodes.append(helper.make_node("Conv", ["six_f", "onesV"], ["up_sum"],    kernel_shape=[15,1], pads=[14,0,0,0]))
nodes.append(helper.make_node("Conv", ["six_f", "onesV"], ["down_sum"],  kernel_shape=[15,1], pads=[0,0,14,0]))
add_const("zero16", np.array(0.0, dtype=np.float16))
nodes.append(helper.make_node("Greater", ["left_sum", "zero16"], ["L"]))
nodes.append(helper.make_node("Greater", ["right_sum","zero16"], ["R"]))
nodes.append(helper.make_node("Greater", ["up_sum",   "zero16"], ["U"]))
nodes.append(helper.make_node("Greater", ["down_sum", "zero16"], ["D"]))
nodes.append(helper.make_node("And", ["L", "R"], ["LR"]))
nodes.append(helper.make_node("And", ["U", "D"], ["UD"]))
nodes.append(helper.make_node("And", ["LR", "UD"], ["inside"]))
nodes.append(helper.make_node("Not", ["six_b"], ["notsix"]))
nodes.append(helper.make_node("And", ["inside", "notsix"], ["holes_b"]))   # color 4
nodes.append(helper.make_node("Or",  ["inside", "six_b"], ["filled_b"]))

# ring via MaxPool dilation (fp16)
nodes.append(helper.make_node("Cast", ["filled_b"], ["filled_f"], to=TensorProto.FLOAT16))
nodes.append(helper.make_node("MaxPool", ["filled_f"], ["dil_f"], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Greater", ["dil_f", "zero16"], ["dil_b"]))
nodes.append(helper.make_node("Not", ["filled_b"], ["notfilled"]))
nodes.append(helper.make_node("And", ["dil_b", "notfilled"], ["ring_b"]))   # color 3

# Build gm uint8 [1,1,15,15]: base = where(six,6,8); holes->4; ring->3
add_const("u6", np.array(6, dtype=np.uint8))
add_const("u8", np.array(8, dtype=np.uint8))
add_const("u4", np.array(4, dtype=np.uint8))
add_const("u3", np.array(3, dtype=np.uint8))
nodes.append(helper.make_node("Where", ["six_b", "u6", "u8"], ["gm0"]))     # 6 or 8
nodes.append(helper.make_node("Where", ["holes_b", "u4", "gm0"], ["gm1"]))
nodes.append(helper.make_node("Where", ["ring_b",  "u3", "gm1"], ["gm2"]))  # uint8 15x15

# pad to 30x30 with sentinel 255
add_const("pads", np.array([0,0,0,0, 0,0,15,15], dtype=np.int64))
add_const("u255", np.array(255, dtype=np.uint8))
nodes.append(helper.make_node("Pad", ["gm2","pads","u255"], ["gm3"], mode="constant"))

# output one-hot (graph output, excluded)
colors = np.arange(10, dtype=np.uint8).reshape(1,10,1,1)
add_const("colors", colors)
nodes.append(helper.make_node("Equal", ["gm3", "colors"], ["output"]))

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])
Y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1,10,30,30])
graph = helper.make_graph(nodes, "task125", [X], [Y], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b13_task125.onnx")
print("saved")
