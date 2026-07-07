import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# ---- helpers ----
def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)

nodes = []
inits = []

def add_const(name, arr):
    inits.append(const(name, arr))

# Input: [1,10,30,30] f32 one-hot.  Output: [1,10,30,30] bool.
# 1) collapse to color grid g [1,1,30,30] f32 via Conv weight [1,10,1,1]=[0..9]
W_collapse = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
add_const("Wcol", W_collapse)
nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g"], kernel_shape=[1, 1]))
# g is f32 [1,1,30,30]

# 2) six = (g == 6) -> we work in float for conv. Build six_f [1,1,30,30] float32 (1.0 where 6)
add_const("c6", np.array(6.0, dtype=np.float32))
nodes.append(helper.make_node("Equal", ["g", "c6"], ["six_b"]))   # bool
nodes.append(helper.make_node("Cast", ["six_b"], ["six_f"], to=TensorProto.FLOAT))

# 3) directional prefix-OR via Conv with look-back/forward ones kernel size 15.
# left_any: sum over columns c-14..c ; kernel [1,1,1,15] ones, pad left=14 right=0
ones_h = np.ones((1, 1, 1, 15), dtype=np.float32)   # horizontal kernel
ones_v = np.ones((1, 1, 15, 1), dtype=np.float32)   # vertical kernel
add_const("onesH", ones_h)
add_const("onesV", ones_v)

# Conv pads format: [x1_begin, x2_begin, x1_end, x2_end] = [top,left,bottom,right]
nodes.append(helper.make_node("Conv", ["six_f", "onesH"], ["left_sum"], kernel_shape=[1,15], pads=[0,14,0,0]))
nodes.append(helper.make_node("Conv", ["six_f", "onesH"], ["right_sum"], kernel_shape=[1,15], pads=[0,0,0,14]))
nodes.append(helper.make_node("Conv", ["six_f", "onesV"], ["up_sum"],   kernel_shape=[15,1], pads=[14,0,0,0]))
nodes.append(helper.make_node("Conv", ["six_f", "onesV"], ["down_sum"], kernel_shape=[15,1], pads=[0,0,14,0]))

add_const("zero", np.array(0.0, dtype=np.float32))
nodes.append(helper.make_node("Greater", ["left_sum", "zero"], ["L"]))
nodes.append(helper.make_node("Greater", ["right_sum", "zero"], ["R"]))
nodes.append(helper.make_node("Greater", ["up_sum", "zero"], ["U"]))
nodes.append(helper.make_node("Greater", ["down_sum", "zero"], ["D"]))

nodes.append(helper.make_node("And", ["L", "R"], ["LR"]))
nodes.append(helper.make_node("And", ["U", "D"], ["UD"]))
nodes.append(helper.make_node("And", ["LR", "UD"], ["inside"]))   # bool, "filled" minus need six
# filled = inside | six ; holes = inside & ~six (since six implies inside? six has six to all dirs -> yes)
# hole = inside AND NOT six
nodes.append(helper.make_node("Not", ["six_b"], ["notsix"]))
nodes.append(helper.make_node("And", ["inside", "notsix"], ["holes_b"]))   # -> color 4

# filled = inside (already includes six because six -> all four dirs true) OR six
nodes.append(helper.make_node("Or", ["inside", "six_b"], ["filled_b"]))

# 4) ring = dilate(filled, 3x3) & ~filled -> color 3
# dilation via MaxPool on float
nodes.append(helper.make_node("Cast", ["filled_b"], ["filled_f"], to=TensorProto.FLOAT))
nodes.append(helper.make_node("MaxPool", ["filled_f"], ["dil_f"], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Greater", ["dil_f", "zero"], ["dil_b"]))
nodes.append(helper.make_node("Not", ["filled_b"], ["notfilled"]))
nodes.append(helper.make_node("And", ["dil_b", "notfilled"], ["ring_b"]))   # -> color 3

# 5) Build color grid gm uint8 [1,1,30,30]:
#    start from g rounded to uint8 (8 bg, 6 shape) ; set holes->4 ; ring->3
nodes.append(helper.make_node("Cast", ["g"], ["gm0"], to=TensorProto.UINT8))   # 8 or 6
# where holes -> 4
add_const("u4", np.array(4, dtype=np.uint8))
add_const("u3", np.array(3, dtype=np.uint8))
nodes.append(helper.make_node("Where", ["holes_b", "u4", "gm0"], ["gm1"]))
nodes.append(helper.make_node("Where", ["ring_b", "u3", "gm1"], ["gm2"]))   # uint8 [1,1,30,30]

# 6) Restrict to 15x15 region: out-of-grid cells must become a sentinel so output one-hot is all-zero there.
#    Build a mask in_grid [1,1,30,30] uint8: 1 for r<15 and c<15.
in_grid = np.zeros((1,1,30,30), dtype=bool)
in_grid[:,:,:15,:15] = True
add_const("ingrid", in_grid)
add_const("u255", np.array(255, dtype=np.uint8))
nodes.append(helper.make_node("Where", ["ingrid", "gm2", "u255"], ["gm3"]))   # uint8

# 7) output one-hot: Equal(gm3 broadcast, colors[1,10,1,1] uint8 0..9) -> bool [1,10,30,30]
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
