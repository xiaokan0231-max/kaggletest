import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

nodes = []
inits = []
def add_const(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))

# Collapse -> g30 f32 [1,1,30,30] (3600)
add_const("Wcol", np.arange(10, dtype=np.float32).reshape(1,10,1,1))
nodes.append(helper.make_node("Conv", ["input","Wcol"], ["g30"], kernel_shape=[1,1]))

# crop f32 -> g (900). six_b = (g==6); six_u uint8
add_const("c6", np.array(6.0, dtype=np.float32))
add_const("starts", np.array([0,0], dtype=np.int64))
add_const("ends",   np.array([15,15], dtype=np.int64))
add_const("axes",   np.array([2,3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["g30","starts","ends","axes"], ["g"]))
nodes.append(helper.make_node("Equal", ["g","c6"], ["six_b"]))
nodes.append(helper.make_node("Cast", ["six_b"], ["six_u"], to=TensorProto.UINT8))

# 4 directional prefix-max via uint8 MaxPool (asymmetric pads)
nodes.append(helper.make_node("MaxPool", ["six_u"], ["L"], kernel_shape=[1,15], pads=[0,14,0,0], strides=[1,1]))
nodes.append(helper.make_node("MaxPool", ["six_u"], ["R"], kernel_shape=[1,15], pads=[0,0,0,14], strides=[1,1]))
nodes.append(helper.make_node("MaxPool", ["six_u"], ["U"], kernel_shape=[15,1], pads=[14,0,0,0], strides=[1,1]))
nodes.append(helper.make_node("MaxPool", ["six_u"], ["D"], kernel_shape=[15,1], pads=[0,0,14,0], strides=[1,1]))
# inside_u = min(L,R,U,D)  (1 iff all four directions have a 6)
nodes.append(helper.make_node("Min", ["L","R","U","D"], ["inside_u"]))
nodes.append(helper.make_node("Cast", ["inside_u"], ["inside"], to=TensorProto.BOOL))

nodes.append(helper.make_node("Not", ["six_b"], ["notsix"]))
nodes.append(helper.make_node("And", ["inside","notsix"], ["holes_b"]))          # color4

# ring via uint8 MaxPool dilation of inside_u
nodes.append(helper.make_node("MaxPool", ["inside_u"], ["dil_u"], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Cast", ["dil_u"], ["dil_b"], to=TensorProto.BOOL))
nodes.append(helper.make_node("Not", ["inside"], ["notfilled"]))
nodes.append(helper.make_node("And", ["dil_b","notfilled"], ["ring_b"]))         # color3

# gm uint8 15x15
add_const("u6", np.array(6, dtype=np.uint8))
add_const("u8", np.array(8, dtype=np.uint8))
add_const("u4", np.array(4, dtype=np.uint8))
add_const("u3", np.array(3, dtype=np.uint8))
nodes.append(helper.make_node("Where", ["six_b","u6","u8"], ["gm0"]))
nodes.append(helper.make_node("Where", ["holes_b","u4","gm0"], ["gm1"]))
nodes.append(helper.make_node("Where", ["ring_b","u3","gm1"], ["gm2"]))

# pad 30x30 sentinel + output one-hot
add_const("pads", np.array([0,0,0,0, 0,0,15,15], dtype=np.int64))
add_const("u255", np.array(255, dtype=np.uint8))
nodes.append(helper.make_node("Pad", ["gm2","pads","u255"], ["gm3"], mode="constant"))
add_const("colors", np.arange(10, dtype=np.uint8).reshape(1,10,1,1))
nodes.append(helper.make_node("Equal", ["gm3","colors"], ["output"]))

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])
Y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1,10,30,30])
graph = helper.make_graph(nodes, "task125", [X], [Y], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)], ir_version=7)
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b13_task125.onnx")
print("saved")
