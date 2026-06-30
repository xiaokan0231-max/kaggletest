import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

OPSET = [onnx.helper.make_opsetid("", 10)]
IRV = 10
nodes, inits = [], []

def add_init(name, arr):
    inits.append(nh.from_array(arr, name))

# present [1,1,30,30] (1 in-grid)
nodes.append(helper.make_node("ReduceSum", ["input"], ["present"], axes=[1], keepdims=1))
# cell = channel0
add_init("c0", np.array([0], np.int64)); add_init("c1", np.array([1], np.int64)); add_init("cax", np.array([1], np.int64))
nodes.append(helper.make_node("Slice", ["input", "c0", "c1", "cax"], ["cellf"]))

# scalar L
nodes.append(helper.make_node("ReduceMax", ["input"], ["chanmax"], axes=[2, 3], keepdims=1))
add_init("colidx", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Mul", ["chanmax", "colidx"], ["chanc"]))
nodes.append(helper.make_node("ReduceMax", ["chanc"], ["Lf"], axes=[1, 2, 3], keepdims=1))
nodes.append(helper.make_node("Cast", ["Lf"], ["Lu"], to=TensorProto.UINT8))

# bool masks
add_init("half", np.array([0.5], np.float32))
nodes.append(helper.make_node("Greater", ["present", "half"], ["presb"]))      # in-grid
nodes.append(helper.make_node("Greater", ["cellf", "half"], ["cellb"]))        # cell (color0 in-grid)
nodes.append(helper.make_node("Not", ["cellb"], ["notcell"]))
nodes.append(helper.make_node("And", ["presb", "notcell"], ["isLb"]))          # line
nodes.append(helper.make_node("Not", ["presb"], ["notpres"]))

# line positions via tiny per-row/col counts (avoid isLf [1,1,30,30])
nodes.append(helper.make_node("ReduceSum", ["present"], ["rowP"], axes=[3], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["cellf"], ["rowC"], axes=[3], keepdims=1))
nodes.append(helper.make_node("Sub", ["rowP", "rowC"], ["rowL"]))               # line count per row [1,1,30,1]
nodes.append(helper.make_node("ReduceSum", ["present"], ["colP"], axes=[2], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["cellf"], ["colC"], axes=[2], keepdims=1))
nodes.append(helper.make_node("Sub", ["colP", "colC"], ["colL"]))
add_init("two", np.array([2.0], np.float32))
nodes.append(helper.make_node("Mul", ["rowL", "two"], ["rowL2"]))
nodes.append(helper.make_node("Greater", ["rowL2", "rowP"], ["linerow"]))
nodes.append(helper.make_node("Mul", ["colL", "two"], ["colL2"]))
nodes.append(helper.make_node("Greater", ["colL2", "colP"], ["linecol"]))
nodes.append(helper.make_node("Or", ["linerow", "linecol"], ["linepos"]))
nodes.append(helper.make_node("And", ["linepos", "cellb"], ["gapb"]))

# flood (float16, ping-pong, 10 iters)
nodes.append(helper.make_node("Cast", ["gapb"], ["cur0"], to=TensorProto.FLOAT16))
nodes.append(helper.make_node("Cast", ["cellb"], ["cellh"], to=TensorProto.FLOAT16))
prev = "cur0"
for it in range(10):
    if it % 2 == 0:
        nodes.append(helper.make_node("MaxPool", [prev], [f"d{it}"], kernel_shape=[1, 3], pads=[0, 1, 0, 1]))
    else:
        nodes.append(helper.make_node("MaxPool", [prev], [f"d{it}"], kernel_shape=[3, 1], pads=[1, 0, 1, 0]))
    nxt = f"cur{it+1}"
    nodes.append(helper.make_node("Min", [f"d{it}", "cellh"], [nxt]))
    prev = nxt
nodes.append(helper.make_node("Cast", [prev], ["fourb"], to=TensorProto.BOOL))

# assemble gm in uint8 (small), cast to int32 once for Equal
add_init("u255", np.array(255, np.uint8)); add_init("u4", np.array(4, np.uint8)); add_init("u3", np.array(3, np.uint8))
nodes.append(helper.make_node("Where", ["fourb", "u4", "u3"], ["w34"]))         # uint8
nodes.append(helper.make_node("Where", ["isLb", "Lu", "w34"], ["wl"]))          # uint8
nodes.append(helper.make_node("Where", ["notpres", "u255", "wl"], ["gmu"]))     # uint8
nodes.append(helper.make_node("Cast", ["gmu"], ["gm"], to=TensorProto.INT32))
add_init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "task198", [inp], [out], inits)
model = helper.make_model(graph, ir_version=IRV, opset_imports=OPSET)
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b19_task198.onnx")
print("saved nodes", len(nodes), "inits", len(inits))
