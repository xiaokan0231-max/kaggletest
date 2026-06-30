import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT; BOOL = TensorProto.BOOL
H, W = 3, 5
def ci(name, v): return numpy_helper.from_array(np.asarray(v, np.int64), name=name)

def build(path):
    nodes, inits = [], []
    inits += [ci("is",[2,0,0]), ci("ie",[3,H,W]), ci("ia",[1,2,3])]
    nodes.append(helper.make_node("Slice", ["input","is","ie","ia"], ["m32"]))  # f32 [1,1,3,5] 60B
    K = np.array([[7,0,8],[0,0,0],[6,0,3]], np.float32).reshape(1,1,3,3)
    inits.append(numpy_helper.from_array(K, "Kdiag"))
    nodes.append(helper.make_node("Conv", ["m32","Kdiag"], ["gmf"], pads=[1,1,1,1]))  # f32 [1,1,3,5] 60B
    # Equal directly in float (colors as f32) -> one-hot bool [1,10,3,5]
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.float32).reshape(1,10,1,1), "colors"))
    nodes.append(helper.make_node("Equal", ["gmf","colors"], ["oh"]))  # bool [1,10,3,5] 150B
    inits.append(ci("pads",[0,0,0,0, 0,0,30-H,30-W]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "pf"))
    nodes.append(helper.make_node("Pad", ["oh","pads","pf"], ["output"]))

    x_in = helper.make_tensor_value_info("input", F32, [1,10,30,30])
    y = helper.make_tensor_value_info("output", BOOL, [1,10,30,30])
    graph = helper.make_graph(nodes, "t266", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))
build("/tmp/b16_task266.onnx")
