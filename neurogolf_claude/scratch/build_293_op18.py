"""task293 opset-18 variant: swap L<->B via BitwiseXor(g, L^B) -> one big tensor
instead of Equal+Where (saves one [1,1,30,30] tensor). opset 18 takes Reduce* axes
as an input.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []
    inits.append(ci("ax2", [2]))
    inits.append(ci("ax3", [3]))

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    nodes.append(helper.make_node("Cast", ["g30"], ["g"], to=U8))

    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("ReduceMax", ["g", "ax3"], ["rowmax"], keepdims=1))  # [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", ["g", "ax2"], ["colmax"], keepdims=1))  # [1,1,1,30]
    nodes.append(helper.make_node("Greater", ["rowmax", "zero"], ["rowmask"]))
    nodes.append(helper.make_node("Greater", ["colmax", "zero"], ["colmask"]))

    nodes.append(helper.make_node("And", ["rowmask", "colmask"], ["real"]))  # [1,1,30,30]
    inits.append(cu("p99", 99))
    nodes.append(helper.make_node("Where", ["real", "g", "p99"], ["g_real"]))  # [1,1,30,30]

    nodes.append(helper.make_node("ReduceMin", ["g_real", "ax3"], ["rowmin"], keepdims=1))
    nodes.append(helper.make_node("ReduceMin", ["g_real", "ax2"], ["colmin"], keepdims=1))
    nodes.append(helper.make_node("Greater", ["rowmin", "zero"], ["rowfull"]))
    nodes.append(helper.make_node("Greater", ["colmin", "zero"], ["colfull"]))
    nodes.append(helper.make_node("And", ["rowfull", "rowmask"], ["line_rows"]))
    nodes.append(helper.make_node("And", ["colfull", "colmask"], ["bar_cols"]))

    nodes.append(helper.make_node("Not", ["line_rows"], ["not_line"]))
    nodes.append(helper.make_node("Not", ["bar_cols"], ["not_bar"]))
    nodes.append(helper.make_node("And", ["not_line", "rowmask"], ["barrow"]))
    nodes.append(helper.make_node("And", ["not_bar", "colmask"], ["linecol"]))
    nodes.append(helper.make_node("Where", ["barrow", "rowmax", "zero"], ["Bg"]))
    nodes.append(helper.make_node("Where", ["linecol", "colmax", "zero"], ["Lg"]))
    nodes.append(helper.make_node("ReduceMax", ["Bg", "ax2"], ["B"], keepdims=1))   # [1,1,1,1]
    nodes.append(helper.make_node("ReduceMax", ["Lg", "ax3"], ["L"], keepdims=1))   # [1,1,1,1]
    nodes.append(helper.make_node("BitwiseXor", ["L", "B"], ["K"]))                 # scalar L^B

    nodes.append(helper.make_node("And", ["line_rows", "bar_cols"], ["inter"]))     # [1,1,30,30]
    nodes.append(helper.make_node("BitwiseXor", ["g", "K"], ["swapped"]))           # [1,1,30,30]
    nodes.append(helper.make_node("Where", ["inter", "swapped", "g_real"], ["gm"])) # [1,1,30,30]

    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t293", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task293_op18.onnx")
