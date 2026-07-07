"""task362 v3: compute c via ReduceMax(remove-5s), reuse nocolor5 grid for full
row/col detection (threshold 9.5*c) so we never materialize a separate nz grid.

c=color, k=#5s, r=full-row idx, col=full-col idx. Output cross at (r+k, col-k).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 10


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g"]))  # f32 100

    inits.append(cf("five", 5.0))
    nodes.append(helper.make_node("Equal", ["g", "five"], ["eq5"]))      # bool 100
    nodes.append(helper.make_node("Cast", ["g"], ["g16"], to=F16))       # f16 200
    inits.append(c16("z16", 0.0))
    nodes.append(helper.make_node("Where", ["eq5", "z16", "g16"], ["nc5"]))  # f16 200 (5s -> 0)
    # c = max over grid (only color left is c on cross, 0 elsewhere)
    nodes.append(helper.make_node("ReduceMax", ["nc5"], ["c"], keepdims=0))   # scalar f16
    # k from arithmetic: sum(g16) = 19*c + 5*k => k = (sum - 19c)/5 (no extra grid)
    nodes.append(helper.make_node("ReduceSum", ["g16"], ["sumg"], keepdims=0))
    inits.append(c16("n19", 19.0))
    inits.append(c16("five16", 5.0))
    nodes.append(helper.make_node("Mul", ["c", "n19"], ["c19"]))
    nodes.append(helper.make_node("Sub", ["sumg", "c19"], ["fivek"]))
    nodes.append(helper.make_node("Div", ["fivek", "five16"], ["k"]))  # exact: 5k/5

    # full row / col via nc5 rowsum/colsum vs 9.5*c
    inits.append(ci("ax3", [3]))
    inits.append(ci("ax2", [2]))
    nodes.append(helper.make_node("ReduceSum", ["nc5", "ax3"], ["rowsum"], keepdims=1))  # [1,1,10,1]
    nodes.append(helper.make_node("ReduceSum", ["nc5", "ax2"], ["colsum"], keepdims=1))  # [1,1,1,10]
    inits.append(c16("t95", 9.5))
    nodes.append(helper.make_node("Mul", ["c", "t95"], ["thr"]))         # scalar = 9.5*c
    nodes.append(helper.make_node("Greater", ["rowsum", "thr"], ["frb"]))  # [1,1,10,1] bool
    nodes.append(helper.make_node("Greater", ["colsum", "thr"], ["fcb"]))  # [1,1,1,10] bool
    nodes.append(helper.make_node("Cast", ["frb"], ["fr"], to=F16))
    nodes.append(helper.make_node("Cast", ["fcb"], ["fc"], to=F16))
    inits.append(c16("idxrow", np.arange(10, dtype=np.float16).reshape(1, 1, 10, 1)))
    inits.append(c16("idxcol", np.arange(10, dtype=np.float16).reshape(1, 1, 1, 10)))
    nodes.append(helper.make_node("Mul", ["fr", "idxrow"], ["rprod"]))
    nodes.append(helper.make_node("Mul", ["fc", "idxcol"], ["cprod"]))
    nodes.append(helper.make_node("ReduceSum", ["rprod"], ["r"], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["cprod"], ["col"], keepdims=0))

    nodes.append(helper.make_node("Add", ["r", "k"], ["nr"]))
    nodes.append(helper.make_node("Sub", ["col", "k"], ["nc"]))

    nodes.append(helper.make_node("Equal", ["idxrow", "nr"], ["rmask1"]))  # [1,1,10,1] bool
    nodes.append(helper.make_node("Equal", ["idxcol", "nc"], ["cmask1"]))  # [1,1,1,10] bool
    nodes.append(helper.make_node("Or", ["rmask1", "cmask1"], ["crossb"]))  # [1,1,10,10] bool

    inits.append(c16("rhalf", 0.5))
    nodes.append(helper.make_node("Add", ["c", "rhalf"], ["crnd"]))
    nodes.append(helper.make_node("Cast", ["crnd"], ["cu8"], to=U8))
    inits.append(cu("zero8", 0))
    nodes.append(helper.make_node("Where", ["crossb", "cu8", "zero8"], ["gm"]))  # u8 100

    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Pad", ["gm", "pads", "sent"], ["gm30"]))  # u8 900

    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t362", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b13_task362.onnx")
