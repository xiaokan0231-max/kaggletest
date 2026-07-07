"""task362 v2: leaner. Eliminate RI/CI grids; build cross from vector masks.

Cross color c, shift k = #5s, full row r, full col col.
Output cross: row r+k, col col-k, color c.
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

    # color grid -> g30 f32 [1,1,30,30]; crop to g [1,1,10,10] f32
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g"]))  # f32 100

    # masks (bool 10x10)
    inits.append(cf("half", 0.5))
    inits.append(cf("five", 5.0))
    nodes.append(helper.make_node("Greater", ["g", "half"], ["pos"]))   # bool 100
    nodes.append(helper.make_node("Equal", ["g", "five"], ["isfive"]))  # bool 100
    # nz = pos & not isfive : cross. Cast to f16 for arithmetic.
    nodes.append(helper.make_node("Not", ["isfive"], ["notfive"]))      # bool 100
    nodes.append(helper.make_node("And", ["pos", "notfive"], ["nzb"]))  # bool 100
    nodes.append(helper.make_node("Cast", ["nzb"], ["nz"], to=F16))     # f16 200

    # c = sum(g*nz)/19. Need g in f16.
    nodes.append(helper.make_node("Cast", ["g"], ["g16"], to=F16))      # f16 200
    nodes.append(helper.make_node("Mul", ["g16", "nz"], ["gc"]))        # f16 200
    nodes.append(helper.make_node("ReduceSum", ["gc"], ["csum"], keepdims=0))
    inits.append(c16("inv19", 1.0 / 19.0))
    nodes.append(helper.make_node("Mul", ["csum", "inv19"], ["c"]))     # scalar

    # k = sum(isfive)
    nodes.append(helper.make_node("Cast", ["isfive"], ["fivef"], to=F16))
    nodes.append(helper.make_node("ReduceSum", ["fivef"], ["k"], keepdims=0))

    # rowsum/colsum -> r, col
    inits.append(ci("ax3", [3]))
    inits.append(ci("ax2", [2]))
    nodes.append(helper.make_node("ReduceSum", ["nz", "ax3"], ["rowsum"], keepdims=1))  # [1,1,10,1] f16
    nodes.append(helper.make_node("ReduceSum", ["nz", "ax2"], ["colsum"], keepdims=1))  # [1,1,1,10]
    inits.append(c16("nine5", 9.5))
    nodes.append(helper.make_node("Greater", ["rowsum", "nine5"], ["frb"]))  # [1,1,10,1] bool
    nodes.append(helper.make_node("Greater", ["colsum", "nine5"], ["fcb"]))  # [1,1,1,10] bool
    nodes.append(helper.make_node("Cast", ["frb"], ["fr"], to=F16))
    nodes.append(helper.make_node("Cast", ["fcb"], ["fc"], to=F16))
    inits.append(c16("idxrow", np.arange(10, dtype=np.float16).reshape(1, 1, 10, 1)))
    inits.append(c16("idxcol", np.arange(10, dtype=np.float16).reshape(1, 1, 1, 10)))
    nodes.append(helper.make_node("Mul", ["fr", "idxrow"], ["rprod"]))
    nodes.append(helper.make_node("Mul", ["fc", "idxcol"], ["cprod"]))
    nodes.append(helper.make_node("ReduceSum", ["rprod"], ["r"], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["cprod"], ["col"], keepdims=0))

    nodes.append(helper.make_node("Add", ["r", "k"], ["nr"]))    # scalar
    nodes.append(helper.make_node("Sub", ["col", "k"], ["nc"]))  # scalar

    # vector masks: rmask1 [1,1,10,1], cmask1 [1,1,1,10]; Or broadcasts to [1,1,10,10]
    nodes.append(helper.make_node("Equal", ["idxrow", "nr"], ["rmask1"]))  # bool 10
    nodes.append(helper.make_node("Equal", ["idxcol", "nc"], ["cmask1"]))  # bool 10
    nodes.append(helper.make_node("Or", ["rmask1", "cmask1"], ["crossb"]))  # bool 100

    # gm = where(crossb, round(c), 0) uint8
    inits.append(c16("rhalf", 0.5))
    nodes.append(helper.make_node("Add", ["c", "rhalf"], ["crnd"]))
    nodes.append(helper.make_node("Cast", ["crnd"], ["cu8"], to=U8))
    inits.append(cu("zero8", 0))
    nodes.append(helper.make_node("Where", ["crossb", "cu8", "zero8"], ["gm"]))  # u8 100

    # pad to 30x30 sentinel 255
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
