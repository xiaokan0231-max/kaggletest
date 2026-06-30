"""Build compact ONNX for task362.

Rule (verified 267/267): input has a cross (one full row + one full column) of
color c, plus k cells of color 5 (in the right column). Output = a clean cross of
color c with the row moved DOWN by k and the column moved LEFT by k, on a black
background. k = number of 5's.

Tensor formulation (all on 10x10):
  g    = color grid (Conv weight [0..9] on one-hot input; input excluded)
  nz   = (g>0)&(g!=5)  -> the 19-cell cross mask
  c    = sum(g*nz)/19  (scalar color)
  k    = sum(g==5)     (scalar shift)
  fullrow/fullcol via rowsum/colsum == 10 -> r, col scalars
  nr = r+k ; nc = col-k
  crossmask = (rowidx==nr) | (colidx==nc)
  gm = where(crossmask, c, 0) as uint8; pad to 30x30 with 255; Equal vs colors -> output

All grids 10x10, cropped from the 30x30 input.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 10


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # 1. color grid from one-hot input (input excluded) -> g30 f32 [1,1,30,30]
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    # crop to 10x10 -> g f32 [1,1,10,10]
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g"]))

    # 2. masks. nz = cross mask = (g>0.5) & (|g-5|>0.5)
    inits.append(cf("half", 0.5))
    inits.append(cf("five", 5.0))
    nodes.append(helper.make_node("Greater", ["g", "half"], ["pos"]))      # g>0
    nodes.append(helper.make_node("Equal", ["g", "five"], ["isfive"]))     # g==5
    nodes.append(helper.make_node("Not", ["isfive"], ["notfive"]))
    nodes.append(helper.make_node("And", ["pos", "notfive"], ["nzb"]))     # bool cross
    nodes.append(helper.make_node("Cast", ["nzb"], ["nz"], to=F16))        # f16 [1,1,10,10]

    # 3. color c = sum(g*nz)/19  (scalar)
    nodes.append(helper.make_node("Cast", ["g"], ["g16"], to=F16))
    nodes.append(helper.make_node("Mul", ["g16", "nz"], ["gc"]))           # f16
    nodes.append(helper.make_node("ReduceSum", ["gc"], ["csum"], keepdims=0))
    inits.append(c16("inv19", 1.0 / 19.0))
    nodes.append(helper.make_node("Mul", ["csum", "inv19"], ["c"]))        # scalar f16

    # 4. k = sum(g==5)
    nodes.append(helper.make_node("Cast", ["isfive"], ["fivef"], to=F16))
    nodes.append(helper.make_node("ReduceSum", ["fivef"], ["k"], keepdims=0))  # scalar f16

    # 5. fullrow / fullcol -> r, col scalars
    # rowsum over axis=3 (cols) -> [1,1,10,1]; colsum over axis=2 -> [1,1,1,10]
    nodes.append(helper.make_node("ReduceSum", ["nz", "ax3"], ["rowsum"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["nz", "ax2"], ["colsum"], keepdims=1))
    inits.append(ci("ax3", [3]))
    inits.append(ci("ax2", [2]))
    inits.append(c16("nine5", 9.5))
    nodes.append(helper.make_node("Greater", ["rowsum", "nine5"], ["frb"]))  # [1,1,10,1] bool
    nodes.append(helper.make_node("Greater", ["colsum", "nine5"], ["fcb"]))  # [1,1,1,10] bool
    nodes.append(helper.make_node("Cast", ["frb"], ["fr"], to=F16))
    nodes.append(helper.make_node("Cast", ["fcb"], ["fc"], to=F16))
    # r = sum(idx_row * fr); col = sum(idx_col * fc)
    inits.append(c16("idxrow", np.arange(10, dtype=np.float16).reshape(1, 1, 10, 1)))
    inits.append(c16("idxcol", np.arange(10, dtype=np.float16).reshape(1, 1, 1, 10)))
    nodes.append(helper.make_node("Mul", ["fr", "idxrow"], ["rprod"]))
    nodes.append(helper.make_node("Mul", ["fc", "idxcol"], ["cprod"]))
    nodes.append(helper.make_node("ReduceSum", ["rprod"], ["r"], keepdims=0))   # scalar
    nodes.append(helper.make_node("ReduceSum", ["cprod"], ["col"], keepdims=0)) # scalar

    # 6. nr = r+k ; nc = col-k
    nodes.append(helper.make_node("Add", ["r", "k"], ["nr"]))   # scalar f16
    nodes.append(helper.make_node("Sub", ["col", "k"], ["nc"])) # scalar f16

    # 7. crossmask = (RIfull==nr) | (CIfull==nc) over the 30x30 grid directly.
    #    Build coordinate grids on 10x10, compare, and only keep in-grid via the
    #    color path below (out-of-grid forced to sentinel). We compute on 10x10.
    inits.append(c16("RI", np.broadcast_to(np.arange(10, dtype=np.float16).reshape(1, 1, 10, 1), (1, 1, 10, 10)).copy()))
    inits.append(c16("CI", np.broadcast_to(np.arange(10, dtype=np.float16).reshape(1, 1, 1, 10), (1, 1, 10, 10)).copy()))
    nodes.append(helper.make_node("Equal", ["RI", "nr"], ["rmask"]))  # bool [1,1,10,10]
    nodes.append(helper.make_node("Equal", ["CI", "nc"], ["cmask"]))
    nodes.append(helper.make_node("Or", ["rmask", "cmask"], ["crossb"]))  # bool [1,1,10,10]

    # 8. gm = where(crossb, round(c), 0) as uint8. Round c to nearest int first.
    #    c is exact-ish from /19; round via Cast to nearest? Cast truncates. Add 0.5.
    inits.append(c16("rhalf", 0.5))
    nodes.append(helper.make_node("Add", ["c", "rhalf"], ["crnd"]))
    nodes.append(helper.make_node("Cast", ["crnd"], ["cu8"], to=U8))   # scalar uint8 color
    inits.append(numpy_helper.from_array(np.asarray(0, np.uint8), "zero8"))
    nodes.append(helper.make_node("Where", ["crossb", "cu8", "zero8"], ["gm"]))  # u8 [1,1,10,10]

    # 9. pad to 30x30 with sentinel 255
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(255, np.uint8), "sent"))
    nodes.append(helper.make_node("Pad", ["gm", "pads", "sent"], ["gm30"]))  # u8 [1,1,30,30]

    # 10. output one-hot via Equal (graph output -> excluded)
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))  # bool [1,10,30,30]

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
