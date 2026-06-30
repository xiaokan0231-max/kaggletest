"""Build compact ONNX for task078.

Rule (verified 262/262 arc-gen + train/test), purely per-column:
  Let g be the color grid. In each column there is a contiguous run of 1s from
  the top (length t) and a stack of k cells of color 2 at the bottom. The output
  keeps the 1-shape unchanged and places k cells of color 2 immediately below the
  top 1-run, i.e. at rows [t, t+k).

ONNX formulation (per column):
  is1 = (g==1), is2 = (g==2)
  k  = sum_r is2                       # cells to drop in
  leading1[r,c] = 1 iff cumsum(is1 down to r) == r+1   (all of rows 0..r are 1)
  t  = sum_r leading1
  d  = rowidx - t
  two = (d>=0) & (d<k)
  out color = 1*is1 + 2*two            (two-cells are never 1s)

All grids are 10x10, so all interior tensors are cropped to 10x10. The only big
tensor is g30 [1,1,30,30] f32 (3600B, the irreducible Conv-from-input cost).
Everything else is f16/bool on 10x10 (<=200B). The final Equal that materialises
the [1,10,30,30] one-hot IS the graph output (excluded from memory).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 10  # all grids are 10x10


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # 1) color grid from full f32 input (input excluded) -> g30 [1,1,30,30] f32
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))  # f32 [1,1,30,30]
    # crop to 10x10, cast to f16
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["gf"]))  # f32 [1,1,10,10]

    # 2) is1, is2 as f16 (0/1) -- compare the f32 crop directly (no f16 color grid)
    inits.append(cf("one", 1.0))
    inits.append(cf("two", 2.0))
    nodes.append(helper.make_node("Equal", ["gf", "one"], ["is1b"]))   # bool
    nodes.append(helper.make_node("Equal", ["gf", "two"], ["is2b"]))   # bool
    nodes.append(helper.make_node("Cast", ["is1b"], ["is1"], to=F16))  # f16 [1,1,10,10]
    nodes.append(helper.make_node("Cast", ["is2b"], ["is2"], to=F16))  # f16 [1,1,10,10]

    # 3) k per column = sum over rows (axis=2), keepdims -> [1,1,1,10]
    inits.append(ci("ax2", [2]))
    nodes.append(helper.make_node("ReduceSum", ["is2", "ax2"], ["k"], keepdims=1))  # f16 [1,1,1,10]

    # 4) leading1: cumsum of is1 down rows == rowp1.
    #    cumsum-down-rows via lower-triangular MatMul: cs1[r,c]=sum_{i<=r} is1[i,c]
    L = np.tril(np.ones((S, S), np.float16)).reshape(1, 1, S, S)
    inits.append(c16("L", L))
    nodes.append(helper.make_node("MatMul", ["L", "is1"], ["cs1"]))  # f16 [1,1,10,10]
    rowp1 = np.arange(1, S + 1, dtype=np.float16).reshape(1, 1, S, 1)
    inits.append(c16("rowp1", rowp1))
    nodes.append(helper.make_node("Equal", ["cs1", "rowp1"], ["leadb"]))  # bool
    nodes.append(helper.make_node("Cast", ["leadb"], ["lead"], to=F16))   # f16 [1,1,10,10]
    nodes.append(helper.make_node("ReduceSum", ["lead", "ax2"], ["t"], keepdims=1))  # f16 [1,1,1,10]

    # 5) d = rowidx - t ; two = (d>=0) & (d<k)
    rowidx = np.arange(S, dtype=np.float16).reshape(1, 1, S, 1)
    inits.append(c16("rowidx", rowidx))
    nodes.append(helper.make_node("Sub", ["rowidx", "t"], ["d"]))  # f16 [1,1,10,10] (broadcast)
    inits.append(c16("zero", 0.0))
    nodes.append(helper.make_node("GreaterOrEqual", ["d", "zero"], ["ge0"]))  # bool
    nodes.append(helper.make_node("Less", ["d", "k"], ["ltk"]))               # bool (broadcast)
    nodes.append(helper.make_node("And", ["ge0", "ltk"], ["twob"]))           # bool [1,1,10,10]

    # 6) output color grid gm: where two -> 2, else the 1-shape (is1). Disjoint.
    inits.append(c16("c2", 2.0))
    nodes.append(helper.make_node("Where", ["twob", "c2", "is1"], ["gmf"]))  # f16 colors 0/1/2
    nodes.append(helper.make_node("Cast", ["gmf"], ["gmu"], to=U8))          # u8 [1,1,10,10]

    # 7) pad to 30x30 with sentinel 255 (out-of-grid matches no color) -> all-false there
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(255, np.uint8), "sent"))
    nodes.append(helper.make_node("Pad", ["gmu", "pads", "sent"], ["gm30"]))  # u8 [1,1,30,30]

    # 8) one-hot output = (gm30 == colors). Equal IS the graph output (excluded).
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t078", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b14_task078.onnx")
