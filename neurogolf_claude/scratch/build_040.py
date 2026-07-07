"""Build compact ONNX for task040.

Rule (verified 266/266): two parallel solid-color border lines exist, either
vertical (col0 = colorL, col9 = colorR) or horizontal (row0 = colorT, row9 =
colorB). Every cell with color 3 is recolored to the nearest border's color
(by axis distance; no ties occur in the corpus). All grids are 10x10.

Pipeline (all interior on a single [1,1,10,10] grid):
  g30 = Conv(input, [0..9])           # f32 color grid [1,1,30,30] (input excluded)
  g   = Slice 0:10 -> cast uint8      # [1,1,10,10]
  orientation: vertical iff col0 is constant. Detect with a tiny conv/reduce.
  replacement color map cm:
     vertical : cm = where(c<=4, L, R)   (L,R column-edge colors, constant cols)
     horizontal: cm = where(r<=4, T, B)
  answer = where(g==3, cm, g)
  pad answer to 30x30 with sentinel 255; Equal(ans, colors 0..9) = output one-hot
  (the only [1,10,30,30] tensor is the excluded graph output).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
S = 10


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    # 1. color grid from one-hot input (input excluded) -> g30 [1,1,30,30] f32
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    # crop to 10x10
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["gf"]))  # f32 [1,1,10,10]
    nodes.append(helper.make_node("Cast", ["gf"], ["g"], to=U8))                 # u8 [1,1,10,10]

    # 2. edge colors (scalars via Slice on g, kept as u8 [1,1,1,1])
    #    L = g[:,:,0,0] ; R = g[:,:,0,9] ; T = g[:,:,0,0] ; B = g[:,:,9,0]
    inits += [ci("s00", [0, 0]), ci("e11", [1, 1]),
              ci("s0_9", [0, 9]), ci("e1_10", [1, 10]),
              ci("s9_0", [9, 0]), ci("e10_1", [10, 1])]
    nodes.append(helper.make_node("Slice", ["g", "s00", "e11", "ca"], ["L"]))    # u8 [1,1,1,1]
    nodes.append(helper.make_node("Slice", ["g", "s0_9", "e1_10", "ca"], ["R"]))
    nodes.append(helper.make_node("Slice", ["g", "s9_0", "e10_1", "ca"], ["B"]))
    # T == L (both top-left corner). reuse L for T.

    # 3. orientation flag: vertical iff col0 constant (all 10 rows equal).
    #    col0 = g[:,:,:,0] [1,1,10,1]. constant iff max-min == 0.
    inits += [ci("sc0", [0, 0]), ci("ec0", [10, 1])]
    nodes.append(helper.make_node("Slice", ["g", "sc0", "ec0", "ca"], ["col0"]))  # u8 [1,1,10,1]
    nodes.append(helper.make_node("Cast", ["col0"], ["col0f"], to=F32))
    nodes.append(helper.make_node("ReduceMax", ["col0f"], ["col0mx"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ReduceMin", ["col0f"], ["col0mn"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Equal", ["col0mx", "col0mn"], ["vert"]))  # bool [1,1,1,1]

    # 4. position-based replacement maps (constants).
    #    leftmask[r,c] = c<=4 ; tophalf[r,c] = r<=4  (10x10 bool)
    cols = np.arange(10).reshape(1, 1, 1, 10)
    rows = np.arange(10).reshape(1, 1, 10, 1)
    leftmask = np.broadcast_to(cols <= 4, (1, 1, 10, 10))
    tophalf = np.broadcast_to(rows <= 4, (1, 1, 10, 10))
    inits.append(numpy_helper.from_array(np.ascontiguousarray(leftmask), "leftmask"))
    inits.append(numpy_helper.from_array(np.ascontiguousarray(tophalf), "tophalf"))

    # vertical cm = where(leftmask, L, R) ; horizontal cm = where(tophalf, T(=L), B)
    nodes.append(helper.make_node("Where", ["leftmask", "L", "R"], ["cmV"]))  # u8 [1,1,10,10]
    nodes.append(helper.make_node("Where", ["tophalf", "L", "B"], ["cmH"]))   # u8 [1,1,10,10]
    nodes.append(helper.make_node("Where", ["vert", "cmV", "cmH"], ["cm"]))   # u8 [1,1,10,10]

    # 5. answer = where(g==3, cm, g)
    inits.append(cu("three", 3))
    nodes.append(helper.make_node("Equal", ["g", "three"], ["is3"]))          # bool [1,1,10,10]
    nodes.append(helper.make_node("Where", ["is3", "cm", "g"], ["ans"]))      # u8 [1,1,10,10]

    # 6. pad to 30x30 with sentinel 255 -> outside is all-false one-hot (all-zero)
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Pad", ["ans", "pads", "sent"], ["ans30"]))  # u8 [1,1,30,30]
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["ans30", "colors"], ["output"]))   # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t040", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1c_task040.onnx")
