"""Build compact ONNX for task293 (pure uint8/bool interior, no fp16 arithmetic).

Rule (verified 267/267): grid has one full-width horizontal stripe (line color L)
and one full-height vertical stripe (bar color B). At every intersection cell
swap the visible color to the OTHER stripe's color. Background = 0.

Padding-invariant detection on the 30x30 padded grid:
  rowmax/colmax = ReduceMax of color grid over W/H -> real rows/cols are those >0.
  real = rowmask AND colmask (bounding box, == full real grid since stripes touch
  every real row/col).
  g_real = where(real, g, 99) so padding never reads as background.
  line_rows = (ReduceMin_W(g_real) > 0) AND rowmask   (every real col nonzero)
  bar_cols  = (ReduceMin_H(g_real) > 0) AND colmask
  inter = line_rows x bar_cols.
  B = max rowmax over non-line real rows (those rows hold only bar cells = B).
  L = max colmax over non-bar real cols.
  out = where(inter, where(g==L, B, L), g); pad with sentinel 255 outside real.
  output one-hot = Equal(gm, colors) as the graph output (excluded from memory).
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


def build(path):
    nodes, inits = [], []

    # 1. collapse one-hot input -> color grid g30 f32 [1,1,30,30] (input excluded)
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    nodes.append(helper.make_node("Cast", ["g30"], ["g"], to=U8))  # uint8 [1,1,30,30]

    # 2. row/col maxima (small) -> masks of real rows/cols
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("ReduceMax", ["g"], ["rowmax"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", ["g"], ["colmax"], axes=[2], keepdims=1))  # [1,1,1,30]
    nodes.append(helper.make_node("Greater", ["rowmax", "zero"], ["rowmask"]))  # bool [1,1,30,1]
    nodes.append(helper.make_node("Greater", ["colmax", "zero"], ["colmask"]))  # bool [1,1,1,30]

    # 3. real bounding box and g_real (padding -> 99)
    nodes.append(helper.make_node("And", ["rowmask", "colmask"], ["real"]))  # bool [1,1,30,30]
    inits.append(cu("p99", 99))
    nodes.append(helper.make_node("Where", ["real", "g", "p99"], ["g_real"]))  # u8 [1,1,30,30]

    # 4. line_rows / bar_cols via ReduceMin (every real col/row nonzero)
    nodes.append(helper.make_node("ReduceMin", ["g_real"], ["rowmin"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(helper.make_node("ReduceMin", ["g_real"], ["colmin"], axes=[2], keepdims=1))  # [1,1,1,30]
    nodes.append(helper.make_node("Greater", ["rowmin", "zero"], ["rowfull"]))  # bool [1,1,30,1]
    nodes.append(helper.make_node("Greater", ["colmin", "zero"], ["colfull"]))  # bool [1,1,1,30]
    nodes.append(helper.make_node("And", ["rowfull", "rowmask"], ["line_rows"]))  # [1,1,30,1]
    nodes.append(helper.make_node("And", ["colfull", "colmask"], ["bar_cols"]))   # [1,1,1,30]

    # 5. global line color L and bar color B from small reduced tensors
    nodes.append(helper.make_node("Not", ["line_rows"], ["not_line"]))  # [1,1,30,1]
    nodes.append(helper.make_node("Not", ["bar_cols"], ["not_bar"]))    # [1,1,1,30]
    nodes.append(helper.make_node("And", ["not_line", "rowmask"], ["barrow"]))   # [1,1,30,1]
    nodes.append(helper.make_node("And", ["not_bar", "colmask"], ["linecol"]))   # [1,1,1,30]
    nodes.append(helper.make_node("Where", ["barrow", "rowmax", "zero"], ["Bg"]))   # [1,1,30,1] u8
    nodes.append(helper.make_node("Where", ["linecol", "colmax", "zero"], ["Lg"]))  # [1,1,1,30] u8
    nodes.append(helper.make_node("ReduceMax", ["Bg"], ["B"], keepdims=1))  # [1,1,1,1] u8
    nodes.append(helper.make_node("ReduceMax", ["Lg"], ["L"], keepdims=1))  # [1,1,1,1] u8

    # 6. swap at intersections. Base is g_real, so padding cells carry sentinel 99
    #    (no color matches) -> no extra sentinel Where needed.
    nodes.append(helper.make_node("And", ["line_rows", "bar_cols"], ["inter"]))  # bool [1,1,30,30]
    nodes.append(helper.make_node("Equal", ["g", "L"], ["is_L"]))   # bool [1,1,30,30]
    nodes.append(helper.make_node("Where", ["is_L", "B", "L"], ["swapped"]))     # u8 [1,1,30,30]
    nodes.append(helper.make_node("Where", ["inter", "swapped", "g_real"], ["gm"]))  # u8

    # 7. one-hot output (graph output, excluded from memory)
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t293", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task293.onnx")
