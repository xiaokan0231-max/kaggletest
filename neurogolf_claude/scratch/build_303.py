"""Build compact ONNX for task303.

Rule (verified 265/265): within the real grid, any row whose in-grid cells are all
color 0 becomes color 2; any column whose in-grid cells are all color 0 becomes
color 2. All other in-grid cells keep their color. Out-of-grid cells stay blank.

Grids vary in size (no common crop). Work on the full [1,1,30,30] grid.

Tricks:
* One Conv weight [1,2,...,10] with bias -1 gives g = color in-grid, -1 out-of-grid.
  Casting g to uint8 maps in-grid colors to 0..9 and the -1 sentinel to 255 (C-style
  wraparound), so out-of-grid cells match no color channel in the final Equal.
* A row is all-zero iff ReduceMax(g over the row) == 0 (every in-grid cell is color
  0; a fully out-of-grid row maxes to -1). Same for columns.
* A triggered cell always had color 0, so overriding it to color 2 is just an
  element-wise Max against a 2-valued flag. The 255 sentinel survives Max (255 >= 2),
  so out-of-grid cells need no extra presence masking.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    # g [1,1,30,30] f32 = color (0..9) in-grid, -1 out-of-grid.
    inits.append(cf("Wcp", (np.arange(10, dtype=np.float32) + 1).reshape(1, 10, 1, 1)))
    inits.append(cf("Bm1", [-1.0]))
    nodes.append(helper.make_node("Conv", ["input", "Wcp", "Bm1"], ["g"]))

    inits.append(cf("zero_f", 0.0))

    # row all-zero iff max over row == 0 (in-grid color-0 -> 0; out-of-grid -> -1)
    nodes.append(helper.make_node("ReduceMax", ["g"], ["rowmax"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("Equal", ["rowmax", "zero_f"], ["rowall0"]))  # bool [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", ["g"], ["colmax"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Equal", ["colmax", "zero_f"], ["colall0"]))  # bool [1,1,1,30]

    # uint8 color grid: in-grid 0..9, out-of-grid 255 (sentinel).
    nodes.append(helper.make_node("Cast", ["g"], ["gpu"], to=U8))               # u8 [1,1,30,30]

    # 2-valued flags (tiny tensors). Triggered cells become color 2 via Max.
    inits.append(cu("two", 2))
    inits.append(cu("zero_u", 0))
    nodes.append(helper.make_node("Where", ["rowall0", "two", "zero_u"], ["rowflag"]))  # u8 [1,1,30,1]
    nodes.append(helper.make_node("Where", ["colall0", "two", "zero_u"], ["colflag"]))  # u8 [1,1,1,30]

    # Variadic Max overrides color-0 trigger cells to 2 in one op; 255 sentinel
    # survives (255 >= 2). rowflag/colflag broadcast against gpu.
    nodes.append(helper.make_node("Max", ["gpu", "rowflag", "colflag"], ["gm"]))  # u8 [1,1,30,30]

    # output one-hot directly: channel c matches color c. Equal IS the graph output.
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))       # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t303", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b13_task303.onnx")
