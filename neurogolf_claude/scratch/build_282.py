"""Build compact ONNX for task282.

Rule (verified 265/265): every seed cell (color 5) is stamped with a hollow 3x3
ring: the 4 diagonal neighbors become 5, the 4 orthogonal neighbors become 1,
the seed center becomes 0. Seeds are always >=3 apart (Chebyshev) so stamps
never overlap. All grids are 9x9 at the top-left of the 30x30 canvas.

Cheap pipeline (no 30x30 f32 grid, no 30x30 intermediate at all):
 - Slice the input one-hot to channel 5, region 9x9  -> seed [1,1,9,9] f32
   (input is excluded; this directly reads the seed-color plane).
 - cornerN = Conv(seed, diag-kernel, pad1);  edgeN = Conv(seed, cross-kernel,pad1)
 - 9x9 color grid (u8): corner>0 -> 5, elif edge>0 -> 1, else 0.
 - one-hot at 9x9: Equal(gm9, colors u8 0..9) -> oh9 [1,10,9,9] bool.
 - Pad oh9 with 0 to [1,10,30,30] -> graph output (excluded). Padded cells are
   all-zero (empty) and the decoder trims them, yielding the 9x9 answer.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 9


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    # Slice input one-hot: channel 5 (seed color), spatial 0:9 x 0:9 -> seed f32 [1,1,9,9]
    inits += [ci("ss", [5, 0, 0]), ci("se", [6, S, S]), ci("sa", [1, 2, 3])]
    nodes.append(helper.make_node("Slice", ["input", "ss", "se", "sa"], ["seedf"]))  # f32 [1,1,9,9]
    nodes.append(helper.make_node("Cast", ["seedf"], ["seed"], to=F16))              # f16 [1,1,9,9]

    # diagonal-neighbor and orthogonal-neighbor counts via Conv (pad1).
    diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.float16).reshape(1, 1, 3, 3)
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float16).reshape(1, 1, 3, 3)
    inits.append(numpy_helper.from_array(diag, "Wdiag"))
    inits.append(numpy_helper.from_array(cross, "Wcross"))
    nodes.append(helper.make_node("Conv", ["seed", "Wdiag"], ["cornerN"], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node("Conv", ["seed", "Wcross"], ["edgeN"], pads=[1, 1, 1, 1]))

    inits.append(c16("half", 0.5))
    nodes.append(helper.make_node("Greater", ["cornerN", "half"], ["isCorner"]))  # bool
    nodes.append(helper.make_node("Greater", ["edgeN", "half"], ["isEdge"]))      # bool

    # 9x9 color grid (u8): corner -> 5, edge -> 1, else 0
    inits.append(cu("c5", 5))
    inits.append(cu("c1", 1))
    inits.append(cu("c0", 0))
    nodes.append(helper.make_node("Where", ["isEdge", "c1", "c0"], ["tmp1"]))    # u8 [1,1,9,9]
    nodes.append(helper.make_node("Where", ["isCorner", "c5", "tmp1"], ["gm9"]))  # u8 [1,1,9,9]

    # one-hot at 9x9: Equal(gm9, colors) -> oh9 [1,10,9,9] bool
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm9", "colors"], ["oh9"]))  # bool [1,10,9,9]

    # Pad to [1,10,30,30] with 0 (all-zero empty cells, trimmed by decoder) -> output
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "padval"))
    nodes.append(helper.make_node("Pad", ["oh9", "pads", "padval"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t282", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b17_task282.onnx")
