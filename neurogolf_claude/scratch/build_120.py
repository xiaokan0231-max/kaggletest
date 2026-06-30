"""Build compact ONNX for task120.

Rule (verified 266/266): a nonzero cell (r,c) of color v becomes color 8 iff its
4 orthogonal neighbors (up,down,left,right) are all == v (out-of-grid = 0).
Every other cell keeps its input color.

Pipeline on a single [1,1,30,30] color grid:
  g30 (f32, from Conv collapse) -> cast u8 -> pad ring of 0 -> slice 4 shifts ->
  interior = (up==c)&(down==c)&(left==c)&(right==c)&(c!=0) ->
  gm = Where(interior, 8, c) -> Equal(gm, colors[0..9]) = output one-hot.
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


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    # color grid from full f32 input (input excluded) -> g30 [1,1,30,30] f32
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))   # f32 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["g30"], ["c"], to=U8))        # u8 [1,1,30,30]

    # pad a ring of 0 -> [1,1,32,32], then 4 directional shifts cropped to 30x30
    inits.append(ci("pad1", [0, 0, 1, 1, 0, 0, 1, 1]))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Pad", ["c", "pad1", "zero"], ["cp"]))  # u8 [1,1,32,32]
    # up   = cp[0:30, 1:31]  (neighbor above)
    # down = cp[2:32, 1:31]
    # left = cp[1:31, 0:30]
    # right= cp[1:31, 2:32]
    inits.append(ci("ax", [2, 3]))
    inits.append(ci("up_s", [0, 1]));    inits.append(ci("up_e", [30, 31]))
    inits.append(ci("dn_s", [2, 1]));    inits.append(ci("dn_e", [32, 31]))
    inits.append(ci("lf_s", [1, 0]));    inits.append(ci("lf_e", [31, 30]))
    inits.append(ci("rt_s", [1, 2]));    inits.append(ci("rt_e", [31, 32]))
    nodes.append(helper.make_node("Slice", ["cp", "up_s", "up_e", "ax"], ["up"]))
    nodes.append(helper.make_node("Slice", ["cp", "dn_s", "dn_e", "ax"], ["dn"]))
    nodes.append(helper.make_node("Slice", ["cp", "lf_s", "lf_e", "ax"], ["lf"]))
    nodes.append(helper.make_node("Slice", ["cp", "rt_s", "rt_e", "ax"], ["rt"]))

    # equality masks (bool [1,1,30,30])
    nodes.append(helper.make_node("Equal", ["c", "up"], ["eu"]))
    nodes.append(helper.make_node("Equal", ["c", "dn"], ["ed"]))
    nodes.append(helper.make_node("Equal", ["c", "lf"], ["el"]))
    nodes.append(helper.make_node("Equal", ["c", "rt"], ["er"]))
    inits.append(cu("zg", 0))
    nodes.append(helper.make_node("Greater", ["c", "zg"], ["nz"]))   # c != 0  (c>0)

    nodes.append(helper.make_node("And", ["eu", "ed"], ["a1"]))
    nodes.append(helper.make_node("And", ["el", "er"], ["a2"]))
    nodes.append(helper.make_node("And", ["a1", "a2"], ["a3"]))
    nodes.append(helper.make_node("And", ["a3", "nz"], ["interior"]))  # bool [1,1,30,30]

    # output = Where(interior, onehot(8), input). interior is False in the padded
    # out-of-grid region -> output = input = all-zero there (matches benchmark).
    # The only [1,10,30,30] tensor is the graph output (excluded from memory).
    e8 = np.zeros((1, 10, 1, 1), np.float32); e8[0, 8, 0, 0] = 1.0
    inits.append(cf("e8", e8))
    nodes.append(helper.make_node("Where", ["interior", "e8", "input"], ["output"]))  # f32 [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t120", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b16_task120.onnx")
