"""Build compact ONNX for task256.

Rule (verified 266/266): input has a single horizontal bar of color 2 starting at
column 0, in row r0, length L. Let K = L + r0. The output fills every cell (i,j)
with i+j < K:  color 3 above the bar (i<r0), 2 on the bar (i==r0), 1 below (i>r0).
Everything else (incl. 30x30 padding) is 0. Because L + r0 < W <= 30 in every case
the fill never reaches a padding cell, so no width/height detection is needed.

Lean pipeline. The only big f32 [1,1,30,30] tensor is the irreducible 2-mask conv
output g2; reductions run directly on it (it is 1.0 at 2-cells, 0 elsewhere).
  g2     = Conv(input, w2)            f32 [1,1,30,30]   (irreducible)
  L      = ReduceSum(g2)              scalar f32   (count of 2s == bar length)
  rowhas2= ReduceMax(g2, axis=3)      [1,1,30,1] f32
  r0     = ReduceMax(iidx*rowhas2)    scalar f32   (bar row)
  K      = Cast(L + r0, uint8)        scalar uint8
  fill   = Less(ipj_const, K)         bool [1,1,30,30]   (ipj_const[i,j]=i+j, params)
  rowcol = 3 if i<r0, 2 if i==r0, 1 if i>r0   [1,1,30,1] uint8
  gm     = Where(fill, rowcol, 0)     uint8 [1,1,30,30]
  output = Equal(gm, colors)          bool [1,10,30,30]  (= graph output, excluded)
"""
import sys
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

    # g2 = 2-detector conv -> f32 [1,1,30,30], 1.0 where color==2 else 0
    w2 = np.zeros((1, 10, 1, 1), np.float32)
    w2[0, 2, 0, 0] = 1.0
    inits.append(cf("w2", w2))
    nodes.append(helper.make_node("Conv", ["input", "w2"], ["g2"]))

    # L = total count of 2s (bar length), scalar f32
    nodes.append(helper.make_node("ReduceSum", ["g2"], ["L"], keepdims=0))

    # rowhas2 [1,1,30,1]: 1.0 if row contains a 2
    nodes.append(helper.make_node("ReduceMax", ["g2"], ["rowhas2"], axes=[3], keepdims=1))

    # r0 = max over rows of (rowidx * rowhas2), scalar f32
    iidx = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    inits.append(cf("iidx", iidx))
    nodes.append(helper.make_node("Mul", ["iidx", "rowhas2"], ["ri"]))
    nodes.append(helper.make_node("ReduceMax", ["ri"], ["r0"], keepdims=0))

    # K = L + r0 -> uint8 scalar
    nodes.append(helper.make_node("Add", ["L", "r0"], ["Kf"]))
    nodes.append(helper.make_node("Cast", ["Kf"], ["K"], to=U8))

    # fill = (i+j) < K. ipj is a uint8 constant initializer (counts as params).
    ipj = (np.arange(30).reshape(30, 1) + np.arange(30).reshape(1, 30)).astype(np.uint8)
    inits.append(cu("ipj", ipj.reshape(1, 1, 30, 30)))
    nodes.append(helper.make_node("Less", ["ipj", "K"], ["fill"]))  # bool [1,1,30,30]

    # rowcolor[i] = 3 if i<r0, 2 if i==r0, 1 if i>r0  (compare f32 iidx vs r0)
    nodes.append(helper.make_node("Less", ["iidx", "r0"], ["above"]))   # bool [1,1,30,1]
    nodes.append(helper.make_node("Equal", ["iidx", "r0"], ["onbar"]))  # bool [1,1,30,1]
    inits.append(cu("c3", 3))
    inits.append(cu("c2", 2))
    inits.append(cu("c1", 1))
    inits.append(cu("c0", 0))
    nodes.append(helper.make_node("Where", ["onbar", "c2", "c1"], ["rc1"]))
    nodes.append(helper.make_node("Where", ["above", "c3", "rc1"], ["rowcolor"]))  # uint8 [1,1,30,1]

    # gm = Where(fill, rowcolor, 0)  uint8 [1,1,30,30] (rowcolor broadcasts over j)
    nodes.append(helper.make_node("Where", ["fill", "rowcolor", "c0"], ["gm"]))

    # output one-hot = Equal(gm, colors[1,10,1,1]) bool [1,10,30,30] (graph output, excluded)
    colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    inits.append(cu("colors", colors))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t256", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b11_task256.onnx")
