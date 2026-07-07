"""Build compact ONNX for task139: fill the 3x3 bounding-box of each shape's 0-cells with 7.

Rule (verified 265/265 official + 2000 random isolated-shape grids):
  Every grid holds exactly 2 color-4 shapes, each sitting in an isolated 3x3
  bounding box (inter-shape Chebyshev gap >=3). Every 0-cell inside a shape's 3x3
  box becomes 7; the 4s stay; everything else stays 0.

No connected-component labeling is needed. Because each shape's bbox is exactly
3x3 and the shapes are isolated, the box CENTER bc is the unique cell whose 5x5
neighborhood holds the whole shape inside its inner 3x3 with an empty surrounding
ring. A single 5x5 Conv computes
        score = (sum of 4s in inner 3x3) - 8 * (count of 4s in the ring)
so score >= 4  iff  the inner 3x3 has >=4 fours AND the ring is empty  iff  bc is
a real box center. cover = dilate(center) over 3x3 (= MaxPool(score) > 3.5). Inside
a box the only non-4 cells are 0s (inputs use colors {0,4} only), so the filled 7s
are exactly cover & not-4 -- expressed implicitly by the nested Where below.

Memory tricks (interior tensors only; input & output are excluded from the metric):
 * No color grid: channel 4 of the one-hot input IS the 4-mask, sliced as a 9x9
   window straight out of the [1,10,30,30] input (input excluded) -- ~324B instead
   of a 3600B [1,1,30,30] color grid.
 * All interior logic runs on 9x9 f16/uint8/bool tensors (81-162 B each).
 * One 5x5 Conv + one MaxPool do the whole detection (no 4 directional convs).
 * The one-hot is built by an Equal at 9x9 (bool [1,10,9,9]) and a Pad whose result
   is the graph 'output' (excluded), so the only [1,10,*] tensor is the output.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 9


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def f16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def build(path):
    nodes, inits = [], []

    # ---- 1) 4-mask = input channel 4, cropped to the 9x9 grid; cast to f16 ----
    inits += [ci("s4", [4, 0, 0]), ci("e4", [5, S, S]), ci("ax", [1, 2, 3])]
    nodes.append(helper.make_node("Slice", ["input", "s4", "e4", "ax"], ["F32"]))  # f32 [1,1,9,9]
    nodes.append(helper.make_node("Cast", ["F32"], ["F"], to=F16))                  # f16 [1,1,9,9]

    # ---- 2) box-center score via a single 5x5 conv: inner 3x3 +1, ring -8 ----
    K = np.ones((5, 5), np.float16) * (-8.0)
    K[1:4, 1:4] = 1.0
    inits.append(numpy_helper.from_array(K.reshape(1, 1, 5, 5), "Wk"))
    nodes.append(helper.make_node("Conv", ["F", "Wk"], ["score"], pads=[2, 2, 2, 2]))  # f16 [1,1,9,9]

    # ---- 3) cover = dilate(center) = MaxPool(score) > 3.5 ----
    nodes.append(helper.make_node("MaxPool", ["score"], ["covmax"],
                                  kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))  # f16
    inits.append(f16("t35", 3.5))
    nodes.append(helper.make_node("Greater", ["covmax", "t35"], ["coverb"]))  # bool [1,1,9,9]

    # ---- 4) answer-color uint8 grid: 4 where 4-mask, else 7 inside a box, else 0 ----
    nodes.append(helper.make_node("Cast", ["F"], ["Fb"], to=BOOL))            # 4-mask bool
    inits.append(numpy_helper.from_array(np.asarray(0, np.uint8), "c0u"))
    inits.append(numpy_helper.from_array(np.asarray(4, np.uint8), "c4u"))
    inits.append(numpy_helper.from_array(np.asarray(7, np.uint8), "c7u"))
    nodes.append(helper.make_node("Where", ["coverb", "c7u", "c0u"], ["gB"]))  # u8: 7 in box else 0
    nodes.append(helper.make_node("Where", ["Fb", "c4u", "gB"], ["gm"]))       # u8: 4 where four

    # ---- 5) one-hot at 9x9 (Equal -> bool [1,10,9,9]), then Pad-as-output to 30x30 ----
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["oh9"]))  # bool [1,10,9,9]
    inits.append(ci("padsB", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "falseB"))
    nodes.append(helper.make_node("Pad", ["oh9", "padsB", "falseB"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t139", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1f_task139.onnx")
