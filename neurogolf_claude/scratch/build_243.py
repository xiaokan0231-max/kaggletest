"""Build compact flood-fill ONNX for task243.

Rule: any color-0 cell 4-connected (through 0s) to a color-1 cell becomes color 1.
Equivalently: flood color 1 into adjacent 0-regions (4-connectivity).

Boundary IO: input/output [1,10,30,30] float32 one-hot.
Strategy: crop to SxS (S>=max grid=18), uint8 flood, reconstruct, pad back.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

S = 18          # crop size
STEPS = 28      # flood iterations (max distance observed = 28)


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build(path, S=S, STEPS=STEPS):
    nodes = []
    inits = []

    # ---- crop input [1,10,30,30] -> [1,10,S,S] uint8 ----
    inits.append(const("s0", np.array([0, 0, 0, 0], np.int64)))
    inits.append(const("eS", np.array([1, 10, S, S], np.int64)))
    inits.append(const("ax", np.array([0, 1, 2, 3], np.int64)))
    nodes.append(helper.make_node("Slice", ["input", "s0", "eS", "ax"], ["in_crop"]))
    nodes.append(helper.make_node("Cast", ["in_crop"], ["in_u"], to=TensorProto.UINT8))

    # ch0 (zeros) and ch1 (ones) masks [1,1,S,S]
    inits.append(const("s_c0", np.array([0, 0, 0, 0], np.int64)))
    inits.append(const("e_c0", np.array([1, 1, S, S], np.int64)))
    inits.append(const("s_c1", np.array([0, 1, 0, 0], np.int64)))
    inits.append(const("e_c1", np.array([1, 2, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["in_u", "s_c0", "e_c0", "ax"], ["ch0"]))
    nodes.append(helper.make_node("Slice", ["in_u", "s_c1", "e_c1", "ax"], ["ch1"]))

    # ---- flood ----
    # step0: fill0 = cross_dilate(ch1) & ch0   (zeros adjacent to a one)
    def cross(src, dst_v, dst_h, dst_c, dst_fill, mask):
        nodes.append(helper.make_node("MaxPool", [src], [dst_v],
                                      kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1]))
        nodes.append(helper.make_node("MaxPool", [src], [dst_h],
                                      kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1]))
        nodes.append(helper.make_node("Max", [dst_v, dst_h], [dst_c]))
        nodes.append(helper.make_node("Mul", [dst_c, mask], [dst_fill]))

    cross("ch1", "v_-1", "h_-1", "c_-1", "fill0", "ch0")
    prev = "fill0"
    for k in range(STEPS):
        cross(prev, f"v{k}", f"h{k}", f"c{k}", f"fill{k+1}", "ch0")
        prev = f"fill{k+1}"
    flood = prev  # flooded zeros mask [1,1,S,S] uint8

    # ---- reconstruct output channels ----
    # ch0_out = ch0 - flood ; ch1_out = max(ch1, flood)
    nodes.append(helper.make_node("Sub", ["ch0", flood], ["ch0o"]))
    nodes.append(helper.make_node("Max", ["ch1", flood], ["ch1o"]))
    # ch2..9 in one slice
    inits.append(const("s_c2", np.array([0, 2, 0, 0], np.int64)))
    inits.append(const("e_c2", np.array([1, 10, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["in_u", "s_c2", "e_c2", "ax"], ["ch29"]))
    # concat -> [1,10,S,S] uint8, pad to 30x30 (uint8), cast to float AS output
    nodes.append(helper.make_node("Concat", ["ch0o", "ch1o", "ch29"], ["out_u"], axis=1))
    inits.append(const("padv", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64)))
    inits.append(const("pad0", np.uint8(0)))
    nodes.append(helper.make_node("Pad", ["out_u", "padv", "pad0"], ["out_pad"], mode="constant"))
    nodes.append(helper.make_node("Cast", ["out_pad"], ["output"], to=TensorProto.FLOAT))

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "t243", [x], [y], inits)
    m = helper.make_model(g, ir_version=9,
                          opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b4_task243.onnx"
    S_ = int(sys.argv[2]) if len(sys.argv) > 2 else S
    ST_ = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    build(out, S_, ST_)
