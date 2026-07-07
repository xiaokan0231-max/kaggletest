"""Build compact flood-fill ONNX for task243 (alternating V/H flood).

Rule: color-0 cells 4-connected (through 0s) to a color-1 cell become 1.
Boundary IO: input/output [1,10,30,30] float32 one-hot.
Crop to SxS uint8; flood by alternating single-direction MaxPool dilations
each followed by a mask (Mul by allowed=zeros|ones); reconstruct; pad back.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

S = 18
NDIL = 39  # alternating V/H dilations (min found = 39)


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build(path, S=S, NDIL=NDIL):
    nodes, inits = [], []
    inits.append(const("s0", np.array([0, 0, 0, 0], np.int64)))
    inits.append(const("eS", np.array([1, 10, S, S], np.int64)))
    inits.append(const("ax", np.array([0, 1, 2, 3], np.int64)))
    nodes.append(helper.make_node("Slice", ["input", "s0", "eS", "ax"], ["in_crop"]))
    nodes.append(helper.make_node("Cast", ["in_crop"], ["in_u"], to=TensorProto.UINT8))

    inits.append(const("e_c0", np.array([1, 1, S, S], np.int64)))
    inits.append(const("s_c1", np.array([0, 1, 0, 0], np.int64)))
    inits.append(const("e_c1", np.array([1, 2, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["in_u", "s0", "e_c0", "ax"], ["ch0"]))
    nodes.append(helper.make_node("Slice", ["in_u", "s_c1", "e_c1", "ax"], ["ch1"]))
    # allowed = max(ch0, ch1)  (cells flood may occupy)
    nodes.append(helper.make_node("Max", ["ch0", "ch1"], ["allowed"]))

    cur = "ch1"
    for i in range(NDIL):
        if i % 2 == 0:
            ks, pd = [3, 1], [1, 0, 1, 0]
        else:
            ks, pd = [1, 3], [0, 1, 0, 1]
        nodes.append(helper.make_node("MaxPool", [cur], [f"p{i}"],
                                      kernel_shape=ks, pads=pd, strides=[1, 1]))
        nodes.append(helper.make_node("Mul", [f"p{i}", "allowed"], [f"d{i}"]))
        cur = f"d{i}"
    # flooded zeros = cur AND ch0  (drop the seed ones, keep only filled zeros)
    nodes.append(helper.make_node("Mul", [cur, "ch0"], ["flood"]))

    nodes.append(helper.make_node("Sub", ["ch0", "flood"], ["ch0o"]))
    nodes.append(helper.make_node("Max", ["ch1", "flood"], ["ch1o"]))
    inits.append(const("s_c2", np.array([0, 2, 0, 0], np.int64)))
    inits.append(const("e_c2", np.array([1, 10, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["in_u", "s_c2", "e_c2", "ax"], ["ch29"]))
    nodes.append(helper.make_node("Concat", ["ch0o", "ch1o", "ch29"], ["out_u"], axis=1))
    inits.append(const("padv", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64)))
    inits.append(const("pad0", np.uint8(0)))
    nodes.append(helper.make_node("Pad", ["out_u", "padv", "pad0"], ["out_pad"], mode="constant"))
    nodes.append(helper.make_node("Cast", ["out_pad"], ["output"], to=TensorProto.FLOAT))

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "t243", [x], [y], inits)
    m = helper.make_model(g, ir_version=9, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b4_task243.onnx"
    S_ = int(sys.argv[2]) if len(sys.argv) > 2 else S
    N_ = int(sys.argv[3]) if len(sys.argv) > 3 else NDIL
    build(out, S_, N_)
