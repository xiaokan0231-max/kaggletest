"""task243 flood-fill ONNX, v3: minimize 10-channel-tensor overhead.

Crop ch0+ch1 (2 ch) and ch2-9 (8 ch) separately as float, cast small uint8.
Flood (alternating V/H MaxPool + mask) at 18x18 uint8.
Reconstruct ch0o/ch1o, concat with ch29, pad to 30x30, cast to float output.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

S = 18
NDIL = 39


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build(path, S=S, NDIL=NDIL):
    nodes, inits = [], []
    inits.append(const("ax", np.array([0, 1, 2, 3], np.int64)))
    # crop ch0..1 -> [1,2,S,S] float, cast uint8, split
    inits.append(const("s01", np.array([0, 0, 0, 0], np.int64)))
    inits.append(const("e01", np.array([1, 2, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["input", "s01", "e01", "ax"], ["c01f"]))
    nodes.append(helper.make_node("Cast", ["c01f"], ["c01"], to=TensorProto.UINT8))
    inits.append(const("e_0", np.array([1, 1, S, S], np.int64)))
    inits.append(const("s_1", np.array([0, 1, 0, 0], np.int64)))
    inits.append(const("e_1", np.array([1, 2, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["c01", "s01", "e_0", "ax"], ["ch0"]))
    nodes.append(helper.make_node("Slice", ["c01", "s_1", "e_1", "ax"], ["ch1"]))
    nodes.append(helper.make_node("Max", ["ch0", "ch1"], ["allowed"]))

    # crop ch2..9 -> [1,8,S,S] uint8 (passthrough)
    inits.append(const("s29", np.array([0, 2, 0, 0], np.int64)))
    inits.append(const("e29", np.array([1, 10, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["input", "s29", "e29", "ax"], ["c29f"]))
    nodes.append(helper.make_node("Cast", ["c29f"], ["ch29"], to=TensorProto.UINT8))

    cur = "ch1"
    for i in range(NDIL):
        ks, pd = ([3, 1], [1, 0, 1, 0]) if i % 2 == 0 else ([1, 3], [0, 1, 0, 1])
        nodes.append(helper.make_node("MaxPool", [cur], [f"p{i}"],
                                      kernel_shape=ks, pads=pd, strides=[1, 1]))
        nodes.append(helper.make_node("Mul", [f"p{i}", "allowed"], [f"d{i}"]))
        cur = f"d{i}"
    nodes.append(helper.make_node("Mul", [cur, "ch0"], ["flood"]))

    nodes.append(helper.make_node("Sub", ["ch0", "flood"], ["ch0o"]))
    nodes.append(helper.make_node("Max", ["ch1", "flood"], ["ch1o"]))
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
