"""task243 flood v5: GENUINE plus-step BFS (28 steps), robust as deployed.

Each step: cur_next = Min( Add(MaxPoolV(cur), MaxPoolH(cur)), allowed )  -- masked plus-dilation.
uint8 throughout; optimized IO (cast-first).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

S = 18
STEPS = 28


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build(path, S=S, STEPS=STEPS):
    nodes, inits = [], []
    inits.append(const("ax", np.array([0, 1, 2, 3], np.int64)))
    nodes.append(helper.make_node("Cast", ["input"], ["inu"], to=TensorProto.UINT8))
    inits.append(const("sb0", np.array([0, 0, 0, 0], np.int64)))
    inits.append(const("e_0", np.array([1, 1, S, S], np.int64)))
    inits.append(const("s_1", np.array([0, 1, 0, 0], np.int64)))
    inits.append(const("e_1", np.array([1, 2, S, S], np.int64)))
    inits.append(const("s29", np.array([0, 2, 0, 0], np.int64)))
    inits.append(const("e29", np.array([1, 10, S, S], np.int64)))
    nodes.append(helper.make_node("Slice", ["inu", "sb0", "e_0", "ax"], ["ch0"]))
    nodes.append(helper.make_node("Slice", ["inu", "s_1", "e_1", "ax"], ["ch1"]))
    nodes.append(helper.make_node("Slice", ["inu", "s29", "e29", "ax"], ["ch29"]))
    nodes.append(helper.make_node("Max", ["ch0", "ch1"], ["allowed"]))

    cur = "ch1"
    for i in range(STEPS):
        nodes.append(helper.make_node("MaxPool", [cur], [f"v{i}"],
                                      kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1]))
        nodes.append(helper.make_node("MaxPool", [cur], [f"h{i}"],
                                      kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1]))
        nodes.append(helper.make_node("Add", [f"v{i}", f"h{i}"], [f"s{i}"]))
        nodes.append(helper.make_node("Min", [f"s{i}", "allowed"], [f"d{i}"]))
        cur = f"d{i}"
    nodes.append(helper.make_node("Mul", ["ch0", cur], ["fl0"]))
    nodes.append(helper.make_node("Sub", ["ch0", "fl0"], ["ch0o"]))
    nodes.append(helper.make_node("Concat", ["ch0o", cur, "ch29"], ["out_u"], axis=1))
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
    N_ = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    build(out, S_, N_)
