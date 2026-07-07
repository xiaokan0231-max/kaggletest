"""Build compact ONNX for task222: keep the unique solid-color rectangle, zero rest.

Rule (verified 266/266): a cell belongs to the output iff it is part of a solid
nonzero 2x2 block whose 2x2-block (top-left) has a 4-neighbor block. Kept cells
keep their input color; every other in-grid cell becomes background (0).

All grids are 16x16. We compute the color grid g from the full input (excluded),
crop to 16x16, run the block pipeline in float16, reconstruct the one-hot from g
(cheap bool tensors), select with Where, and pad back to 30x30.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
S = 16


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cb(name, v):
    return numpy_helper.from_array(np.asarray(v, np.bool_), name=name)


def build(path, out_dtype=BOOL):
    nodes, inits = [], []

    # color grid from full f32 input (input excluded) -> g30 [1,1,30,30] f32
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))  # f32 [1,1,30,30]
    # crop to 16x16 and cast to f16
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g16f"]))  # f32 [1,1,16,16]
    nodes.append(helper.make_node("Cast", ["g16f"], ["g"], to=F16))               # f16 [1,1,16,16]

    # 2x2 solid-nonzero detection without a min pool:
    #   gmax = window max ; gq = -windowmean = -sum/4
    #   r = gmax + gq = max - mean >= 0, == 0 iff all four cells equal.
    nodes.append(helper.make_node("MaxPool", ["g"], ["gmax"], kernel_shape=[2, 2], strides=[1, 1]))
    inits.append(numpy_helper.from_array(np.full((1, 1, 2, 2), -0.25, np.float16), "Wmean"))
    nodes.append(helper.make_node("Conv", ["g", "Wmean"], ["gq"]))        # -sum/4 [1,1,15,15]
    nodes.append(helper.make_node("Add", ["gmax", "gq"], ["r"]))          # max - mean

    inits.append(c16("half", 0.5))
    inits.append(c16("q25", 0.25))
    nodes.append(helper.make_node("Greater", ["gmax", "half"], ["nz"]))   # max>=1
    nodes.append(helper.make_node("Less", ["r", "q25"], ["eq"]))          # all equal
    nodes.append(helper.make_node("And", ["nz", "eq"], ["blockb"]))
    nodes.append(helper.make_node("Cast", ["blockb"], ["block"], to=F16))  # [1,1,15,15]

    # neighbor count (plus conv, pad1) -> [1,1,15,15]
    plus = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float16).reshape(1, 1, 3, 3)
    inits.append(numpy_helper.from_array(plus, "Wplus"))
    nodes.append(helper.make_node("Conv", ["block", "Wplus"], ["nbsum"], pads=[1, 1, 1, 1]))
    # bn>0 iff this is a block with >=1 neighbor block; combine in fewer ops
    nodes.append(helper.make_node("Mul", ["block", "nbsum"], ["bn"]))        # f16 [1,1,15,15]
    nodes.append(helper.make_node("Greater", ["bn", "half"], ["keepb"]))     # bool
    nodes.append(helper.make_node("Cast", ["keepb"], ["keepblk"], to=F16))   # f16 [1,1,15,15]

    # dilate to 2x2 footprint -> [1,1,16,16] (values 0/1)
    nodes.append(helper.make_node("MaxPool", ["keepblk"], ["cell"],
                                  kernel_shape=[2, 2], strides=[1, 1], pads=[1, 1, 1, 1]))  # f16

    # masked color grid: keep color only inside the rectangle, else 0
    nodes.append(helper.make_node("Mul", ["g", "cell"], ["gm"]))             # f16 [1,1,16,16]
    # cast to uint8 (colors are 0..9, exact) to halve the padded-grid memory
    nodes.append(helper.make_node("Cast", ["gm"], ["gmu"], to=TensorProto.UINT8))  # u8 [1,1,16,16]
    # pad to 30x30 with sentinel 255 (no color matches) -> out-of-crop all-false.
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(255, np.uint8), "sent"))
    nodes.append(helper.make_node("Pad", ["gmu", "pads", "sent"], ["gm30"]))  # u8 [1,1,30,30]
    # full output one-hot directly: output[c] = (gm30 == c). Equal IS the graph
    # output (excluded) -> no [1,10,*] intermediate.
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))  # bool [1,10,30,30]
    out_t = BOOL

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", out_t, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t222", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes), "out_dtype", out_t)


if __name__ == "__main__":
    import sys
    od = BOOL
    if len(sys.argv) > 2 and sys.argv[2] == "f16":
        od = F16
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b5_task222.onnx", od)
