"""task299 v3: read is8 and is2 directly from input with combined channel+spatial Slice.

Each Slice reads input[1,10,30,30] and emits [1,1,6,6] (144B f32) directly -- no [1,10,6,6]
intermediate. Input is one-hot so channel 8 == is8 mask, channel 2 == is2 mask.

  is8f = Slice(input, axes[1,2,3] starts[8,0,0] ends[9,6,6]) -> [1,1,6,6] f32
  is2f = Slice(input, axes[1,2,3] starts[2,0,0] ends[3,6,6]) -> [1,1,6,6] f32
  cast u8
  colmask[1,1,1,6] = ReduceMax(is8u over rows)
  rowmask[1,1,6,1] = ReduceMax(is2u over cols)
  gm6 = 8*colmask + 2*rowmask - 6*(colmask*rowmask)
  pad -> [1,1,30,30] sentinel 255
  output = Equal(gm30, colors[1,10,1,1])
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
S = 6


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # direct combined channel+spatial slices: input -> [1,1,6,6] each
    inits += [ci("ax123", [1, 2, 3])]
    inits += [ci("s8", [8, 0, 0]), ci("e8", [9, S, S])]
    inits += [ci("s2", [2, 0, 0]), ci("e2", [3, S, S])]
    nodes.append(helper.make_node("Slice", ["input", "s8", "e8", "ax123"], ["is8f"]))
    nodes.append(helper.make_node("Slice", ["input", "s2", "e2", "ax123"], ["is2f"]))
    nodes.append(helper.make_node("Cast", ["is8f"], ["is8u"], to=U8))
    nodes.append(helper.make_node("Cast", ["is2f"], ["is2u"], to=U8))

    inits.append(ci("ax2", [2]))
    inits.append(ci("ax3", [3]))
    nodes.append(helper.make_node("ReduceMax", ["is8u", "ax2"], ["colmask"], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["is2u", "ax3"], ["rowmask"], keepdims=1))

    nodes.append(helper.make_node("Mul", ["colmask", "rowmask"], ["both"]))
    inits.append(cu("k8", 8))
    inits.append(cu("k2", 2))
    inits.append(cu("k6", 6))
    nodes.append(helper.make_node("Mul", ["colmask", "k8"], ["a"]))
    nodes.append(helper.make_node("Mul", ["rowmask", "k2"], ["b"]))
    nodes.append(helper.make_node("Add", ["a", "b"], ["ab"]))
    nodes.append(helper.make_node("Mul", ["both", "k6"], ["subv"]))
    nodes.append(helper.make_node("Sub", ["ab", "subv"], ["gm6"]))

    inits.append(ci("padv", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(cu("padc", 255))
    nodes.append(helper.make_node("Pad", ["gm6", "padv", "padc"], ["gm"], mode="constant"))

    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

    inp = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task299", [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1f_task299.onnx")
