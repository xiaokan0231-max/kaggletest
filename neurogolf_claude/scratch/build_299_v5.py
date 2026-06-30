"""task299 v5: ReduceMax on f32 slices, then cast the TINY masks to u8 (not 6x6).

Keeps the full-grid genuine reduction (robust). All per-cell arithmetic on u8.
  is8f [1,1,6,6] f32 (144B), is2f [1,1,6,6] f32 (144B)
  colmaskf [1,1,1,6] f32 = ReduceMax(is8f rows) (24B) -> colmask u8 (6B)
  rowmaskf [1,1,6,1] f32 = ReduceMax(is2f cols) (24B) -> rowmask u8 (6B)
  gm6 (u8 6x6, 36B) = 8*colmask + 2*rowmask - 6*(colmask*rowmask)
  gm = Pad -> [1,1,30,30] u8 sentinel 255 (900B)
  output = Equal(gm, colors)
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

    inits += [ci("ax123", [1, 2, 3])]
    inits += [ci("s8", [8, 0, 0]), ci("e8", [9, S, S])]
    inits += [ci("s2", [2, 0, 0]), ci("e2", [3, S, S])]
    nodes.append(helper.make_node("Slice", ["input", "s8", "e8", "ax123"], ["is8f"]))
    nodes.append(helper.make_node("Slice", ["input", "s2", "e2", "ax123"], ["is2f"]))

    inits.append(ci("ax2", [2]))
    inits.append(ci("ax3", [3]))
    nodes.append(helper.make_node("ReduceMax", ["is8f", "ax2"], ["colmaskf"], keepdims=1))  # [1,1,1,6] f32
    nodes.append(helper.make_node("ReduceMax", ["is2f", "ax3"], ["rowmaskf"], keepdims=1))  # [1,1,6,1] f32
    nodes.append(helper.make_node("Cast", ["colmaskf"], ["colmask"], to=U8))  # [1,1,1,6] u8
    nodes.append(helper.make_node("Cast", ["rowmaskf"], ["rowmask"], to=U8))  # [1,1,6,1] u8

    inits.append(cu("k8", 8))
    inits.append(cu("k2", 2))
    inits.append(cu("k6", 6))
    nodes.append(helper.make_node("Mul", ["colmask", "rowmask"], ["both"]))  # [1,1,6,6] u8
    nodes.append(helper.make_node("Mul", ["colmask", "k8"], ["a"]))          # [1,1,1,6]
    nodes.append(helper.make_node("Mul", ["rowmask", "k2"], ["b"]))          # [1,1,6,1]
    nodes.append(helper.make_node("Add", ["a", "b"], ["ab"]))                # [1,1,6,6]
    nodes.append(helper.make_node("Mul", ["both", "k6"], ["subv"]))          # [1,1,6,6]
    nodes.append(helper.make_node("Sub", ["ab", "subv"], ["gm6"]))           # [1,1,6,6] u8

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
