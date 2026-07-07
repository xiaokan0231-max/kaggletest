"""Build compact ONNX for task299.

Rule (verified 21/21 visible): input has two 8s (vertical, top, in column c8)
and two 2s (horizontal, in row r2). Output:
  - column c8 filled with 8 in every row,
  - row r2 filled with 2 in every column,
  - intersection (r2, c8) = 4,
  - everything else 0.

Pipeline (all interior tensors on a single 6x6 color grid, uint8/bool):
  g = Conv(input, [0..9]) f32 -> crop 6x6 -> u8
  is8 = (g==8); is2 = (g==2)
  colmask[1,1,1,6] = ReduceMax(is8 over rows)   -> which columns are c8
  rowmask[1,1,6,1] = ReduceMax(is2 over cols)   -> which rows are r2
  Build output color grid gm[1,1,6,6] (u8):
     gm = 8*incol + 2*inrow - 6*(incol&inrow)
        incol broadcast colmask over rows, inrow broadcast rowmask over cols
        at intersection: 8+2-6 = 4. col only: 8. row only: 2. else 0.
  Output one-hot = Equal(gm, colors[1,10,1,1] u8 0..9)  -> bool [1,10,6,6] (the graph output, excluded)
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
S = 6


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # color grid from full f32 input (input excluded) -> g30 [1,1,30,30] f32
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    # crop to 6x6, cast to u8
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g6f"]))
    nodes.append(helper.make_node("Cast", ["g6f"], ["g"], to=U8))  # u8 [1,1,6,6]

    # masks
    inits.append(cu("c8", 8))
    inits.append(cu("c2", 2))
    nodes.append(helper.make_node("Equal", ["g", "c8"], ["is8"]))  # bool [1,1,6,6]
    nodes.append(helper.make_node("Equal", ["g", "c2"], ["is2"]))  # bool

    nodes.append(helper.make_node("Cast", ["is8"], ["is8u"], to=U8))
    nodes.append(helper.make_node("Cast", ["is2"], ["is2u"], to=U8))

    # colmask [1,1,1,6] : columns containing an 8  (reduce over rows axis=2)
    inits.append(ci("ax2", [2]))
    inits.append(ci("ax3", [3]))
    nodes.append(helper.make_node("ReduceMax", ["is8u", "ax2"], ["colmask"], keepdims=1))
    # rowmask [1,1,6,1] : rows containing a 2 (reduce over cols axis=3)
    nodes.append(helper.make_node("ReduceMax", ["is2u", "ax3"], ["rowmask"], keepdims=1))

    # incol = colmask broadcast to [1,1,6,6] (u8 0/1); inrow = rowmask broadcast
    # both = incol*inrow (intersection)
    nodes.append(helper.make_node("Mul", ["colmask", "rowmask"], ["both"]))  # [1,1,6,6] u8 0/1

    # gm = 8*incol + 2*inrow - 6*both  (compute in u8 via weighted adds, no negatives intermediate)
    # do: a = 8*colmask ; b = 2*rowmask ; ab = a+b (broadcast -> [1,1,6,6]); sub 6*both
    inits.append(cu("k8", 8))
    inits.append(cu("k2", 2))
    inits.append(cu("k6", 6))
    nodes.append(helper.make_node("Mul", ["colmask", "k8"], ["a"]))   # [1,1,1,6]
    nodes.append(helper.make_node("Mul", ["rowmask", "k2"], ["b"]))   # [1,1,6,1]
    nodes.append(helper.make_node("Add", ["a", "b"], ["ab"]))         # [1,1,6,6] u8: 0/2/8/10
    nodes.append(helper.make_node("Mul", ["both", "k6"], ["sub"]))    # [1,1,6,6] u8: 0 or 6
    nodes.append(helper.make_node("Sub", ["ab", "sub"], ["gm6"]))     # 0/2/8/4

    # pad gm6 [1,1,6,6] -> [1,1,30,30] with sentinel 255 (matches no color 0..9)
    # so out-of-grid cells are all-false across the 10 channels (matches all-zero benchmark).
    inits.append(ci("padv", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))  # pad after on H,W
    inits.append(cu("padc", 255))
    nodes.append(helper.make_node("Pad", ["gm6", "padv", "padc"], ["gm"], mode="constant"))

    # one-hot output = Equal(gm, colors[1,10,1,1]) -> bool [1,10,30,30] = graph output (excluded)
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

    inp = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task299", [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("saved", path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1f_task299.onnx")
