"""task019: output = 2x2 tiling of input, then diagonal-halo of 8s on background.

ONNX plan (opset 12, ir_version 10):
  1. G = ReduceSum(input, ch)            -> [1,1,30,30] grid mask
  2. row_in [1,30], col_in [30,1] via ReduceMax + Reshape; H, W via ReduceSum
  3. D1[i,r] = i - r from iota; permutation matrices
       T_row  = (Eq(D1,0) + Eq(D1,H))  * row_in   [30,30]
       T_colT = (Eq(D1,0) + Eq(D1,-W)) * col_in   [30,30]
  4. tiled = T_row @ input @ T_colT   (all 10 channels, incl. background ch0)
  5. diagcnt = Conv(tiled, w[1,10,3,3]: ch0=0, ch1..9 = diag corners), pads=1
  6. eight = diagcnt * tiled_ch0  (kills colored cells and out-of-grid spill;
     no Sign needed: counts 1..4 stay >0, and ch0 = 1-count <= 0 is legal)
  7. output = tiled + e8 * eight,  e8[1,10,1,1]: ch0=-1, ch8=+1
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def tensor(name, arr):
    arr = np.asarray(arr)
    return helper.make_tensor(
        name,
        F if arr.dtype == np.float32 else I64,
        arr.shape,
        arr.flatten().tolist(),
    )


def build():
    iota = np.arange(30, dtype=np.float32).reshape(1, 30)

    wdiag = np.zeros((1, 10, 3, 3), dtype=np.float32)
    for ch in range(1, 10):
        for r, c in ((0, 0), (0, 2), (2, 0), (2, 2)):
            wdiag[0, ch, r, c] = 1.0

    e8 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    e8[0, 0, 0, 0] = -1.0
    e8[0, 8, 0, 0] = 1.0

    inits = [
        tensor("iota", iota),
        tensor("czero", np.zeros((), dtype=np.float32)),
        tensor("shp_row", np.array([1, 30], dtype=np.int64)),
        tensor("shp_col", np.array([30, 1], dtype=np.int64)),
        tensor("wdiag", wdiag),
        tensor("e8", e8),
        tensor("sl_st", np.array([0], dtype=np.int64)),
        tensor("sl_en", np.array([1], dtype=np.int64)),
        tensor("sl_ax", np.array([1], dtype=np.int64)),
    ]

    n = []
    # index difference matrix D1[i,r] = i - r
    n.append(helper.make_node("Transpose", ["iota"], ["iota_t"], perm=[1, 0]))
    n.append(helper.make_node("Sub", ["iota_t", "iota"], ["D1"]))
    # grid mask and extents
    n.append(helper.make_node("ReduceSum", ["input"], ["G"], axes=[1], keepdims=1))
    n.append(helper.make_node("ReduceMax", ["G"], ["rowm"], axes=[3], keepdims=0))
    n.append(helper.make_node("ReduceMax", ["G"], ["colm"], axes=[2], keepdims=0))
    n.append(helper.make_node("Reshape", ["rowm", "shp_row"], ["row_in"]))
    n.append(helper.make_node("Reshape", ["colm", "shp_col"], ["col_in"]))
    n.append(helper.make_node("ReduceSum", ["row_in"], ["H"], keepdims=0))
    n.append(helper.make_node("ReduceSum", ["col_in"], ["W"], keepdims=0))
    n.append(helper.make_node("Neg", ["W"], ["negW"]))
    # equality masks (Equal-11 supports float)
    for nm, rhs in (("E0", "czero"), ("EH", "H"), ("EW", "negW")):
        n.append(helper.make_node("Equal", ["D1", rhs], [nm + "b"]))
        n.append(helper.make_node("Cast", [nm + "b"], [nm], to=F))
    # permutation matrices
    n.append(helper.make_node("Add", ["E0", "EH"], ["SR"]))
    n.append(helper.make_node("Mul", ["SR", "row_in"], ["T_row"]))
    n.append(helper.make_node("Add", ["E0", "EW"], ["SC"]))
    n.append(helper.make_node("Mul", ["SC", "col_in"], ["T_colT"]))
    # 2x2 tiling of all 10 channels
    n.append(helper.make_node("MatMul", ["T_row", "input"], ["step1"]))
    n.append(helper.make_node("MatMul", ["step1", "T_colT"], ["tiled"]))
    # diagonal halo of colored cells
    n.append(helper.make_node("Conv", ["tiled", "wdiag"], ["diagcnt"],
                              pads=[1, 1, 1, 1]))
    # restrict to in-grid background cells
    n.append(helper.make_node("Slice", ["tiled", "sl_st", "sl_en", "sl_ax"], ["ch0"]))
    n.append(helper.make_node("Mul", ["diagcnt", "ch0"], ["eight"]))
    # flip ch0 -> ch8 at those cells
    n.append(helper.make_node("Mul", ["eight", "e8"], ["delta"]))
    n.append(helper.make_node("Add", ["tiled", "delta"], ["output"]))

    graph = helper.make_graph(
        n, "task019",
        [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task019.onnx"
    onnx.save(build(), out)
    import os
    print("saved", out, os.path.getsize(out), "bytes")
