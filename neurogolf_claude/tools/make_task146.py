"""task146: 9x3 grid = 3 stacked 3x3 tiles; two equal their own transpose
(diagonally symmetric), one does not. Output = the non-symmetric tile.

Align each tile to the top-left, transpose the whole grid (which transposes the
origin-anchored 3x3 in place), score asymmetry, output the max-asymmetry tile.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task146"):
    inits = ck.base_inits() + [
        ck.const("t3", np.float32(3.0)), ck.const("t6", np.float32(6.0)),
        ck.const("t9", np.float32(9.0)), ck.const("tax123", np.array([1, 2, 3], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    n("Neg", ["t3"], ["n3"]); n("Neg", ["t6"], ["n6"])
    n("Equal", ["rmc", "n3"], ["Qu3B"]); n("Cast", ["Qu3B"], ["Qu3"], to=1)   # shift up 3
    n("Equal", ["rmc", "n6"], ["Qu6B"]); n("Cast", ["Qu6B"], ["Qu6"], to=1)   # shift up 6
    # row band masks [1,1,30,1]
    n("Less", ["ck_iota_r", "t3"], ["m0B"]); n("Cast", ["m0B"], ["m0"], to=1)
    n("GreaterOrEqual", ["ck_iota_r", "t3"], ["ge3"]); n("Cast", ["ge3"], ["ge3f"], to=1)
    n("Less", ["ck_iota_r", "t6"], ["lt6"]); n("Cast", ["lt6"], ["lt6f"], to=1)
    n("Mul", ["ge3f", "lt6f"], ["m1"])
    n("GreaterOrEqual", ["ck_iota_r", "t6"], ["ge6"]); n("Cast", ["ge6"], ["ge6f"], to=1)
    n("Less", ["ck_iota_r", "t9"], ["lt9"]); n("Cast", ["lt9"], ["lt9f"], to=1)
    n("Mul", ["ge6f", "lt9f"], ["m2"])
    # aligned tiles
    n("Mul", ["input", "m0"], ["A0"])
    n("Mul", ["input", "m1"], ["T1"]); n("MatMul", ["Qu3", "T1"], ["A1"])
    n("Mul", ["input", "m2"], ["T2"]); n("MatMul", ["Qu6", "T2"], ["A2"])
    # asymmetry score per tile via full transpose (swap H,W)
    for k in ("A0", "A1", "A2"):
        n("Transpose", [k], [f"{k}t"], perm=[0, 1, 3, 2])
        n("Sub", [k, f"{k}t"], [f"{k}d"]); n("Abs", [f"{k}d"], [f"{k}da"])
        n("ReduceSum", [f"{k}da", "tax123"], [f"s{k}"], keepdims=1)
    n("Max", ["sA0", "sA1", "sA2"], ["smax"])
    for k in ("A0", "A1", "A2"):
        n("Equal", [f"s{k}", "smax"], [f"w{k}B"]); n("Cast", [f"w{k}B"], [f"w{k}"], to=1)
        n("Mul", [f"w{k}", k], [f"o{k}"])
    n("Add", ["oA0", "oA1"], ["o01"]); n("Add", ["o01", "oA2"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
