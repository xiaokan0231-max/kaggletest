"""task296: fixed 5x7 -> 3x3 projection. Input rows grouped by rmap=(0,1,1,1,2)
and cols by cmap=(0,1,1,1,1,1,2); non-background cells overlay into the 3x3.

Project only the foreground channels (zero channel 0 first) so background cells
don't pollute the sum, then fill channel 0 wherever the 3x3 cell has no foreground.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
RMAP = (0, 1, 1, 1, 2)
CMAP = (0, 1, 1, 1, 1, 1, 2)


def build(tid="task296"):
    Prow = np.zeros((1, 1, 30, 30), np.float32)
    for r, a in enumerate(RMAP):
        Prow[0, 0, a, r] = 1.0
    PcolT = np.zeros((1, 1, 30, 30), np.float32)
    for c, b in enumerate(CMAP):
        PcolT[0, 0, c, b] = 1.0
    inits = ck.base_inits() + [
        ck.const("p_row", Prow), ck.const("p_colT", PcolT),
        ck.const("p_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("p_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("p_one", np.float32(1.0)), ck.const("p_three", np.float32(3.0)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Mul", ["input", "p_nzch"], ["fg"])                  # zero background channel
    n("MatMul", ["p_row", "fg"], ["rows"])
    n("MatMul", ["rows", "p_colT"], ["proj"])              # [1,10,30,30] foreground projection
    n("ReduceMax", ["proj"], ["fgpres"], axes=[1], keepdims=1)   # [1,1,30,30]
    # 3x3 region mask
    n("Less", ["ck_iota_r", "p_three"], ["r3B"]); n("Cast", ["r3B"], ["r3"], to=1)
    n("Less", ["ck_iota_c", "p_three"], ["c3B"]); n("Cast", ["c3B"], ["c3"], to=1)
    n("Mul", ["r3", "c3"], ["reg"])
    n("Sub", ["p_one", "fgpres"], ["nofg"]); n("Mul", ["reg", "nofg"], ["bgmask"])
    n("Mul", ["bgmask", "p_e0"], ["bg"])
    n("Add", ["proj", "bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
