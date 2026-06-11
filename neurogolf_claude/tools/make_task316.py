"""task316: scattered single nonzero cells (in distinct columns) sorted by column,
then snake-filled into a 3x3 (row0 L->R, row1 R->L, row2 L->R); leftovers stay 0.

Column-compaction yields colors in column order (= sorted order, no Sort op). A
fixed permutation reorders them so a row-major reshape lands in boustrophedon
order; background-fill the leftover cells; pad to 30x30.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
PERM = [0, 1, 2, 5, 4, 3, 6, 7, 8]   # snake order for a row-major 3x3 reshape


def build(tid="task316"):
    Pp = np.zeros((1, 1, 30, 30), np.float32)
    for m, src in enumerate(PERM):
        Pp[0, 0, src, m] = 1.0                          # newrow col m = rankrow col src
    inits = ck.base_inits() + [
        ck.const("x_wnz", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("x_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("x_zero", np.float32(0.0)), ck.const("x_one", np.float32(1.0)),
        ck.const("x_ax2", np.array([2], np.int64)),
        ck.const("x_Pp", Pp),
        ck.const("x_sstart", np.array([0, 0], np.int64)), ck.const("x_send", np.array([1, 9], np.int64)),
        ck.const("x_saxes", np.array([2, 3], np.int64)),
        ck.const("x_shp", np.array([1, 10, 3, 3], np.int64)),
        ck.const("x_pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Conv", ["input", "x_wnz"], ["nz"])                          # nonzero indicator
    n("ReduceMax", ["nz"], ["coloc"], axes=[2], keepdims=1)        # [1,1,1,30] occupied cols
    n("Mul", ["input", "x_wnz"], ["fg"])                          # zero background channel
    n("ReduceSum", ["fg", "x_ax2"], ["colcolor"], keepdims=1)      # [1,10,1,30] per-col color
    n("Equal", ["ck_iota_r", "x_zero"], ["r0B"]); n("Cast", ["r0B"], ["r0m"], to=1)  # [1,1,30,1]
    n("Mul", ["colcolor", "r0m"], ["ccg"])                        # colors placed in row 0
    ck.emit_compaction(n, "r0m", "coloc", "ccg", "rankrow")        # row0 = colors in column order
    n("MatMul", ["rankrow", "x_Pp"], ["permrow"])                 # reorder cols to snake order
    n("Slice", ["permrow", "x_sstart", "x_send", "x_saxes"], ["sl"])   # [1,10,1,9]
    n("Reshape", ["sl", "x_shp"], ["g33"])                        # [1,10,3,3] boustrophedon colors
    n("ReduceMax", ["g33"], ["pres"], axes=[1], keepdims=1)       # [1,1,3,3]
    n("Sub", ["x_one", "pres"], ["nopres"]); n("Mul", ["nopres", "x_e0"], ["bg33"])
    n("Add", ["g33", "bg33"], ["full33"])                        # fill leftovers with bg
    n("Pad", ["full33", "x_pads"], ["output"])                    # -> [1,10,30,30]
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
