"""Dedup a 2x-duplicated grid by detecting which axis is duplicated.

The grid is its output tile repeated twice along exactly one axis. We detect
horizontal duplication (left half == right half) via a MatMul column-shift and
compare; if horizontal -> output = left half, else -> output = top half.
Usage: python make_halvelong.py task188
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid):
    inits = ck.base_inits() + [
        ck.const("hl_half", np.float32(0.5)),
        ck.const("hl_one", np.float32(1.0)),
        ck.const("hl_ax123", np.array([1, 2, 3], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    n("Mul", ["Wdim", "hl_half"], ["hWf0"]); n("Floor", ["hWf0"], ["Whalf"])   # [1,1,1,1]
    n("Mul", ["Hdim", "hl_half"], ["hHf0"]); n("Floor", ["hHf0"], ["Hhalf"])
    # detect horizontal duplication: shift input left by Whalf, compare on left cols
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rc"])               # [1,1,30,30]  b-a
    n("Equal", ["rc", "Whalf"], ["QhB"]); n("Cast", ["QhB"], ["Qh"], to=1)
    n("MatMul", ["input", "Qh"], ["shifted"])                  # [:,:,r,a]=input[r,a+Whalf]
    n("Sub", ["input", "shifted"], ["dh"]); n("Abs", ["dh"], ["adh"])
    n("Less", ["ck_iota_c", "Whalf"], ["lcB"]); n("Cast", ["lcB"], ["lc"], to=1)  # [1,1,1,30]
    n("Mul", ["adh", "lc"], ["mh"])
    n("ReduceSum", ["mh", "hl_ax123"], ["mism"], keepdims=1)    # [1,1,1,1]
    n("Sign", ["mism"], ["ms"]); n("Sub", ["hl_one", "ms"], ["is_hdup"])
    # candidate masks
    n("Mul", ["lc", "col_present"], ["halfcol"])               # first-half cols
    n("Less", ["ck_iota_r", "Hhalf"], ["lrB"]); n("Cast", ["lrB"], ["lr"], to=1)
    n("Mul", ["lr", "row_present"], ["halfrow"])               # first-half rows
    # keep_row = hdup?rows_full:halfrow ; keep_col = hdup?halfcol:cols_full
    n("Sub", ["hl_one", "is_hdup"], ["not_h"])
    n("Mul", ["is_hdup", "row_present"], ["kr_a"]); n("Mul", ["not_h", "halfrow"], ["kr_b"])
    n("Add", ["kr_a", "kr_b"], ["keeprow"])
    n("Mul", ["is_hdup", "halfcol"], ["kc_a"]); n("Mul", ["not_h", "col_present"], ["kc_b"])
    n("Add", ["kc_a", "kc_b"], ["keepcol"])
    ck.emit_compaction(n, "keeprow", "keepcol", "input", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build(sys.argv[1])
