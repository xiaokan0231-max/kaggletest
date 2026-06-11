"""task178: striped grid -> run-length-collapsed line of stripe colors.

The grid is striped along one axis (every row uniform => row-striped, or every
column uniform => col-striped). Output = the sequence of run colors (collapsing
consecutive equal stripes) as a single column (row-striped) or row (col-striped).

Detect stripe direction by shift-compare, mark run-boundary rows/cols, then use
the crop compaction backend to gather them.
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
        ck.const("t_one", np.float32(1.0)),
        ck.const("t_zero", np.float32(0.0)),
        ck.const("t_ax13_r", np.array([1, 3], np.int64)),   # reduce channels+cols -> per row
        ck.const("t_ax12_c", np.array([1, 2], np.int64)),   # reduce channels+rows -> per col
        ck.const("t_ax123", np.array([1, 2, 3], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    # shift matrices
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])                 # b-a
    n("Equal", ["rmc", "t_one"], ["Q1B"]); n("Cast", ["Q1B"], ["Q1"], to=1)   # diff==1
    n("Neg", ["t_one"], ["neg1"])
    n("Equal", ["rmc", "neg1"], ["Qm1B"]); n("Cast", ["Qm1B"], ["Qm1"], to=1)  # diff==-1
    n("Sub", ["Wdim", "t_one"], ["Wm1"]); n("Sub", ["Hdim", "t_one"], ["Hm1"])
    # row markers
    n("Equal", ["ck_iota_r", "t_zero"], ["r0B"]); n("Cast", ["r0B"], ["row0"], to=1)
    n("Equal", ["ck_iota_c", "t_zero"], ["c0B"]); n("Cast", ["c0B"], ["col0"], to=1)
    n("Mul", ["row0", "row_present"], ["row0m"])
    n("Mul", ["col0", "col_present"], ["col0m"])

    # --- is_rowstriped: every row uniform <=> input == (cols shifted left by1) on cols<W-1 ---
    n("MatMul", ["input", "Q1"], ["sc"])                          # sc[r,a]=input[r,a+1]
    n("Sub", ["input", "sc"], ["dc0"]); n("Abs", ["dc0"], ["dca"])
    n("Less", ["ck_iota_c", "Wm1"], ["cmB"]); n("Cast", ["cmB"], ["cm"], to=1)
    n("Mul", ["dca", "cm"], ["dcm"])
    n("ReduceSum", ["dcm", "t_ax123"], ["dctot"], keepdims=1)
    n("Sign", ["dctot"], ["dcs"]); n("Sub", ["t_one", "dcs"], ["is_rs"])  # 1 if row-striped

    # --- row run-boundaries: row r differs from row r-1 (rows-down shift = Q1 @ input) ---
    n("MatMul", ["Q1", "input"], ["rd"])                          # rd[r]=input[r-1]
    n("Sub", ["input", "rd"], ["drr0"]); n("Abs", ["drr0"], ["drra"])
    n("ReduceSum", ["drra", "t_ax13_r"], ["drrow"], keepdims=1)   # [1,1,30,1]
    n("Sign", ["drrow"], ["chrow"])
    n("Max", ["row0", "chrow"], ["brow0"]); n("Mul", ["brow0", "row_present"], ["brow"])
    # --- col run-boundaries: col c differs from col c-1 (cols-right shift = input @ Qm1) ---
    n("MatMul", ["input", "Qm1"], ["cr"])                         # cr[:,c]=input[:,c-1]
    n("Sub", ["input", "cr"], ["dcc0"]); n("Abs", ["dcc0"], ["dcca"])
    n("ReduceSum", ["dcca", "t_ax12_c"], ["dccol"], keepdims=1)   # [1,1,1,30]
    n("Sign", ["dccol"], ["chcol"])
    n("Max", ["col0", "chcol"], ["bcol0"]); n("Mul", ["bcol0", "col_present"], ["bcol"])

    # blend by stripe direction
    n("Sub", ["t_one", "is_rs"], ["is_cs"])
    n("Mul", ["is_rs", "brow"], ["kr_a"]); n("Mul", ["is_cs", "row0m"], ["kr_b"])
    n("Add", ["kr_a", "kr_b"], ["keeprow"])
    n("Mul", ["is_rs", "col0m"], ["kc_a"]); n("Mul", ["is_cs", "bcol"], ["kc_b"])
    n("Add", ["kc_a", "kc_b"], ["keepcol"])
    ck.emit_compaction(n, "keeprow", "keepcol", "input", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "task178")
