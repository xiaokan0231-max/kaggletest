"""task115: banded grid (with sparse noise) -> RLE-collapsed majority colors.

The grid is split into solid-color bands along one axis (plus sparse noise).
Output = the run-length-collapsed per-line majority colors of the banded axis,
where the banded axis is chosen as the one with higher per-line purity (fraction
of cells equal to that line's majority color):
  - vertical bands (columns purer) -> a single ROW vector of column majorities,
  - horizontal bands (rows purer)  -> a single COLUMN vector of row majorities.

ONNX expression (no banned ops):
  per-line counts = ReduceSum(one-hot input) over the cross axis.
  per-line majority one-hot = Equal(score, ReduceMax(score)) with
    score = count - eps*channel_index  (tie-break to lowest color index, like
    np.bincount(...).argmax()).
  purity sums cp/rp = ReduceSum(ReduceMax(count, ch)); pick_col = cp>rp.
  build a majority-color content grid (broadcast the per-line majority across the
  cross axis), mark run-boundary lines (majority differs from previous line, or
  it's the first present line), then gather them to the top-left via the
  cropkit compaction backend.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task115"):
    inits = ck.base_inits() + [
        ck.const("t_one", np.float32(1.0)),
        ck.const("t_zero", np.float32(0.0)),
        ck.const("t_eps", np.float32(1e-3)),
        # channel index [1,10,1,1] for tie-break to lowest color
        ck.const("t_chidx", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        ck.const("t_ax1", np.array([1], np.int64)),      # reduce channels
        ck.const("t_ax2", np.array([2], np.int64)),      # reduce rows
        ck.const("t_ax3", np.array([3], np.int64)),      # reduce cols
        ck.const("t_ax23", np.array([2, 3], np.int64)),
        ck.const("ck_ax_row", np.array(2, np.int64)),    # CumSum axis 2 (rows), scalar
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)  # presence, row_present[1,1,30,1], col_present[1,1,1,30], Hdim, Wdim, nz_*

    # ---- per-line color counts ----
    n("ReduceSum", ["input", "t_ax2"], ["colcnt"], keepdims=1)   # [1,10,1,30] count per (color,col)
    n("ReduceSum", ["input", "t_ax3"], ["rowcnt"], keepdims=1)   # [1,10,30,1] count per (color,row)

    # ---- purity sums and axis pick ----
    n("ReduceMax", ["colcnt"], ["colmaxcnt"], axes=[1], keepdims=1)   # [1,1,1,30]
    n("ReduceMax", ["rowcnt"], ["rowmaxcnt"], axes=[1], keepdims=1)   # [1,1,30,1]
    n("ReduceSum", ["colmaxcnt", "t_ax23"], ["cp"], keepdims=1)       # [1,1,1,1]
    n("ReduceSum", ["rowmaxcnt", "t_ax23"], ["rp"], keepdims=1)       # [1,1,1,1]
    # pick_col = 1.0 if cp>rp else 0.0
    n("Greater", ["cp", "rp"], ["pick_colB"])
    n("Cast", ["pick_colB"], ["pick_col"], to=1)                      # FLOAT [1,1,1,1]
    n("Sub", ["t_one", "pick_col"], ["pick_row"])

    # ---- tie-broken majority one-hot per line (lowest color index wins ties) ----
    n("Mul", ["t_chidx", "t_eps"], ["chpen"])                        # eps*idx
    # columns
    n("Sub", ["colcnt", "chpen"], ["col_score"])                     # [1,10,1,30]
    n("ReduceMax", ["col_score"], ["col_smax"], axes=[1], keepdims=1)  # [1,1,1,30]
    n("Equal", ["col_score", "col_smax"], ["col_majB"])
    n("Cast", ["col_majB"], ["col_maj"], to=1)                       # one-hot [1,10,1,30]
    # rows
    n("Sub", ["rowcnt", "chpen"], ["row_score"])                     # [1,10,30,1]
    n("ReduceMax", ["row_score"], ["row_smax"], axes=[1], keepdims=1)  # [1,1,30,1]
    n("Equal", ["row_score", "row_smax"], ["row_majB"])
    n("Cast", ["row_majB"], ["row_maj"], to=1)                       # one-hot [1,10,30,1]

    # ---- structured shift matrices (col-right / row-down by 1) ----
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])                    # r - c
    n("Neg", ["t_one"], ["neg1"])
    n("Equal", ["rmc", "neg1"], ["QcrB"]); n("Cast", ["QcrB"], ["Qcr"], to=1)  # cols-right: input@Qcr -> col c <- c-1
    n("Equal", ["rmc", "t_one"], ["QrdB"]); n("Cast", ["QrdB"], ["Qrd"], to=1)  # rows-down: Qrd@input -> row r <- r-1

    # ---- col run boundaries: col c majority differs from col c-1, or c is first present col ----
    n("MatMul", ["col_maj", "Qcr"], ["col_maj_prev"])               # col c holds col (c-1) majority
    n("Sub", ["col_maj", "col_maj_prev"], ["col_d0"]); n("Abs", ["col_d0"], ["col_da"])
    n("ReduceSum", ["col_da", "t_ax1"], ["col_diff"], keepdims=1)   # [1,1,1,30] >0 if changed
    n("Sign", ["col_diff"], ["col_chg"])                            # 1 if changed from prev col
    # first present column: cumulative present count == 1
    n("CumSum", ["col_present", "ck_ax3"], ["col_pcs"])             # [1,1,1,30]
    n("Equal", ["col_pcs", "t_one"], ["col_firstB"]); n("Cast", ["col_firstB"], ["col_first"], to=1)
    n("Max", ["col_chg", "col_first"], ["col_b0"])
    n("Mul", ["col_b0", "col_present"], ["col_bound"])              # [1,1,1,30] boundary cols within W

    # ---- row run boundaries: symmetric ----
    n("MatMul", ["Qrd", "row_maj"], ["row_maj_prev"])               # row r holds row (r-1) majority
    n("Sub", ["row_maj", "row_maj_prev"], ["row_d0"]); n("Abs", ["row_d0"], ["row_da"])
    n("ReduceSum", ["row_da", "t_ax1"], ["row_diff"], keepdims=1)   # [1,1,30,1]
    n("Sign", ["row_diff"], ["row_chg"])
    n("CumSum", ["row_present", "ck_ax_row"], ["row_pcs2"])         # cumulative present count along axis 2 (rows)
    n("Equal", ["row_pcs2", "t_one"], ["row_firstB"]); n("Cast", ["row_firstB"], ["row_first"], to=1)
    n("Max", ["row_chg", "row_first"], ["row_b0"])
    n("Mul", ["row_b0", "row_present"], ["row_bound"])              # [1,1,30,1]

    # ---- assemble keep masks blended by pick_col ----
    # row-0 / col-0 selectors
    n("Equal", ["ck_iota_r", "t_zero"], ["r0B"]); n("Cast", ["r0B"], ["row0"], to=1)  # [1,1,30,1]
    n("Equal", ["ck_iota_c", "t_zero"], ["c0B"]); n("Cast", ["c0B"], ["col0"], to=1)  # [1,1,1,30]

    # col-banded case: keep_row = row0, keep_col = col_bound
    # row-banded case: keep_row = row_bound, keep_col = col0
    n("Mul", ["pick_col", "row0"], ["kr_col"])         # [1,1,30,1]
    n("Mul", ["pick_row", "row_bound"], ["kr_row"])    # [1,1,30,1]
    n("Add", ["kr_col", "kr_row"], ["keeprow"])
    n("Mul", ["pick_col", "col_bound"], ["kc_col"])    # [1,1,1,30]
    n("Mul", ["pick_row", "col0"], ["kc_row"])         # [1,1,1,30]
    n("Add", ["kc_col", "kc_row"], ["keepcol"])

    # ---- content grid blended by pick_col ----
    # col content [1,10,30,30] = col_maj broadcast down rows
    # row content [1,10,30,30] = row_maj broadcast across cols
    # build a [1,1,30,30] ones field implicitly via broadcasting add to zero grid.
    n("Mul", ["col_maj", "pick_col"], ["col_maj_pc"])  # [1,10,1,30]
    n("Mul", ["row_maj", "pick_row"], ["row_maj_pr"])  # [1,10,30,1]
    # broadcast both into [1,10,30,30] and add
    n("Add", ["col_maj_pc", "row_maj_pr"], ["content"])  # broadcasting -> [1,10,30,30]

    ck.emit_compaction(n, "keeprow", "keepcol", "content", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
