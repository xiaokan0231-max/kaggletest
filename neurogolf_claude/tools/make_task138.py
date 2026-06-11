"""task138: a box bounded by 2 separator rows + 2 separator cols (each a line whose
nonzero cells fill >=55% of the grid dim). Crop to the inclusive box. The interior
holds marks of a single color that matches exactly one of the 4 border colors; each
mark projects toward that matching border as a solid bar (directional cumulative
fill). Border ring is kept unchanged.

Pipeline:
  - separator rows/cols via nonzero-count >= 0.55*dim  (exactly 2 each)
  - keep_row = (iota_r in [r0,r1]), keep_col = (iota_c in [c0,c1]); compact -> box frame
  - border colors = argmax-channel of each border-line sum (chan0 excluded)
  - mark color   = argmax-channel of interior sum (chan0 excluded)
  - direction = which border color == mark color
  - fill = directional cumulative-OR of the interior mark mask toward that border
  - output = border_ring(box) + interior(fill->mark, else background)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task138"):
    e0 = np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)
    wnz = np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)
    kill0 = np.zeros((1, 10, 1, 1), np.float32); kill0[0, 0, 0, 0] = BIG
    inits = ck.base_inits() + [
        ck.const("t_e0", e0), ck.const("t_wnz", wnz), ck.const("t_kill0", kill0),
        ck.const("t_one", np.float32(1.0)), ck.const("t_two", np.float32(2.0)),
        ck.const("t_zero", np.float32(0.0)), ck.const("t_055", np.float32(0.55)),
        ck.const("t_ax2", np.array([2], np.int64)), ck.const("t_ax3", np.array([3], np.int64)),
        ck.const("t_ax23", np.array([2, 3], np.int64)),
        ck.const("t_cax2", np.array(2, np.int64)), ck.const("t_cax3", np.array(3, np.int64)),
        ck.const("t_big", np.float32(BIG)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # --- grid dims H,W and presence ---
    n("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)        # [1,1,30,30]
    n("ReduceMax", ["presence"], ["row_present"], axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["presence"], ["col_present"], axes=[2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["row_present", "t_ax2"], ["Hdim"], keepdims=1)       # [1,1,1,1]
    n("ReduceSum", ["col_present", "t_ax3"], ["Wdim"], keepdims=1)
    n("Conv", ["input", "t_wnz"], ["nz"])                                # [1,1,30,30] nonzero indicator

    # --- separator rows/cols: nonzero-count >= 0.55*dim ---
    n("ReduceSum", ["nz", "t_ax3"], ["row_cnt"], keepdims=1)             # [1,1,30,1]
    n("ReduceSum", ["nz", "t_ax2"], ["col_cnt"], keepdims=1)             # [1,1,1,30]
    n("Mul", ["Wdim", "t_055"], ["thr_r"])                              # 0.55*W
    n("Mul", ["Hdim", "t_055"], ["thr_c"])                              # 0.55*H
    n("GreaterOrEqual", ["row_cnt", "thr_r"], ["sep_rowB"]); n("Cast", ["sep_rowB"], ["sep_row"], to=1)
    n("GreaterOrEqual", ["col_cnt", "thr_c"], ["sep_colB"]); n("Cast", ["sep_colB"], ["sep_col"], to=1)

    # --- box extent: r0=min sep row, r1=max sep row (and cols) ---
    # max via iota*mask ; min via iota*mask + (1-mask)*BIG
    n("Mul", ["ck_iota_r", "sep_row"], ["sri_m"])
    n("ReduceMax", ["sri_m"], ["r1"], axes=[2], keepdims=1)              # max sep row idx
    n("Sub", ["t_one", "sep_row"], ["nsr"]); n("Mul", ["nsr", "t_big"], ["nsrb"])
    n("Add", ["sri_m", "nsrb"], ["sri_min"]); n("Neg", ["sri_min"], ["sri_minn"])
    n("ReduceMax", ["sri_minn"], ["r0n"], axes=[2], keepdims=1); n("Neg", ["r0n"], ["r0"])
    n("Mul", ["ck_iota_c", "sep_col"], ["sci_m"])
    n("ReduceMax", ["sci_m"], ["c1"], axes=[3], keepdims=1)
    n("Sub", ["t_one", "sep_col"], ["nsc"]); n("Mul", ["nsc", "t_big"], ["nscb"])
    n("Add", ["sci_m", "nscb"], ["sci_min"]); n("Neg", ["sci_min"], ["sci_minn"])
    n("ReduceMax", ["sci_minn"], ["c0n"], axes=[3], keepdims=1); n("Neg", ["c0n"], ["c0"])

    # keep_row = (iota_r>=r0)&(iota_r<=r1) ; keep_col similar
    n("GreaterOrEqual", ["ck_iota_r", "r0"], ["krloB"]); n("Cast", ["krloB"], ["krlo"], to=1)
    n("GreaterOrEqual", ["r1", "ck_iota_r"], ["krhiB"]); n("Cast", ["krhiB"], ["krhi"], to=1)
    n("Mul", ["krlo", "krhi"], ["keep_row"])                            # [1,1,30,1]
    n("GreaterOrEqual", ["ck_iota_c", "c0"], ["kcloB"]); n("Cast", ["kcloB"], ["kclo"], to=1)
    n("GreaterOrEqual", ["c1", "ck_iota_c"], ["kchiB"]); n("Cast", ["kchiB"], ["kchi"], to=1)
    n("Mul", ["kclo", "kchi"], ["keep_col"])                            # [1,1,1,30]

    # box dims
    n("ReduceSum", ["keep_row", "t_ax2"], ["Hb"], keepdims=1)
    n("ReduceSum", ["keep_col", "t_ax3"], ["Wb"], keepdims=1)
    n("Sub", ["Hb", "t_one"], ["Hb1"]); n("Sub", ["Wb", "t_one"], ["Wb1"])

    # compact input -> box frame [1,10,Hb,Wb] at top-left
    ck.emit_compaction(n, "keep_row", "keep_col", "input", "box")

    # --- region / border / interior masks in box frame ---
    n("Less", ["ck_iota_r", "Hb"], ["rinB"]); n("Cast", ["rinB"], ["rin"], to=1)
    n("Less", ["ck_iota_c", "Wb"], ["cinB"]); n("Cast", ["cinB"], ["cin"], to=1)
    n("Mul", ["rin", "cin"], ["inreg"])                                 # [1,1,30,30] box region
    # interior = rows 1..Hb-2, cols 1..Wb-2
    n("GreaterOrEqual", ["ck_iota_r", "t_one"], ["irloB"]); n("Cast", ["irloB"], ["irlo"], to=1)
    n("Sub", ["Hb", "t_two"], ["Hb2"]); n("Sub", ["Wb", "t_two"], ["Wb2"])
    n("GreaterOrEqual", ["Hb2", "ck_iota_r"], ["irhiB"]); n("Cast", ["irhiB"], ["irhi"], to=1)
    n("Mul", ["irlo", "irhi"], ["irow"])                               # [1,1,30,1]
    n("GreaterOrEqual", ["ck_iota_c", "t_one"], ["icloB"]); n("Cast", ["icloB"], ["iclo"], to=1)
    n("GreaterOrEqual", ["Wb2", "ck_iota_c"], ["ichiB"]); n("Cast", ["ichiB"], ["ichi"], to=1)
    n("Mul", ["iclo", "ichi"], ["icol"])                               # [1,1,1,30]
    n("Mul", ["irow", "icol"], ["interior"])                           # [1,1,30,30]
    n("Sub", ["inreg", "interior"], ["border"])                        # border ring

    # --- border colors (one-hot) via argmax-channel of each border line sum ---
    n("Equal", ["ck_iota_r", "t_zero"], ["r0eB"]); n("Cast", ["r0eB"], ["r0e"], to=1)
    n("Equal", ["ck_iota_r", "Hb1"], ["rHeB"]); n("Cast", ["rHeB"], ["rHe"], to=1)
    n("Equal", ["ck_iota_c", "t_zero"], ["c0eB"]); n("Cast", ["c0eB"], ["c0e"], to=1)
    n("Equal", ["ck_iota_c", "Wb1"], ["cWeB"]); n("Cast", ["cWeB"], ["cWe"], to=1)

    def edgecolor(linemask, out):
        # linemask [1,1,30,1] or [1,1,1,30]; restrict to box region then argmax channel
        n("Mul", [linemask, "inreg"], [f"{out}_lm"])
        n("Mul", ["box", f"{out}_lm"], [f"{out}_e"])
        n("ReduceSum", [f"{out}_e", "t_ax23"], [f"{out}_c"], keepdims=1)   # [1,10,1,1]
        n("Sub", [f"{out}_c", "t_kill0"], [f"{out}_s"])                    # drop channel0
        n("ReduceMax", [f"{out}_s"], [f"{out}_m"], axes=[1], keepdims=1)
        n("Equal", [f"{out}_s", f"{out}_m"], [f"{out}_B"]); n("Cast", [f"{out}_B"], [out], to=1)
    edgecolor("r0e", "rc0")   # top border color one-hot [1,10,1,1]
    edgecolor("rHe", "rc1")   # bottom
    edgecolor("c0e", "cc0")   # left
    edgecolor("cWe", "cc1")   # right

    # --- mark color one-hot: argmax-channel of interior sum (chan0 excluded) ---
    n("Mul", ["box", "interior"], ["inbox"])
    n("ReduceSum", ["inbox", "t_ax23"], ["mcnt"], keepdims=1)              # [1,10,1,1]
    n("Sub", ["mcnt", "t_kill0"], ["msc"])
    n("ReduceMax", ["msc"], ["mmax"], axes=[1], keepdims=1)
    n("Equal", ["msc", "mmax"], ["markB"]); n("Cast", ["markB"], ["mark"], to=1)  # [1,10,1,1]
    # also need a guard for empty interior: if no interior nonzero, mark mask must be empty.
    # We compute mark mask from the box and intersect with interior; empty interior => mm all 0.

    # interior mark mask mm [1,1,30,30]
    n("Conv", ["box", "mark"], ["markpix"])                               # [1,1,30,30]
    n("Mul", ["markpix", "interior"], ["mm"])

    # --- direction indicators: dot(mark, border_color) over channels ---
    def dirind2(bc, out):
        n("Mul", ["mark", bc], [f"{out}_p"])                              # [1,10,1,1]
        n("ReduceSum", [f"{out}_p", "t_ax_c"], [out], keepdims=1)         # over channel -> [1,1,1,1]
    inits.append(ck.const("t_ax_c", np.array([1], np.int64)))
    dirind2("rc1", "is_down")   # mark==bottom -> fill downward
    dirind2("rc0", "is_up")     # mark==top    -> fill upward
    dirind2("cc1", "is_right")  # mark==right  -> fill rightward
    dirind2("cc0", "is_left")   # mark==left   -> fill leftward

    # --- directional cumulative fills of mm ---
    # down: forward cumsum along rows (axis2)
    n("CumSum", ["mm", "t_cax2"], ["cd0"]); n("Sign", ["cd0"], ["fill_down0"])
    n("Mul", ["fill_down0", "interior"], ["fill_down"])
    # up: reverse cumsum along rows = total - (cumsum - mm)
    n("ReduceSum", ["mm", "t_ax2"], ["mm_rtot"], keepdims=1)             # [1,1,1,30]
    n("Sub", ["cd0", "mm"], ["cd_excl"])
    n("Sub", ["mm_rtot", "cd_excl"], ["rcu0"]); n("Sign", ["rcu0"], ["fill_up0"])
    n("Mul", ["fill_up0", "interior"], ["fill_up"])
    # right: forward cumsum along cols (axis3)
    n("CumSum", ["mm", "t_cax3"], ["cr0"]); n("Sign", ["cr0"], ["fill_right0"])
    n("Mul", ["fill_right0", "interior"], ["fill_right"])
    # left: reverse cumsum along cols
    n("ReduceSum", ["mm", "t_ax3"], ["mm_ctot"], keepdims=1)             # [1,1,30,1]
    n("Sub", ["cr0", "mm"], ["cr_excl"])
    n("Sub", ["mm_ctot", "cr_excl"], ["lcu0"]); n("Sign", ["lcu0"], ["fill_left0"])
    n("Mul", ["fill_left0", "interior"], ["fill_left"])

    # select fill by direction
    n("Mul", ["is_down", "fill_down"], ["sd"])
    n("Mul", ["is_up", "fill_up"], ["su"])
    n("Mul", ["is_right", "fill_right"], ["srt"])
    n("Mul", ["is_left", "fill_left"], ["slt"])
    n("Add", ["sd", "su"], ["s01"]); n("Add", ["srt", "slt"], ["s23"])
    n("Add", ["s01", "s23"], ["fill"])                                   # [1,1,30,30]

    # --- assemble output ---
    # border ring kept as-is
    n("Mul", ["box", "border"], ["out_border"])                          # [1,10,30,30]
    # interior: filled cells -> mark color
    n("Mul", ["fill", "mark"], ["out_fill"])                             # [1,10,30,30]
    # interior non-filled -> background channel0
    n("Sub", ["interior", "fill"], ["ibg0"]); n("Mul", ["ibg0", "t_e0"], ["out_ibg"])
    n("Add", ["out_border", "out_fill"], ["o0"]); n("Add", ["o0", "out_ibg"], ["output"])

    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
