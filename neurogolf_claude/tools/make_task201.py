"""task201: 4-corner frame defines output bbox (H x W). Left/right frame columns
(legs) are solid single colors lcol/rcol. The 'shapeblock' = input with the frame
ring removed, cropped to its own bbox, is dropped verbatim into the interior
(rows 1..H-2, cols 1..W-2). It is flipped LR iff the lcol-colored cells in the
shapeblock have a larger mean column than the rcol-colored cells (so the left-leg
color ends up on the left). Output: corners=4, left col interior=lcol, right col
interior=rcol, interior=shapeblock, every other in-bbox cell = background (ch0=1).

ONNX plan:
  - color-4 mask -> bbox-fill rows/cols -> rowfill/colfill, H, W.
  - leftcol/rightcol selectors (Equal on cumsum of colfill); interior-row band.
  - leg colors lsel/rsel via argmax of input summed over leg region.
  - frame ring removal -> m; compact m to bbox -> sb0 (shapeblock at TL).
  - conditional LR flip via reversal matrix Equal(iota_r+iota_c, (W-2)-1),
    decided by Greater(sl*cr, sr*cl) (cross-multiplied mean-col comparison).
  - shift sb into interior (down 1, right 1); assemble with bg fill of bbox.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task201"):
    oh4 = np.zeros((1, 10, 1, 1), np.float32); oh4[0, 4, 0, 0] = 1.0
    notch4 = np.ones((1, 10, 1, 1), np.float32); notch4[0, 4, 0, 0] = 0.0
    kill0 = (BIG * np.array([1] + [0] * 9, np.float32)).reshape(1, 10, 1, 1)
    inits = ck.base_inits() + [
        ck.const("p_oh4", oh4), ck.const("p_notch4", notch4),
        ck.const("p_kill0", kill0),
        ck.const("p_one", np.float32(1.0)), ck.const("p_two", np.float32(2.0)),
        ck.const("p_zero", np.float32(0.0)), ck.const("p_n1", np.float32(-1.0)),
        ck.const("p_ax2", np.array([2], np.int64)),
        ck.const("p_ax3", np.array([3], np.int64)),
        ck.const("p_ax23", np.array([2, 3], np.int64)),
        ck.const("p_cax2", np.array(2, np.int64)),
        ck.const("p_cax3", np.array(3, np.int64)),
        ck.const("p_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("p_wnz", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])

    # --- color-4 frame: presence rows/cols ---
    n("Conv", ["input", "p_oh4"], ["m4"])                       # [1,1,30,30]
    n("ReduceMax", ["m4"], ["r4"], axes=[3], keepdims=1)        # [1,1,30,1]
    n("ReduceMax", ["m4"], ["c4"], axes=[2], keepdims=1)        # [1,1,1,30]

    # bbox-fill helper: band -> filled range between first/last occupied index.
    def fill(band, cax, rax, out):
        n("CumSum", [band, cax], [f"{out}_cum"]); n("Sign", [f"{out}_cum"], [f"{out}_hl"])
        n("ReduceSum", [band, rax], [f"{out}_tot"], keepdims=1)
        n("Sub", [f"{out}_cum", band], [f"{out}_cex"])
        n("Sub", [f"{out}_tot", f"{out}_cex"], [f"{out}_rr0"]); n("Sign", [f"{out}_rr0"], [f"{out}_hr"])
        n("Mul", [f"{out}_hl", f"{out}_hr"], [out])
    fill("r4", "p_cax2", "p_ax2", "rowfill")                    # [1,1,30,1] rows r0..r1
    fill("c4", "p_cax3", "p_ax3", "colfill")                    # [1,1,1,30] cols c0..c1

    n("ReduceSum", ["rowfill", "p_ax2"], ["H"], keepdims=1)     # [1,1,1,1]
    n("ReduceSum", ["colfill", "p_ax3"], ["W"], keepdims=1)
    n("Sub", ["H", "p_one"], ["H1"])                            # H-1
    n("Sub", ["W", "p_one"], ["W1"])                            # W-1
    n("Sub", ["W", "p_two"], ["W2"])                            # W-2

    # --- left/right frame columns, interior row band ---
    n("CumSum", ["colfill", "p_cax3"], ["ccum"])               # [1,1,1,30]
    n("Equal", ["ccum", "p_one"], ["leftcB"]); n("Cast", ["leftcB"], ["leftc"], to=1)   # col==c0
    n("Equal", ["ccum", "W"], ["rcumB"]); n("Cast", ["rcumB"], ["rcum"], to=1)
    n("Mul", ["rcum", "colfill"], ["rightc"])                  # col==c1
    n("CumSum", ["rowfill", "p_cax2"], ["rcum2"])              # [1,1,30,1]
    n("Equal", ["rcum2", "p_one"], ["toprB"]); n("Cast", ["toprB"], ["topr"], to=1)     # row==r0
    n("Equal", ["rcum2", "H"], ["botcB"]); n("Cast", ["botcB"], ["botcum"], to=1)
    n("Mul", ["botcum", "rowfill"], ["botr"])                  # row==r1
    n("Sub", ["rowfill", "topr"], ["intr0"]); n("Sub", ["intr0", "botr"], ["introws"])  # rows r0+1..r1-1

    # --- leg colors lsel/rsel (one-hot) via argmax over leg region ---
    def legcol(colmask, out):
        n("Mul", [colmask, "introws"], [f"{out}_reg"])         # [1,1,30,30]
        n("Mul", ["input", f"{out}_reg"], [f"{out}_in"])
        n("ReduceSum", [f"{out}_in", "p_ax23"], [f"{out}_cnt"], keepdims=1)  # [1,10,1,1]
        n("Sub", [f"{out}_cnt", "p_kill0"], [f"{out}_s"])
        n("ReduceMax", [f"{out}_s"], [f"{out}_mx"], axes=[1], keepdims=1)
        n("Equal", [f"{out}_s", f"{out}_mx"], [f"{out}_B"]); n("Cast", [f"{out}_B"], [out], to=1)
    legcol("leftc", "lsel")
    legcol("rightc", "rsel")

    # --- shapeblock: remove frame ring from input, compact to bbox ---
    n("Mul", ["rowfill", "colfill"], ["rect"])                 # [1,1,30,30] full bbox rect
    n("Sub", ["rowfill", "topr"], ["irow0"]); n("Sub", ["irow0", "botr"], ["irow"])     # interior rows
    n("Sub", ["colfill", "leftc"], ["icol0"]); n("Sub", ["icol0", "rightc"], ["icol"])  # interior cols
    n("Mul", ["irow", "icol"], ["int_rect"])                   # interior of bbox
    n("Sub", ["rect", "int_rect"], ["ring"])                   # 1-thick perimeter
    n("Sub", ["p_one", "ring"], ["notring"])
    n("Mul", ["input", "notring"], ["m"])                      # [1,10,30,30]
    n("Conv", ["m", "p_wnz"], ["nzm"])                         # nonzero presence [1,1,30,30]
    n("ReduceMax", ["nzm"], ["sbrow"], axes=[3], keepdims=1)
    n("ReduceMax", ["nzm"], ["sbcol"], axes=[2], keepdims=1)
    ck.emit_compaction(n, "sbrow", "sbcol", "m", "sb0")        # shapeblock packed at TL

    # --- conditional LR flip ---
    n("Conv", ["sb0", "lsel"], ["lmask"])                      # lcol cells in shapeblock
    n("Conv", ["sb0", "rsel"], ["rmask"])
    n("Mul", ["ck_iota_c", "lmask"], ["lwx"])
    n("ReduceSum", ["lwx", "p_ax23"], ["sl"], keepdims=1)      # sum of cols, [1,1,1,1]
    n("ReduceSum", ["lmask", "p_ax23"], ["cl"], keepdims=1)
    n("Mul", ["ck_iota_c", "rmask"], ["rwx"])
    n("ReduceSum", ["rwx", "p_ax23"], ["sr"], keepdims=1)
    n("ReduceSum", ["rmask", "p_ax23"], ["cr"], keepdims=1)
    n("Mul", ["sl", "cr"], ["lhs"]); n("Mul", ["sr", "cl"], ["rhs"])
    n("Greater", ["lhs", "rhs"], ["flipB"]); n("Cast", ["flipB"], ["flip"], to=1)  # scalar [1,1,1,1]

    # reversal matrix around center (W-2)-1: Rev[k,j]=1 iff k+j==W-3
    n("Sub", ["W2", "p_one"], ["revK"])                        # (W-2)-1 = W-3
    n("Add", ["ck_iota_r", "ck_iota_c"], ["ipj"])             # [1,1,30,30] k+j
    n("Equal", ["ipj", "revK"], ["revB"]); n("Cast", ["revB"], ["revM"], to=1)
    n("MatMul", ["sb0", "revM"], ["sbflip"])                  # cols reversed
    # blend: sb = flip ? sbflip : sb0
    n("Mul", ["flip", "sbflip"], ["pf1"])
    n("Sub", ["p_one", "flip"], ["nflip"]); n("Mul", ["nflip", "sb0"], ["pf0"])
    n("Add", ["pf1", "pf0"], ["sbsel"])

    # --- position shapeblock into interior (down 1, right 1) ---
    n("Equal", ["rmc", "p_one"], ["dnB"]); n("Cast", ["dnB"], ["Dn"], to=1)   # down-by-1
    n("Equal", ["rmc", "p_n1"], ["rtB"]); n("Cast", ["rtB"], ["Rt"], to=1)    # right-by-1
    n("MatMul", ["sbsel", "Rt"], ["sb_r"])
    n("MatMul", ["Dn", "sb_r"], ["sb_pos"])                   # interior content at (1,1)
    n("Mul", ["sb_pos", "p_wnz"], ["sb_fg"])                  # drop ch0; keep nonzero colors

    # --- frame pieces in output canvas (anchored TL) ---
    # corners: (0,0),(0,W-1),(H-1,0),(H-1,W-1)
    n("Equal", ["ck_iota_r", "p_zero"], ["row0B"]); n("Cast", ["row0B"], ["row0"], to=1)  # [1,1,30,1]
    n("Equal", ["ck_iota_r", "H1"], ["rowHB"]); n("Cast", ["rowHB"], ["rowH"], to=1)
    n("Equal", ["ck_iota_c", "p_zero"], ["col0B"]); n("Cast", ["col0B"], ["col0"], to=1)  # [1,1,1,30]
    n("Equal", ["ck_iota_c", "W1"], ["colWB"]); n("Cast", ["colWB"], ["colW"], to=1)
    n("Add", ["row0", "rowH"], ["rowTB"])                     # rows {0,H-1} [1,1,30,1]
    n("Add", ["col0", "colW"], ["colLR"])                     # cols {0,W-1} [1,1,1,30]
    n("Mul", ["rowTB", "colLR"], ["corners"])                 # 4 corner cells [1,1,30,30]
    n("Mul", ["corners", "p_oh4"], ["corner_vec"])           # [1,10,30,30]
    # interior row band of output (rows 1..H-2): inside bbox but not top/bottom row
    n("Less", ["ck_iota_r", "H1"], ["rltB"]); n("Cast", ["rltB"], ["rlt"], to=1)  # row<H-1
    n("Sub", ["rlt", "row0"], ["outintr"])                   # rows 1..H-2 [1,1,30,1]
    n("Mul", ["col0", "outintr"], ["leftcol_reg"])          # col0 & interior rows
    n("Mul", ["colW", "outintr"], ["rightcol_reg"])
    n("Mul", ["leftcol_reg", "lsel"], ["leftcol_vec"])      # [1,10,30,30]
    n("Mul", ["rightcol_reg", "rsel"], ["rightcol_vec"])

    # --- assemble: foreground colors then background fill of bbox ---
    n("Add", ["corner_vec", "leftcol_vec"], ["a0"])
    n("Add", ["a0", "rightcol_vec"], ["a1"])
    n("Add", ["a1", "sb_fg"], ["fgcolor"])                   # all nonzero-color content
    n("Conv", ["fgcolor", "p_wnz"], ["nzpres"])             # [1,1,30,30] presence of color
    # output rectangle: row<H and col<W
    n("Less", ["ck_iota_r", "H"], ["rinB"]); n("Cast", ["rinB"], ["rin"], to=1)   # [1,1,30,1]
    n("Less", ["ck_iota_c", "W"], ["cinB"]); n("Cast", ["cinB"], ["cin"], to=1)   # [1,1,1,30]
    n("Mul", ["rin", "cin"], ["outrect"])                    # [1,1,30,30]
    n("Sub", ["outrect", "nzpres"], ["bgcells0"])           # in-bbox, not colored
    n("Max", ["bgcells0", "p_zero"], ["bgcells"])           # clamp (defensive)
    n("Mul", ["bgcells", "p_e0"], ["bg_vec"])               # ch0=1 there
    n("Add", ["fgcolor", "bg_vec"], ["output"])

    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
