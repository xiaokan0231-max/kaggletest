"""task185: lattice of evenly-spaced full rows/cols in the most-common color.
At each lattice intersection there may be an intruder color (non-lattice,
non-background). The intruder occupies a 4x4 block of intersections. The 3x3
output cell (a,b) = the common color iff the 4 surrounding intersections
(a,b),(a,b+1),(a+1,b),(a+1,b+1) are all equal AND nonzero, else background.

Pipeline (all MatMul/Conv/elementwise, no loops):
  lattice color  -> colorkit.emit_select_common -> lat mask via Conv
  full row/col   -> count of lattice per row/col >= 0.7 * (W or H)
  intersection   -> full_row * full_col
  intruder pres  -> inter * nonzero * (1 - lat)
  G one-hot      -> input * intruder_presence
  compact G's nonzero rows/cols -> 4x4 one-hot block at top-left (G4)
  corner-AND     -> G4 * (G4 shift cols<-1) * (G4 shift rows^1) * (both)
  assemble       -> foreground channels + background(e0) within the 3x3 region
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import colorkit as colk  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper, TensorProto  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task185"):
    inits = ck.base_inits() + colk.color_inits() + [
        ck.const("t_one", np.float32(1.0)),
        ck.const("t_07", np.float32(0.7)),
        ck.const("t_ax1", np.array([1], np.int64)),
        ck.const("t_ax2", np.array([2], np.int64)),
        ck.const("t_ax3", np.array([3], np.int64)),
        ck.const("t_f3", np.float32(3.0)),
        ck.const("t_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)  # presence, row_present, col_present, Hdim, Wdim, nz, nz_row, nz_col

    # --- lattice color (most common nonzero) -> per-pixel mask lat[1,1,30,30] ---
    colk.emit_select_common(n, "sel")            # [1,10,1,1] one-hot lattice color
    n("Conv", ["input", "sel"], ["lat"])         # [1,1,30,30] 1 where pixel==lattice

    # --- full rows / full cols ---
    n("ReduceSum", ["lat", "t_ax3"], ["rowcount"], keepdims=1)   # [1,1,30,1]
    n("ReduceSum", ["lat", "t_ax2"], ["colcount"], keepdims=1)   # [1,1,1,30]
    n("Mul", ["Wdim", "t_07"], ["thr_row"])                      # 0.7*W
    n("Mul", ["Hdim", "t_07"], ["thr_col"])                      # 0.7*H
    n("GreaterOrEqual", ["rowcount", "thr_row"], ["frB"])
    n("Cast", ["frB"], ["full_row"], to=TensorProto.FLOAT)       # [1,1,30,1]
    n("GreaterOrEqual", ["colcount", "thr_col"], ["fcB"])
    n("Cast", ["fcB"], ["full_col"], to=TensorProto.FLOAT)       # [1,1,1,30]

    # --- intersection / intruder presence ---
    n("Mul", ["full_row", "full_col"], ["inter"])               # [1,1,30,30]
    n("Sub", ["t_one", "lat"], ["notlat"])
    n("Mul", ["inter", "nz"], ["inter_nz"])
    n("Mul", ["inter_nz", "notlat"], ["intr"])                  # [1,1,30,30] intruder pres
    n("Mul", ["input", "intr"], ["G"])                          # [1,10,30,30] one-hot at intersections

    # nonzero rows/cols of G (== intr presence)
    n("ReduceMax", ["intr"], ["g_row"], axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["intr"], ["g_col"], axes=[2], keepdims=1)   # [1,1,1,30]

    # compact G -> 4x4 one-hot block at top-left
    ck.emit_compaction(n, "g_row", "g_col", "G", "G4")          # [1,10,30,30]

    # --- shift matrices ---
    # rmc = iota_r - iota_c   (broadcast to [1,1,30,30])
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    n("Equal", ["rmc", "t_one"], ["ScB"])                       # rmc==1 -> cols left by1
    n("Cast", ["ScB"], ["Sc"], to=TensorProto.FLOAT)
    n("Sub", ["t_one", "t_one"], ["zero"])                      # 0.0 scalar
    n("Sub", ["zero", "t_one"], ["neg1"])                       # -1.0
    n("Equal", ["rmc", "neg1"], ["SrB"])                       # rmc==-1 -> rows up by1
    n("Cast", ["SrB"], ["Sr"], to=TensorProto.FLOAT)

    # corner alignment onto (a,b)
    # A=(a,b)=G4 ; B=(a,b+1)=G4@Sc ; C=(a+1,b)=Sr@G4 ; D=(a+1,b+1)=Sr@G4@Sc
    n("MatMul", ["G4", "Sc"], ["B"])
    n("MatMul", ["Sr", "G4"], ["C"])
    n("MatMul", ["C", "Sc"], ["D"])
    n("Mul", ["G4", "B"], ["AB"])
    n("Mul", ["C", "D"], ["CD"])
    n("Mul", ["AB", "CD"], ["out_fg"])                         # [1,10,30,30] 4-corner AND

    # --- assemble: foreground + background within 3x3 region ---
    n("ReduceSum", ["out_fg", "t_ax1"], ["fgp"], keepdims=1)   # [1,1,30,30]
    n("Less", ["ck_iota_r", "t_f3"], ["r3B"]); n("Cast", ["r3B"], ["r3"], to=TensorProto.FLOAT)
    n("Less", ["ck_iota_c", "t_f3"], ["c3B"]); n("Cast", ["c3B"], ["c3"], to=TensorProto.FLOAT)
    n("Mul", ["r3", "c3"], ["reg3"])                           # [1,1,30,30]
    n("Sub", ["t_one", "fgp"], ["nofg"])
    n("Mul", ["reg3", "nofg"], ["bgm"])
    n("Mul", ["bgm", "t_e0"], ["out_bg"])
    n("Add", ["out_fg", "out_bg"], ["output"])

    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
