"""task109: (2h+1)x(2w+1) grid with a center cross separator; only the top-left
quadrant holds a shape. Output (2h x 2w, separator removed) = that shape
recolored to the separator color and reflected into 4-fold symmetry.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task109"):
    inits = ck.base_inits() + [
        ck.const("h_half", np.float32(0.5)), ck.const("h_one", np.float32(1.0)),
        ck.const("h_two", np.float32(2.0)),
        ck.const("h_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("h_ax23", np.array([2, 3], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)   # gives nz, Hdim, Wdim, ck_iota_r/c
    # scalars
    n("Mul", ["Hdim", "h_half"], ["hh0"]); n("Floor", ["hh0"], ["Hh"])
    n("Mul", ["Wdim", "h_half"], ["ww0"]); n("Floor", ["ww0"], ["Ww"])
    n("Sub", ["Hdim", "h_one"], ["H1"]); n("Sub", ["Wdim", "h_one"], ["W1"])
    n("Sub", ["Hdim", "h_two"], ["Hm2"]); n("Sub", ["Wdim", "h_two"], ["Wm2"])
    n("Add", ["ck_iota_r", "ck_iota_c"], ["rpc"])      # [1,1,30,30] r+c
    # separator color = one-hot at (Hh, Ww)
    n("Equal", ["ck_iota_r", "Hh"], ["selrB"]); n("Cast", ["selrB"], ["selr"], to=1)
    n("Equal", ["ck_iota_c", "Ww"], ["selcB"]); n("Cast", ["selcB"], ["selc"], to=1)
    n("Mul", ["selr", "selc"], ["pmask"])
    n("Mul", ["input", "pmask"], ["pt"])
    n("ReduceSum", ["pt", "h_ax23"], ["sep"], keepdims=1)            # [1,10,1,1]
    # shape in TL quadrant
    n("Less", ["ck_iota_r", "Hh"], ["tlrB"]); n("Cast", ["tlrB"], ["tlr"], to=1)
    n("Less", ["ck_iota_c", "Ww"], ["tlcB"]); n("Cast", ["tlcB"], ["tlc"], to=1)
    n("Mul", ["tlr", "tlc"], ["tlmask"])
    n("Mul", ["nz", "tlmask"], ["shapeP"])                          # [1,1,30,30]
    # reflect to full 2h x 2w
    n("Equal", ["rpc", "Wm2"], ["crB"]); n("Cast", ["crB"], ["colrev"], to=1)
    n("Equal", ["rpc", "Hm2"], ["rrB"]); n("Cast", ["rrB"], ["rowrev"], to=1)
    n("MatMul", ["shapeP", "colrev"], ["topR"]); n("Add", ["shapeP", "topR"], ["top"])
    n("MatMul", ["rowrev", "top"], ["botR"]); n("Add", ["top", "botR"], ["shapeFull"])
    # region mask & background
    n("Less", ["ck_iota_r", "H1"], ["rgrB"]); n("Cast", ["rgrB"], ["rgr"], to=1)
    n("Less", ["ck_iota_c", "W1"], ["rgcB"]); n("Cast", ["rgcB"], ["rgc"], to=1)
    n("Mul", ["rgr", "rgc"], ["reg"])
    n("Sub", ["h_one", "shapeFull"], ["notshape"]); n("Mul", ["reg", "notshape"], ["bg"])
    # assemble colors
    n("Mul", ["shapeFull", "sep"], ["out_sep"])                     # [1,10,30,30]
    n("Mul", ["bg", "h_e0"], ["out_bg"])
    n("Add", ["out_sep", "out_bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
