"""Solve 'odd-one-out quadrant' tasks (e.g. task065, task207).

Grid is (2h+1)x(2w+1): a center cross separator splits it into 4 h x w
quadrants, 3 identical and 1 different. Output = the odd quadrant.

Plan: align all 4 quadrants to the top-left via MatMul row/col shifts, score
each by total |difference| to the other three, and output the highest-scoring
(the odd) quadrant.
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
        ck.const("oq_half", np.float32(0.5)),
        ck.const("oq_one", np.float32(1.0)),
        ck.const("oq_ax123", np.array([1, 2, 3], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    # separator indices sep_r=floor(H/2), sep_c=floor(W/2); shift amounts +1
    n("Mul", ["Hdim", "oq_half"], ["hh0"]); n("Floor", ["hh0"], ["sepr"])   # [1,1,1,1]
    n("Mul", ["Wdim", "oq_half"], ["ww0"]); n("Floor", ["ww0"], ["sepc"])
    n("Add", ["sepr", "oq_one"], ["kr"])    # shift rows up by sep_r+1
    n("Add", ["sepc", "oq_one"], ["kc"])    # shift cols left by sep_c+1
    # position masks
    n("Less", ["ck_iota_r", "sepr"], ["topB"]); n("Cast", ["topB"], ["top"], to=1)   # [1,1,30,1]
    n("Greater", ["ck_iota_r", "sepr"], ["botB"]); n("Cast", ["botB"], ["bot"], to=1)
    n("Less", ["ck_iota_c", "sepc"], ["lefB"]); n("Cast", ["lefB"], ["lef"], to=1)   # [1,1,1,30]
    n("Greater", ["ck_iota_c", "sepc"], ["rigB"]); n("Cast", ["rigB"], ["rig"], to=1)
    n("Mul", ["top", "lef"], ["mTL"]); n("Mul", ["top", "rig"], ["mTR"])
    n("Mul", ["bot", "lef"], ["mBL"]); n("Mul", ["bot", "rig"], ["mBR"])
    # shift matrices: Qc shifts cols left by kc ; Qr shifts rows up by kr
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])        # [1,1,30,30]  b-a
    n("Equal", ["rmc", "kc"], ["QcB"]); n("Cast", ["QcB"], ["Qc"], to=1)
    n("Neg", ["kr"], ["nkr"])
    n("Equal", ["rmc", "nkr"], ["QrB"]); n("Cast", ["QrB"], ["Qr"], to=1)
    # masked quadrants aligned to TL: A=TL, B=TR<<cols, C=BL^^rows, D=BR both
    n("Mul", ["input", "mTL"], ["A"])
    n("Mul", ["input", "mTR"], ["TRm"]); n("MatMul", ["TRm", "Qc"], ["B"])
    n("Mul", ["input", "mBL"], ["BLm"]); n("MatMul", ["Qr", "BLm"], ["C"])
    n("Mul", ["input", "mBR"], ["BRm"]); n("MatMul", ["BRm", "Qc"], ["BRc"]); n("MatMul", ["Qr", "BRc"], ["D"])

    def diff(x, y, out):
        n("Sub", [x, y], [f"{out}_s"]); n("Abs", [f"{out}_s"], [f"{out}_a"])
        n("ReduceSum", [f"{out}_a", "oq_ax123"], [out], keepdims=1)   # scalar [1,1,1,1]
    for a, b in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
        diff(a, b, f"d{a}{b}")
    n("Add", ["dAB", "dAC"], ["sA0"]); n("Add", ["sA0", "dAD"], ["sA"])
    n("Add", ["dAB", "dBC"], ["sB0"]); n("Add", ["sB0", "dBD"], ["sB"])
    n("Add", ["dAC", "dBC"], ["sC0"]); n("Add", ["sC0", "dCD"], ["sC"])
    n("Add", ["dAD", "dBD"], ["sD0"]); n("Add", ["sD0", "dCD"], ["sD"])
    # max score, then one-hot weights
    n("Max", ["sA", "sB", "sC", "sD"], ["smax"])
    for q in ("A", "B", "C", "D"):
        n("Equal", [f"s{q}", "smax"], [f"w{q}B"]); n("Cast", [f"w{q}B"], [f"w{q}"], to=1)
        n("Mul", [f"w{q}", q], [f"o{q}"])
    n("Add", ["oA", "oB"], ["oab"]); n("Add", ["oC", "oD"], ["ocd"])
    n("Add", ["oab", "ocd"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build(sys.argv[1])
