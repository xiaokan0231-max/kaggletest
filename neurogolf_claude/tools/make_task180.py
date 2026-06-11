"""task180: 8x8 -> 4x4 fixed-priority 4-quadrant fold.

Split into 4 quadrants (4x4 each), align all to the top-left, then overlay with
priority TR > BL > BR > TL (the highest-priority quadrant that is non-background
at a cell wins). Background filled where all quadrants are background.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task180"):
    inits = ck.base_inits() + [
        ck.const("q_one", np.float32(1.0)), ck.const("q4", np.float32(4.0)),
        ck.const("q8", np.float32(8.0)), ck.const("qn4", np.float32(-4.0)),
        ck.const("q_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("q_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    n("Equal", ["rmc", "q4"], ["QlB"]); n("Cast", ["QlB"], ["Ql"], to=1)   # cols left 4 (input@Ql)
    n("Equal", ["rmc", "qn4"], ["QuB"]); n("Cast", ["QuB"], ["Qu"], to=1)  # rows up 4 (Qu@input)
    # quadrant masks
    n("Less", ["ck_iota_r", "q4"], ["rlt4B"]); n("Cast", ["rlt4B"], ["rlt4"], to=1)
    n("GreaterOrEqual", ["ck_iota_r", "q4"], ["rge4B"]); n("Cast", ["rge4B"], ["rge4"], to=1)
    n("Less", ["ck_iota_r", "q8"], ["rlt8B"]); n("Cast", ["rlt8B"], ["rlt8"], to=1)
    n("Less", ["ck_iota_c", "q4"], ["clt4B"]); n("Cast", ["clt4B"], ["clt4"], to=1)
    n("GreaterOrEqual", ["ck_iota_c", "q4"], ["cge4B"]); n("Cast", ["cge4B"], ["cge4"], to=1)
    n("Less", ["ck_iota_c", "q8"], ["clt8B"]); n("Cast", ["clt8B"], ["clt8"], to=1)
    n("Mul", ["rge4", "rlt8"], ["rmid"]); n("Mul", ["cge4", "clt8"], ["cmid"])
    n("Mul", ["rlt4", "clt4"], ["mTL"]); n("Mul", ["rlt4", "cmid"], ["mTR"])
    n("Mul", ["rmid", "clt4"], ["mBL"]); n("Mul", ["rmid", "cmid"], ["mBR"])
    # aligned, foreground-only quadrants + nonzero presence
    def quad(name, mask, shiftL, shiftU):
        n("Mul", ["input", mask], [f"{name}m"])
        cur = f"{name}m"
        if shiftL:
            n("MatMul", [cur, "Ql"], [f"{name}L"]); cur = f"{name}L"
        if shiftU:
            n("MatMul", ["Qu", cur], [f"{name}U"]); cur = f"{name}U"
        n("Mul", [cur, "q_nzch"], [f"{name}fg"])               # foreground only
        n("ReduceMax", [f"{name}fg"], [f"{name}nz"], axes=[1], keepdims=1)
    quad("TL", "mTL", False, False)
    quad("TR", "mTR", True, False)
    quad("BL", "mBL", False, True)
    quad("BR", "mBR", True, True)
    # overlay: base TL, then BR, BL, TR override (priority TR>BL>BR>TL)
    acc = "TLfg"
    for q in ("BR", "BL", "TR"):
        n("Sub", ["q_one", f"{q}nz"], [f"{q}inv"])
        n("Mul", [f"{q}inv", acc], [f"{q}keep"])
        n("Add", [f"{q}fg", f"{q}keep"], [f"acc_{q}"]); acc = f"acc_{q}"
    # background fill within 4x4
    n("ReduceMax", [acc], ["fgres"], axes=[1], keepdims=1)
    n("Mul", ["rlt4", "clt4"], ["reg4"])  # 4x4 region
    n("Sub", ["q_one", "fgres"], ["nofg"]); n("Mul", ["reg4", "nofg"], ["bgmask"])
    n("Mul", ["bgmask", "q_e0"], ["bg"])
    n("Add", [acc, "bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
