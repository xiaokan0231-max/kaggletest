"""task195: nonzero bbox is 9x9 (a 3x3 grid of 3x3 blocks). Let P be the 3x3
macro pattern (block filled iff it has any nonzero). Output (9x9) = the
self-similar fractal P (X) P in the single foreground color: out[3A+a,3B+b] =
fg iff P[A,B] and P[a,b], else background.

P = sign of 3x3 block-sums of the nonzero mask (via a block-sum matrix S).
P_exp = upscale P x3 (E3 @ P @ E3^T); P_tile = tile P x3 (Tr @ P @ Tr^T);
out_presence = P_exp * P_tile.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import colorkit as colk  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task195"):
    S = np.zeros((1, 1, 30, 30), np.float32)
    for r in range(30):
        S[0, 0, r // 3, r] = 1.0                       # block-sum (rows -> 3-blocks)
    E3 = np.zeros((1, 1, 30, 30), np.float32)
    for i in range(30):
        E3[0, 0, i, i // 3] = 1.0                      # upscale x3
    Tr = np.zeros((1, 1, 30, 30), np.float32)
    for r in range(9):
        Tr[0, 0, r, r % 3] = 1.0                       # tile x3, bounded to first 9
    inits = ck.base_inits() + colk.color_inits() + [
        ck.const("f_S", S), ck.const("f_E3", E3), ck.const("f_Tr", Tr),
        ck.const("f_one", np.float32(1.0)), ck.const("f9", np.float32(9.0)),
        ck.const("f_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)                                # nz, nz_row, nz_col
    ck.emit_compaction(n, "nz_row", "nz_col", "nz", "cnz")   # 9x9 nonzero mask at TL
    colk.emit_select_common(n, "selfg")                # the single foreground color
    # macro P = sign(block-sum)
    n("Transpose", ["f_S"], ["f_ST"], perm=[0, 1, 3, 2])
    n("MatMul", ["f_S", "cnz"], ["bs0"]); n("MatMul", ["bs0", "f_ST"], ["bsum"])
    n("Sign", ["bsum"], ["P"])                          # [1,1,30,30], P in [0:3,0:3]
    # kron P (X) P
    n("Transpose", ["f_E3"], ["f_E3T"], perm=[0, 1, 3, 2])
    n("MatMul", ["f_E3", "P"], ["pe0"]); n("MatMul", ["pe0", "f_E3T"], ["Pexp"])
    n("Transpose", ["f_Tr"], ["f_TrT"], perm=[0, 1, 3, 2])
    n("MatMul", ["f_Tr", "P"], ["pt0"]); n("MatMul", ["pt0", "f_TrT"], ["Ptile"])
    n("Mul", ["Pexp", "Ptile"], ["pres"])               # [1,1,30,30], fractal presence
    # assemble color + background within 9x9
    n("Less", ["ck_iota_r", "f9"], ["r9B"]); n("Cast", ["r9B"], ["r9"], to=1)
    n("Less", ["ck_iota_c", "f9"], ["c9B"]); n("Cast", ["c9B"], ["c9"], to=1)
    n("Mul", ["r9", "c9"], ["reg9"])
    n("Mul", ["pres", "selfg"], ["out_fg"])
    n("Sub", ["f_one", "pres"], ["notp"]); n("Mul", ["reg9", "notp"], ["bgm"])
    n("Mul", ["bgm", "f_e0"], ["out_bg"])
    n("Add", ["out_fg", "out_bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
