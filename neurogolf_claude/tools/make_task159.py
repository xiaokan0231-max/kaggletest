"""task159: a hollow color-2 box; output = the box bbox with border kept color 2
and the interior = the unique other-color 3x3 cue shape upscaled (kron) by
k = interior_dim/3 in each axis.

Dynamic upscale via Ek[i,j] = (floor(i/k) == j), shape_up = Ek @ shape @ Ek^T.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task159"):
    oh2 = np.zeros((1, 10, 1, 1), np.float32); oh2[0, 2, 0, 0] = 1.0
    kill = np.zeros((1, 10, 1, 1), np.float32); kill[0, 0, 0, 0] = BIG; kill[0, 2, 0, 0] = BIG
    inits = ck.base_inits() + [
        ck.const("n_oh2", oh2), ck.const("n_kill", kill),
        ck.const("n_one", np.float32(1.0)), ck.const("n_two", np.float32(2.0)),
        ck.const("n_three", np.float32(3.0)), ck.const("n_n1", np.float32(-1.0)),
        ck.const("n_zero", np.float32(0.0)),
        ck.const("n_ax23", np.array([2, 3], np.int64)),
        ck.const("n_ax2", np.array([2], np.int64)), ck.const("n_ax3", np.array([3], np.int64)),
        ck.const("n_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    # box dims (color 2 perimeter touches every row/col)
    n("Conv", ["input", "n_oh2"], ["m2"])
    n("ReduceMax", ["m2"], ["row2"], axes=[3], keepdims=1)
    n("ReduceMax", ["m2"], ["col2"], axes=[2], keepdims=1)
    n("ReduceSum", ["row2", "n_ax2"], ["H"], keepdims=1)
    n("ReduceSum", ["col2", "n_ax3"], ["W"], keepdims=1)
    n("Sub", ["H", "n_one"], ["H1"]); n("Sub", ["W", "n_one"], ["W1"])
    # cue color (not 0, not 2)
    n("ReduceSum", ["input", "n_ax23"], ["cnt"], keepdims=1)
    n("Sub", ["cnt", "n_kill"], ["score"]); n("ReduceMax", ["score"], ["cmax"], axes=[1], keepdims=1)
    n("Equal", ["score", "cmax"], ["cueB"]); n("Cast", ["cueB"], ["cuesel"], to=1)
    n("Conv", ["input", "cuesel"], ["cuemask"])
    n("ReduceMax", ["cuemask"], ["krc"], axes=[3], keepdims=1)
    n("ReduceMax", ["cuemask"], ["kcc"], axes=[2], keepdims=1)
    ck.emit_compaction(n, "krc", "kcc", "cuemask", "shape")        # 3x3 at TL
    # k = (H-2)/3 ; Ek = Equal(floor(iota_r/k), iota_c)
    n("Sub", ["H", "n_two"], ["Hm2"]); n("Div", ["Hm2", "n_three"], ["k"])
    n("Div", ["ck_iota_r", "k"], ["ir"]); n("Floor", ["ir"], ["irf"])
    n("Equal", ["irf", "ck_iota_c"], ["EkB"]); n("Cast", ["EkB"], ["Ek"], to=1)
    n("Transpose", ["Ek"], ["EkT"], perm=[0, 1, 3, 2])
    n("MatMul", ["Ek", "shape"], ["su0"]); n("MatMul", ["su0", "EkT"], ["shapeup"])
    # place interior (shift down 1, right 1)
    n("Equal", ["rmc", "n_one"], ["Qd1B"]); n("Cast", ["Qd1B"], ["Qd1"], to=1)
    n("Equal", ["rmc", "n_n1"], ["Qr1B"]); n("Cast", ["Qr1B"], ["Qr1"], to=1)
    n("MatMul", ["shapeup", "Qr1"], ["sur"]); n("MatMul", ["Qd1", "sur"], ["shp"])
    # region / border / interior masks
    n("Less", ["ck_iota_r", "H"], ["rinB"]); n("Cast", ["rinB"], ["rin"], to=1)
    n("Less", ["ck_iota_c", "W"], ["cinB"]); n("Cast", ["cinB"], ["cin"], to=1)
    n("Mul", ["rin", "cin"], ["inreg"])
    n("Equal", ["ck_iota_r", "n_zero"], ["r0B"]); n("Cast", ["r0B"], ["r0e"], to=1)
    n("Equal", ["ck_iota_r", "H1"], ["rHB"]); n("Cast", ["rHB"], ["rHe"], to=1)
    n("Equal", ["ck_iota_c", "n_zero"], ["c0B"]); n("Cast", ["c0B"], ["c0e"], to=1)
    n("Equal", ["ck_iota_c", "W1"], ["cWB"]); n("Cast", ["cWB"], ["cWe"], to=1)
    n("Max", ["r0e", "rHe"], ["redge"]); n("Max", ["c0e", "cWe"], ["cedge"])
    n("Max", ["redge", "cedge"], ["onedge"])                      # [1,1,30,30]
    n("Mul", ["inreg", "onedge"], ["border"])
    n("Sub", ["n_one", "onedge"], ["notedge"]); n("Mul", ["inreg", "notedge"], ["interior"])
    # assemble
    n("Mul", ["border", "n_oh2"], ["out_b"])
    n("Mul", ["shp", "cuesel"], ["out_s"])
    n("Sub", ["n_one", "shp"], ["notshp"]); n("Mul", ["interior", "notshp"], ["ibg"])
    n("Mul", ["ibg", "n_e0"], ["out_ibg"])
    n("Add", ["out_b", "out_s"], ["ab"]); n("Add", ["ab", "out_ibg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
