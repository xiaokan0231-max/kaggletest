"""task253: 13x13 -> 4x4. Four distinct-colored L-pieces (3 px in a 2x2 bbox),
each placed at a 4x4 corner chosen by its fill orientation:
top if its top row is at least as filled as its bottom row, left likewise.

Per-color: extract channel, compact to a 2x2 at top-left, decide top/left from
row/col fill counts, shift to that corner, recolor; sum all colors; fill bg.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task253"):
    inits = ck.base_inits() + [
        ck.const("t_one", np.float32(1.0)), ck.const("t_zero", np.float32(0.0)),
        ck.const("t_two", np.float32(2.0)), ck.const("t_n2", np.float32(-2.0)),
        ck.const("t_four", np.float32(4.0)),
        ck.const("t_ax2", np.array([2], np.int64)), ck.const("t_ax3", np.array([3], np.int64)),
        ck.const("t_ax123", np.array([1, 2, 3], np.int64)),
        ck.const("t_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    for c in range(1, 10):
        w = np.zeros((1, 10, 1, 1), np.float32); w[0, c, 0, 0] = 1.0
        inits.append(ck.const(f"w{c}", w))
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    n("Equal", ["rmc", "t_n2"], ["QrB"]); n("Cast", ["QrB"], ["Qright"], to=1)  # cols right 2
    n("Equal", ["rmc", "t_two"], ["QdB"]); n("Cast", ["QdB"], ["Qdown"], to=1)  # rows down 2
    n("Equal", ["ck_iota_r", "t_zero"], ["er0B"]); n("Cast", ["er0B"], ["er0"], to=1)
    n("Equal", ["ck_iota_r", "t_one"], ["er1B"]); n("Cast", ["er1B"], ["er1"], to=1)
    n("Equal", ["ck_iota_c", "t_zero"], ["ec0B"]); n("Cast", ["ec0B"], ["ec0"], to=1)
    n("Equal", ["ck_iota_c", "t_one"], ["ec1B"]); n("Cast", ["ec1B"], ["ec1"], to=1)
    n("Less", ["ck_iota_r", "t_four"], ["r4B"]); n("Cast", ["r4B"], ["r4"], to=1)
    n("Less", ["ck_iota_c", "t_four"], ["c4B"]); n("Cast", ["c4B"], ["c4"], to=1)
    n("Mul", ["r4", "c4"], ["reg4"])
    outs = []
    for c in range(1, 10):
        p = f"k{c}"
        n("Conv", ["input", f"w{c}"], [f"{p}_mask"])
        n("ReduceMax", [f"{p}_mask"], [f"{p}_kr"], axes=[3], keepdims=1)
        n("ReduceMax", [f"{p}_mask"], [f"{p}_kc"], axes=[2], keepdims=1)
        ck.emit_compaction(n, f"{p}_kr", f"{p}_kc", f"{p}_mask", f"{p}_pc")
        n("ReduceSum", [f"{p}_pc", "t_ax3"], [f"{p}_rc"], keepdims=1)   # [1,1,30,1]
        n("ReduceSum", [f"{p}_pc", "t_ax2"], [f"{p}_cc"], keepdims=1)   # [1,1,1,30]
        n("Mul", [f"{p}_rc", "er0"], [f"{p}_r0m"]); n("ReduceSum", [f"{p}_r0m", "t_ax123"], [f"{p}_r0"], keepdims=1)
        n("Mul", [f"{p}_rc", "er1"], [f"{p}_r1m"]); n("ReduceSum", [f"{p}_r1m", "t_ax123"], [f"{p}_r1"], keepdims=1)
        n("Mul", [f"{p}_cc", "ec0"], [f"{p}_c0m"]); n("ReduceSum", [f"{p}_c0m", "t_ax123"], [f"{p}_c0"], keepdims=1)
        n("Mul", [f"{p}_cc", "ec1"], [f"{p}_c1m"]); n("ReduceSum", [f"{p}_c1m", "t_ax123"], [f"{p}_c1"], keepdims=1)
        n("GreaterOrEqual", [f"{p}_r0", f"{p}_r1"], [f"{p}_topB"]); n("Cast", [f"{p}_topB"], [f"{p}_top"], to=1)
        n("GreaterOrEqual", [f"{p}_c0", f"{p}_c1"], [f"{p}_lefB"]); n("Cast", [f"{p}_lefB"], [f"{p}_lef"], to=1)
        # horizontal placement
        n("MatMul", [f"{p}_pc", "Qright"], [f"{p}_pr"])
        n("Sub", ["t_one", f"{p}_lef"], [f"{p}_nlef"])
        n("Mul", [f"{p}_lef", f"{p}_pc"], [f"{p}_ha"]); n("Mul", [f"{p}_nlef", f"{p}_pr"], [f"{p}_hb"])
        n("Add", [f"{p}_ha", f"{p}_hb"], [f"{p}_hor"])
        # vertical placement
        n("MatMul", ["Qdown", f"{p}_hor"], [f"{p}_hd"])
        n("Sub", ["t_one", f"{p}_top"], [f"{p}_ntop"])
        n("Mul", [f"{p}_top", f"{p}_hor"], [f"{p}_va"]); n("Mul", [f"{p}_ntop", f"{p}_hd"], [f"{p}_vb"])
        n("Add", [f"{p}_va", f"{p}_vb"], [f"{p}_ver"])
        n("Mul", [f"{p}_ver", f"w{c}"], [f"{p}_out"])     # recolor -> [1,10,30,30]
        outs.append(f"{p}_out")
    acc = outs[0]
    for i, o in enumerate(outs[1:], 1):
        nm = f"fgacc{i}"; n("Add", [acc, o], [nm]); acc = nm
    n("ReduceMax", [acc], ["fgres"], axes=[1], keepdims=1)
    n("Sub", ["t_one", "fgres"], ["nofg"]); n("Mul", ["reg4", "nofg"], ["bgm"])
    n("Mul", ["bgm", "t_e0"], ["bg"])
    n("Add", [acc, "bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
