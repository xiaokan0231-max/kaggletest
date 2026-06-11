"""task355: the grid is tiled by solid rectangular color blocks (one may be
background 0), each peppered with sparse single-pixel 'noise' of one shared
color. Output (1x1) = the color of the block holding the MOST noise pixels;
output 0 if the background block wins or there is a tie.

noise = rarest color; per channel c compute its bbox rectangle and count noise
pixels inside it; pick the channel with the strict unique max; emit that color
at (0,0) unless it is channel 0 or a tie (then 0).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import colorkit as colk  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task355"):
    inits = ck.base_inits() + colk.color_inits() + [
        ck.const("s_one", np.float32(1.0)), ck.const("s_zero", np.float32(0.0)),
        ck.const("s_big", np.float32(BIG)),
        ck.const("s_ax23", np.array([2, 3], np.int64)), ck.const("s_ax1", np.array([1], np.int64)),
        ck.const("s_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("s_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    colk.emit_select_rarest(n, "selnoise")
    colk.emit_colormask(n, "selnoise", "noisemask")              # [1,1,30,30]
    n("ReduceMax", ["input"], ["krp"], axes=[3], keepdims=1)     # [1,10,30,1]
    n("ReduceMax", ["input"], ["kcp"], axes=[2], keepdims=1)     # [1,10,1,30]
    n("Mul", ["krp", "kcp"], ["region"])                        # [1,10,30,30] per-channel bbox
    n("Mul", ["region", "noisemask"], ["nmreg"])
    n("ReduceSum", ["nmreg", "s_ax23"], ["score"], keepdims=1)   # [1,10,1,1] noise count per block
    n("Mul", ["selnoise", "s_big"], ["pen"]); n("Sub", ["score", "pen"], ["score2"])
    n("ReduceMax", ["score2"], ["maxsc"], axes=[1], keepdims=1)  # [1,1,1,1]
    n("Equal", ["score2", "maxsc"], ["ismaxB"]); n("Cast", ["ismaxB"], ["ismax"], to=1)
    n("ReduceSum", ["ismax", "s_ax1"], ["countmax"], keepdims=1)  # [1,1,1,1]
    n("Equal", ["countmax", "s_one"], ["uniqB"]); n("Cast", ["uniqB"], ["uniq"], to=1)
    n("Mul", ["ismax", "uniq"], ["winner"])                     # [1,10,1,1]
    n("Mul", ["winner", "s_nzch"], ["fgwin"])                   # zero background channel
    n("ReduceSum", ["fgwin", "s_ax1"], ["hasfg"], keepdims=1)    # [1,1,1,1]
    n("Sub", ["s_one", "hasfg"], ["nofg"]); n("Mul", ["nofg", "s_e0"], ["bgwin"])
    n("Add", ["fgwin", "bgwin"], ["wvec"])                      # final color vector
    # place at cell (0,0)
    n("Equal", ["ck_iota_r", "s_zero"], ["p0rB"]); n("Cast", ["p0rB"], ["p0r"], to=1)
    n("Equal", ["ck_iota_c", "s_zero"], ["p0cB"]); n("Cast", ["p0cB"], ["p0c"], to=1)
    n("Mul", ["p0r", "p0c"], ["point00"])
    n("Mul", ["wvec", "point00"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
