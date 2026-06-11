"""task100: output = a solid N x N block of the color with the largest bbox area,
where N = the number of distinct nonzero colors. (Background dropped: -[0].)

Per channel: bbox area; argmax over present nonzero channels -> color; count
present nonzero channels -> N; fill the N x N top-left region with that color.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task100"):
    kill0 = np.zeros((1, 10, 1, 1), np.float32); kill0[0, 0, 0, 0] = BIG
    inits = ck.base_inits() + [
        ck.const("h_one", np.float32(1.0)), ck.const("h_big", np.float32(BIG)),
        ck.const("h_kill0", kill0),
        ck.const("h_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("h_ax23", np.array([2, 3], np.int64)), ck.const("h_ax2", np.array([2], np.int64)),
        ck.const("h_ax3", np.array([3], np.int64)), ck.const("h_ax1", np.array([1], np.int64)),
        ck.const("h_cax2", np.array(2, np.int64)), ck.const("h_cax3", np.array(3, np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    def fill(band, cax, rax, out):
        n("CumSum", [band, cax], [f"{out}_c"]); n("Sign", [f"{out}_c"], [f"{out}_hl"])
        n("ReduceSum", [band, rax], [f"{out}_t"], keepdims=1)
        n("Sub", [f"{out}_c", band], [f"{out}_e"])
        n("Sub", [f"{out}_t", f"{out}_e"], [f"{out}_d"]); n("Sign", [f"{out}_d"], [f"{out}_hr"])
        n("Mul", [f"{out}_hl", f"{out}_hr"], [out])
    n("ReduceSum", ["input", "h_ax23"], ["cnt"], keepdims=1)
    n("Sign", ["cnt"], ["present"])
    n("ReduceMax", ["input"], ["krp"], axes=[3], keepdims=1)
    n("ReduceMax", ["input"], ["kcp"], axes=[2], keepdims=1)
    fill("krp", "h_cax2", "h_ax2", "krpf"); fill("kcp", "h_cax3", "h_ax3", "kcpf")
    n("ReduceSum", ["krpf", "h_ax2"], ["hc"], keepdims=1)
    n("ReduceSum", ["kcpf", "h_ax3"], ["wc"], keepdims=1)
    n("Mul", ["hc", "wc"], ["area"])
    # argmax area over present nonzero (exclude absent + background)
    n("Sub", ["h_one", "present"], ["absent"]); n("Mul", ["absent", "h_big"], ["pa"])
    n("Sub", ["area", "pa"], ["sc0"]); n("Sub", ["sc0", "h_kill0"], ["score"])
    n("ReduceMax", ["score"], ["mx"], axes=[1], keepdims=1)
    n("Equal", ["score", "mx"], ["selB"]); n("Cast", ["selB"], ["sel"], to=1)
    # N = number of present nonzero colors
    n("Mul", ["present", "h_nzch"], ["pnz"]); n("ReduceSum", ["pnz", "h_ax1"], ["Ncount"], keepdims=1)
    # fill N x N region with the color
    n("Less", ["ck_iota_r", "Ncount"], ["rB"]); n("Cast", ["rB"], ["rm"], to=1)
    n("Less", ["ck_iota_c", "Ncount"], ["cB"]); n("Cast", ["cB"], ["cm"], to=1)
    n("Mul", ["rm", "cm"], ["region"])
    n("Mul", ["region", "sel"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
