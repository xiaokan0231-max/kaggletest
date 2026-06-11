"""task291: output (1x1) = the color of the object with the LOWEST fill ratio
(pixel_count / bbox_area) among the nonzero colors.

Per channel: pixel count, bbox-fill height/width -> area -> fill = count/area;
argmin over present nonzero channels (background excluded); emit at (0,0).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task291"):
    kill0 = np.zeros((1, 10, 1, 1), np.float32); kill0[0, 0, 0, 0] = BIG
    inits = ck.base_inits() + [
        ck.const("p_one", np.float32(1.0)), ck.const("p_zero", np.float32(0.0)), ck.const("p_big", np.float32(BIG)),
        ck.const("p_kill0", kill0),
        ck.const("p_ax23", np.array([2, 3], np.int64)), ck.const("p_ax2", np.array([2], np.int64)),
        ck.const("p_ax3", np.array([3], np.int64)), ck.const("p_ax1", np.array([1], np.int64)),
        ck.const("p_cax2", np.array(2, np.int64)), ck.const("p_cax3", np.array(3, np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    def fill(band, cax, rax, out):
        n("CumSum", [band, cax], [f"{out}_c"]); n("Sign", [f"{out}_c"], [f"{out}_hl"])
        n("ReduceSum", [band, rax], [f"{out}_t"], keepdims=1)
        n("Sub", [f"{out}_c", band], [f"{out}_e"])
        n("Sub", [f"{out}_t", f"{out}_e"], [f"{out}_d"]); n("Sign", [f"{out}_d"], [f"{out}_hr"])
        n("Mul", [f"{out}_hl", f"{out}_hr"], [out])
    n("ReduceSum", ["input", "p_ax23"], ["cnt"], keepdims=1)          # [1,10,1,1]
    n("Sign", ["cnt"], ["present"])
    n("ReduceMax", ["input"], ["krp"], axes=[3], keepdims=1)          # [1,10,30,1]
    n("ReduceMax", ["input"], ["kcp"], axes=[2], keepdims=1)          # [1,10,1,30]
    fill("krp", "p_cax2", "p_ax2", "krpf"); fill("kcp", "p_cax3", "p_ax3", "kcpf")
    n("ReduceSum", ["krpf", "p_ax2"], ["hc"], keepdims=1)
    n("ReduceSum", ["kcpf", "p_ax3"], ["wc"], keepdims=1)
    n("Mul", ["hc", "wc"], ["area"])
    n("Sub", ["p_one", "present"], ["absent"])
    n("Add", ["area", "absent"], ["areag"])                          # avoid div by zero
    n("Div", ["cnt", "areag"], ["fillr"])                            # [1,10,1,1]
    n("Mul", ["absent", "p_big"], ["pen_a"]); n("Add", ["fillr", "pen_a"], ["sc0"])
    n("Add", ["sc0", "p_kill0"], ["score"])                          # exclude absent + background
    n("ReduceMin", ["score"], ["minsc"], axes=[1], keepdims=1)
    n("Equal", ["score", "minsc"], ["selB"]); n("Cast", ["selB"], ["sel"], to=1)  # one-hot color
    n("Equal", ["ck_iota_r", "p_zero"], ["r0B"]); n("Cast", ["r0B"], ["r0"], to=1)
    n("Equal", ["ck_iota_c", "p_zero"], ["c0B"]); n("Cast", ["c0B"], ["c0"], to=1)
    n("Mul", ["r0", "c0"], ["p00"])
    n("Mul", ["sel", "p00"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
