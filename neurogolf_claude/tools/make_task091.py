"""task091: a box with vertical edges in color 5 and corner markers in color 8.
Output = crop of the input to the color-5 bounding box, row range extended by 1
above and below (to include the 8 corners), columns = full color-5 col range.

keep_col = filled column range of color 5 (between min and max col);
keep_row = filled row range of color 5, dilated by 1; content = input.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task091"):
    w5 = np.zeros((1, 10, 1, 1), np.float32); w5[0, 5, 0, 0] = 1.0
    inits = ck.base_inits() + [
        ck.const("g_w5", w5), ck.const("g_one", np.float32(1.0)), ck.const("g_n1", np.float32(-1.0)),
        ck.const("g_cax2", np.array(2, np.int64)), ck.const("g_cax3", np.array(3, np.int64)),
        ck.const("g_ax2", np.array([2], np.int64)), ck.const("g_ax3", np.array([3], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    n("Equal", ["rmc", "g_one"], ["QdB"]); n("Cast", ["QdB"], ["Qdown"], to=1)
    n("Equal", ["rmc", "g_n1"], ["QuB"]); n("Cast", ["QuB"], ["Qup"], to=1)
    n("Conv", ["input", "g_w5"], ["m5"])
    n("ReduceMax", ["m5"], ["row5"], axes=[3], keepdims=1)        # [1,1,30,1]
    n("ReduceMax", ["m5"], ["col5"], axes=[2], keepdims=1)        # [1,1,1,30]

    def fill(band, cax, rax, out):
        n("CumSum", [band, cax], [f"{out}_cum"]); n("Sign", [f"{out}_cum"], [f"{out}_hl"])
        n("ReduceSum", [band, rax], [f"{out}_tot"], keepdims=1)
        n("Sub", [f"{out}_cum", band], [f"{out}_cex"])
        n("Sub", [f"{out}_tot", f"{out}_cex"], [f"{out}_rr0"]); n("Sign", [f"{out}_rr0"], [f"{out}_hr"])
        n("Mul", [f"{out}_hl", f"{out}_hr"], [out])
    fill("row5", "g_cax2", "g_ax2", "rowrange")
    fill("col5", "g_cax3", "g_ax3", "keepcol")
    n("MatMul", ["Qdown", "rowrange"], ["rdn"]); n("MatMul", ["Qup", "rowrange"], ["rup"])
    n("Max", ["rowrange", "rdn", "rup"], ["keeprow"])
    ck.emit_compaction(n, "keeprow", "keepcol", "input", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
