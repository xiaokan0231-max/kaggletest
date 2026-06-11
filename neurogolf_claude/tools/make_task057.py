"""task057: crop the nonzero bbox, tile it 2x horizontally (out = [shape | shape]).

out[:, j] = cropped[:, j mod w], built as cropped + (cropped shifted right by w),
where w = nonzero bbox width.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task057"):
    inits = ck.base_inits() + [ck.const("f_ax3", np.array([3], np.int64))]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    ck.emit_compaction(n, "nz_row", "nz_col", "input", "cropped")
    n("ReduceSum", ["nz_col", "f_ax3"], ["Wc"], keepdims=1)        # bbox width [1,1,1,1]
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    n("Neg", ["Wc"], ["nWc"])
    n("Equal", ["rmc", "nWc"], ["QrB"]); n("Cast", ["QrB"], ["Qright"], to=1)
    n("MatMul", ["cropped", "Qright"], ["shifted"])               # shift right by Wc
    n("Add", ["cropped", "shifted"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
