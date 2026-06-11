"""task384: crop the nonzero bbox, then upscale 2x (each cell -> 2x2 block).

up[i,j] = cropped[i//2, j//2], done as E @ cropped @ E^T with a fixed
expansion matrix E[i,k] = 1 iff k == i//2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task384"):
    E = np.zeros((1, 1, 30, 30), np.float32)
    for i in range(30):
        if i // 2 < 30:
            E[0, 0, i, i // 2] = 1.0
    inits = ck.base_inits() + [ck.const("up_E", E)]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    ck.emit_compaction(n, "nz_row", "nz_col", "input", "cropped")
    n("Transpose", ["up_E"], ["up_ET"], perm=[0, 1, 3, 2])
    n("MatMul", ["up_E", "cropped"], ["up1"])
    n("MatMul", ["up1", "up_ET"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
