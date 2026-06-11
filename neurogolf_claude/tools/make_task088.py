"""task088: input has 2 colors — a rare 'marker' at the 4 corners of a rectangle
and a 'shape' inside. Output = the rectangle INTERIOR (strictly inside the marker
corners), with the shape recolored to the marker color (background kept).

Select marker = rarest color; interior rows = rows with a marker row both above
and below; crop input to the interior; recolor any non-background cell to marker.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import colorkit as colk  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task088"):
    inits = ck.base_inits() + colk.color_inits() + [
        ck.const("b_one", np.float32(1.0)),
        ck.const("b_cax2", np.array(2, np.int64)), ck.const("b_cax3", np.array(3, np.int64)),
        ck.const("b_ax2", np.array([2], np.int64)), ck.const("b_ax3", np.array([3], np.int64)),
        ck.const("b_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("b_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    colk.emit_select_rarest(n, "selm")
    colk.emit_colormask(n, "selm", "mmask")                       # [1,1,30,30] marker cells
    n("ReduceMax", ["mmask"], ["mrow"], axes=[3], keepdims=1)     # [1,1,30,1]
    n("ReduceMax", ["mmask"], ["mcol"], axes=[2], keepdims=1)     # [1,1,1,30]

    def interior(band, cax, rax, out):
        # interior = rows/cols that have a marker line strictly above AND below
        n("CumSum", [band, cax], [f"{out}_ci"])
        n("Sub", [f"{out}_ci", band], [f"{out}_above0"]); n("Sign", [f"{out}_above0"], [f"{out}_above"])
        n("ReduceSum", [band, rax], [f"{out}_tot"], keepdims=1)
        n("Sub", [f"{out}_tot", f"{out}_ci"], [f"{out}_below0"]); n("Sign", [f"{out}_below0"], [f"{out}_below"])
        n("Mul", [f"{out}_above", f"{out}_below"], [out])
    interior("mrow", "b_cax2", "b_ax2", "irow")
    interior("mcol", "b_cax3", "b_ax3", "icol")
    ck.emit_compaction(n, "irow", "icol", "input", "crop")        # interior content compacted
    # recolor: any nonzero -> marker color; background kept
    n("Mul", ["crop", "b_nzch"], ["cropfg"])
    n("ReduceMax", ["cropfg"], ["nz"], axes=[1], keepdims=1)      # [1,1,30,30] nonzero presence
    n("ReduceMax", ["crop"], ["reg"], axes=[1], keepdims=1)       # any present (interior region)
    n("Mul", ["nz", "selm"], ["out_m"])                          # marker color where shape
    n("Sub", ["reg", "nz"], ["bgm"]); n("Mul", ["bgm", "b_e0"], ["out_bg"])
    n("Add", ["out_m", "out_bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
