"""Build a crop-to-bounding-box-of-color solution.

Usage: python make_bbox.py task310 rarest
       python make_bbox.py task300 common
Output = the input cropped to the bounding box of the selected color.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import colorkit as colk  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid, mode):
    inits = ck.base_inits() + colk.color_inits()
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    if mode == "rarest":
        colk.emit_select_rarest(n)
    elif mode == "common":
        colk.emit_select_common(n)
    else:
        raise SystemExit("mode must be rarest|common")
    colk.emit_colormask(n, "sel", "cmask")                      # [1,1,30,30]
    n("ReduceMax", ["cmask"], ["keeprow"], axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["cmask"], ["keepcol"], axes=[2], keepdims=1)  # [1,1,1,30]
    ck.emit_compaction(n, "keeprow", "keepcol", "input", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
