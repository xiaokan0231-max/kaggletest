"""task174: objects are single-colored; exactly one has a left-right symmetric
shape. Output = that object's bounding-box crop.

For each color c: extract its mask, compact to top-left, flip horizontally
within its width, and test equality (symmetry). Build a one-hot selector of the
symmetric color, then crop the input to that color's bounding box.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task174"):
    inits = ck.base_inits() + [
        ck.const("u_one", np.float32(1.0)),
        ck.const("u_ax3", np.array([3], np.int64)),
        ck.const("u_ax123", np.array([1, 2, 3], np.int64)),
    ]
    # per-color conv weights (select channel c) and channel one-hots
    for c in range(1, 10):
        w = np.zeros((1, 10, 1, 1), np.float32); w[0, c, 0, 0] = 1.0
        inits.append(ck.const(f"sel_w{c}", w))
        oh = np.zeros((1, 10, 1, 1), np.float32); oh[0, c, 0, 0] = 1.0
        inits.append(ck.const(f"oh{c}", oh))
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Add", ["ck_iota_r", "ck_iota_c"], ["rpc"])           # [1,1,30,30] r+c
    contribs = []
    for c in range(1, 10):
        p = f"c{c}"
        n("Conv", ["input", f"sel_w{c}"], [f"{p}_mask"])               # [1,1,30,30]
        n("ReduceMax", [f"{p}_mask"], [f"{p}_kr"], axes=[3], keepdims=1)
        n("ReduceMax", [f"{p}_mask"], [f"{p}_kc"], axes=[2], keepdims=1)
        ck.emit_compaction(n, f"{p}_kr", f"{p}_kc", f"{p}_mask", f"{p}_crop")
        n("ReduceSum", [f"{p}_kc", "u_ax3"], [f"{p}_w"], keepdims=1)
        n("Sub", [f"{p}_w", "u_one"], [f"{p}_wm1"])
        n("Equal", ["rpc", f"{p}_wm1"], [f"{p}_revB"]); n("Cast", [f"{p}_revB"], [f"{p}_rev"], to=1)
        n("MatMul", [f"{p}_crop", f"{p}_rev"], [f"{p}_flip"])
        n("Sub", [f"{p}_crop", f"{p}_flip"], [f"{p}_d"]); n("Abs", [f"{p}_d"], [f"{p}_da"])
        n("ReduceSum", [f"{p}_da", "u_ax123"], [f"{p}_def"], keepdims=1)
        n("Sign", [f"{p}_def"], [f"{p}_ds"]); n("Sub", ["u_one", f"{p}_ds"], [f"{p}_symsh"])
        n("ReduceSum", [f"{p}_mask", "u_ax123"], [f"{p}_cnt"], keepdims=1)
        n("Sign", [f"{p}_cnt"], [f"{p}_pres"])
        n("Mul", [f"{p}_symsh", f"{p}_pres"], [f"{p}_sym"])            # scalar
        n("Mul", [f"{p}_sym", f"oh{c}"], [f"{p}_contrib"])            # [1,10,1,1]
        contribs.append(f"{p}_contrib")
    # sum contributions -> one-hot color selector
    acc = contribs[0]
    for i, cc in enumerate(contribs[1:], 1):
        out = "sel" if i == len(contribs) - 1 else f"acc{i}"
        n("Add", [acc, cc], [out]); acc = out
    # bbox crop of the selected color
    n("Conv", ["input", "sel"], ["cmask"])
    n("ReduceMax", ["cmask"], ["keeprow"], axes=[3], keepdims=1)
    n("ReduceMax", ["cmask"], ["keepcol"], axes=[2], keepdims=1)
    ck.emit_compaction(n, "keeprow", "keepcol", "input", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
