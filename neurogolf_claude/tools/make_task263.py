"""task263: a strip of K 3x3 tiles (vertical or horizontal); all but one share a
binary shape (differing only in color). Output = the unique-shape tile.

Normalize to a vertical strip (conditional transpose), align up to 5 tiles to the
top-left, score each by how many other non-empty tiles share its shape, output
the non-empty tile that matches none, then transpose back if needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
NT = 5  # max tiles


def build(tid="task263"):
    inits = ck.base_inits() + [
        ck.const("z_one", np.float32(1.0)),
        ck.const("z_three", np.float32(3.0)),
        ck.const("z_ax123", np.array([1, 2, 3], np.int64)),
        ck.const("z_wnz", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    for k in range(NT):
        inits.append(ck.const(f"z_sh{k}", np.float32(-3.0 * k)))   # shift up 3k
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    # orientation: vertical if H>W
    n("Greater", ["Hdim", "Wdim"], ["vB"]); n("Cast", ["vB"], ["isv"], to=1)
    n("Sub", ["z_one", "isv"], ["ish"])
    n("Transpose", ["input"], ["inputT"], perm=[0, 1, 3, 2])
    n("Mul", ["isv", "input"], ["cv"]); n("Mul", ["ish", "inputT"], ["ch"])
    n("Add", ["cv", "ch"], ["combo"])                       # vertical strip
    # TL 3x3 mask
    n("Less", ["ck_iota_r", "z_three"], ["lr3B"]); n("Cast", ["lr3B"], ["lr3"], to=1)
    n("Less", ["ck_iota_c", "z_three"], ["lc3B"]); n("Cast", ["lc3B"], ["lc3"], to=1)
    n("Mul", ["lr3", "lc3"], ["tl3"])
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    # aligned tiles A0..A4 and their nonzero shapes
    for k in range(NT):
        if k == 0:
            n("Mul", ["combo", "tl3"], ["A0"])
        else:
            n("Equal", ["rmc", f"z_sh{k}"], [f"Qu{k}B"]); n("Cast", [f"Qu{k}B"], [f"Qu{k}"], to=1)
            n("MatMul", [f"Qu{k}", "combo"], [f"sh{k}"]); n("Mul", [f"sh{k}", "tl3"], [f"A{k}"])
        n("Conv", [f"A{k}", "z_wnz"], [f"S{k}"])           # [1,1,30,30] nonzero shape
        n("ReduceSum", [f"S{k}", "z_ax123"], [f"cnt{k}"], keepdims=1)
        n("Sign", [f"cnt{k}"], [f"ne{k}"])                 # nonempty flag
    # pairwise shape-match same_ij = 1 - sign(sum|Si-Sj|)
    same = {}
    for i in range(NT):
        for j in range(i + 1, NT):
            n("Sub", [f"S{i}", f"S{j}"], [f"df{i}{j}"]); n("Abs", [f"df{i}{j}"], [f"da{i}{j}"])
            n("ReduceSum", [f"da{i}{j}", "z_ax123"], [f"ds{i}{j}"], keepdims=1)
            n("Sign", [f"ds{i}{j}"], [f"sg{i}{j}"]); n("Sub", ["z_one", f"sg{i}{j}"], [f"sm{i}{j}"])
            same[(i, j)] = f"sm{i}{j}"; same[(j, i)] = f"sm{i}{j}"
    # match_k = sum_{j!=k} same_kj  -> odd_k = ne_k * (1 - sign(match_k))
    outs = []
    for k in range(NT):
        terms = [same[(k, j)] for j in range(NT) if j != k]
        acc = terms[0]
        for t, term in enumerate(terms[1:], 1):
            o = f"m{k}_{t}"; n("Add", [acc, term], [o]); acc = o
        n("Sign", [acc], [f"msg{k}"]); n("Sub", ["z_one", f"msg{k}"], [f"uniq{k}"])
        n("Mul", ["ne" + str(k), f"uniq{k}"], [f"odd{k}"])
        n("Mul", [f"odd{k}", f"A{k}"], [f"o{k}"]); outs.append(f"o{k}")
    acc = outs[0]
    for t, o in enumerate(outs[1:], 1):
        nm = f"oc{t}"; n("Add", [acc, o], [nm]); acc = nm
    n("Identity", [acc], ["outcombo"])
    # transpose back if horizontal
    n("Transpose", ["outcombo"], ["outcomboT"], perm=[0, 1, 3, 2])
    n("Mul", ["isv", "outcombo"], ["fv"]); n("Mul", ["ish", "outcomboT"], ["fh"])
    n("Add", ["fv", "fh"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
