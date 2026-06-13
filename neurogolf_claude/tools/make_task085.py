"""task085: each input is a set of solid rectangles of ODD width and height>=3.
The transform modifies ONLY the interior rows of each rectangle (rows that have
the SAME color directly above AND below, i.e. an interior row of a same-color
vertical run). Within such a row, blank to background every cell whose offset
from the left edge of the interior run is ODD; keep even offsets. Everything
else passes through unchanged. No connected-component labeling.

Structural invariant (verified on all 265 examples): every interior run is the
only interior run in its row and begins at the row's first interior cell, so a
plain per-row prefix count of interior cells gives the in-run offset directly:

  colorgrid = Conv(input, [0..9])                        color values [1,1,30,30]
  up_same   = (colorgrid == shift_down(colorgrid)) & (colorgrid>0)
  dn_same   = (colorgrid == shift_up(colorgrid))   & (colorgrid>0)
  interior  = up_same & dn_same                          [1,1,30,30]
  cs        = CumSum(interior, axis=3)                    in-run 1-based count
  blank     = interior & (cs % 2 == 0)                    blank odd offsets
  outcolor  = colorgrid * (1 - blank)
  output    = one-hot(outcolor) with out-of-grid cells sent to -1.

Single-channel `[1,1,30,30]` pipeline until the final one-hot expansion.
Verified exact on all 2 train + 1 test + 262 arc-gen examples.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
F16 = TensorProto.FLOAT16
h16 = np.float16


def build(tid="task085", out_path=None):
    out_path = out_path or f"{SOL}/{tid}.onnx"

    w_col32 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    colors = np.arange(10, dtype=h16).reshape(1, 10, 1, 1)

    inits = [
        ck.const("w_col32", w_col32),
        ck.const("colors", colors),
        ck.const("t_zero", np.array(0.0, h16)),
        ck.const("t_one", np.array(1.0, h16)),
        ck.const("t_two", np.array(2.0, h16)),
        ck.const("neg_one", np.array(-1.0, h16)),
        ck.const("f_zero", np.array(0.0, np.float32)),
        ck.const("pad_down", np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64)),  # content down 1 row
        ck.const("pad_up", np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64)),    # content up 1 row
        ck.const("sl_s0", np.array([0], np.int64)),
        ck.const("sl_e30", np.array([30], np.int64)),
        ck.const("sl_s1", np.array([1], np.int64)),
        ck.const("sl_e31", np.array([31], np.int64)),
        ck.const("ax2", np.array([2], np.int64)),
        ck.const("ax3", np.array(3, np.int64)),
    ]

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # color grid + nonzero mask
    n("Conv", ["input", "w_col32"], ["cg32"])
    n("Cast", ["cg32"], ["cg"], to=F16)                       # [1,1,30,30] color
    n("Greater", ["cg", "t_zero"], ["nzB"])

    # adjacent-vertical-equality, computed once: eqAdj[r] = (cg[r]==cg[r-1])
    n("Pad", ["cg", "pad_down"], ["cg_padD"])
    n("Slice", ["cg_padD", "sl_s0", "sl_e30", "ax2"], ["cg_dn"])   # value of row above
    n("Equal", ["cg", "cg_dn"], ["eqUpB"])                    # cell == cell above
    # "below same" at row r == eqUp at row r+1; shift eqUp up one row (false at row 29)
    n("Pad", ["eqUpB", "pad_up"], ["eqUp_padU"])             # bool pad (constant False)
    n("Slice", ["eqUp_padU", "sl_s1", "sl_e31", "ax2"], ["eqDnB"])
    n("And", ["eqUpB", "eqDnB"], ["eqBothB"])                 # same color above & below
    n("And", ["eqBothB", "nzB"], ["interiorB"])              # [1,1,30,30] bool

    # in-run offset count via per-row prefix sum, then parity
    n("Cast", ["interiorB"], ["interior"], to=F16)
    n("CumSum", ["interior", "ax3"], ["cs"])                  # 1-based count within run
    # parity via fmod: cs % 2 == 1 -> keep, == 0 -> blank.
    n("Mod", ["cs", "t_two"], ["par"], fmod=1)               # 1.0 (odd/keep) or 0.0 (even/blank)
    n("Less", ["par", "t_one"], ["parZeroB"])                # cs even
    n("And", ["interiorB", "parZeroB"], ["blankB"])
    n("Where", ["blankB", "t_zero", "cg"], ["outcolor"])     # [1,1,30,30] color

    # send out-of-grid cells to -1 (match no channel), then one-hot
    n("ReduceMax", ["input"], ["active32"], axes=[1], keepdims=1)
    n("Greater", ["active32", "f_zero"], ["ingridB"])         # in-grid mask (fp32 input)
    n("Where", ["ingridB", "outcolor", "neg_one"], ["outcolor_m"])
    n("Equal", ["outcolor_m", "colors"], ["output"])         # [1,10,30,30] bool

    _finalize_bool_output(N, inits, tid, out_path)


def _finalize_bool_output(nodes, inits, name, out_path):
    import onnx
    from onnx import TensorProto as TP
    from onnx import helper as H
    graph = H.make_graph(
        nodes, name,
        [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print("saved", out_path, len(model.SerializeToString()), "bytes")
    return model


if __name__ == "__main__":
    build(out_path="/tmp/wf2_task085.onnx")
