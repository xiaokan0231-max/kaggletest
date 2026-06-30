"""Build compact ONNX for task306 (bool-golfed, separators baked into tile).

Rule: grid split by color-4 separators into equal 9x9 panels (layouts
19x9->2 panels, 19x19->4, 19x29->6). One panel has content; the rest are
all color-0. Output tiles the content panel into every panel position;
separators (color4) and padding are preserved.

Pipeline (all bool, 1-byte):
  xb        = cast(input) -> bool one-hot [1,10,30,30]
  Sp        = Or over the 6 panel blocks (10ch, ch0 polluted)
  S10       = Sp with ch0 rebuilt as NOT(any color 1..9)  [1,10,9,9]
  unit10x10 = S10 padded to 10x10, with row9/col9 set to color4 (a 10x10
              "panel+borders" tile)
  tiledU    = unit tiled by periodic Gather -> [1,10,19,30] (panels+seps)
  output    = tiledU AND valid_col, where valid_col = xb[ch4,row9,:] is the
              full-width separator marking valid columns (c < width).
Cropped to rows 0-18 (rows 19-29 always padding); Pad restores 30 rows as
the free bool `output` (verifier reads result>0).
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    nodes, inits = [], []

    def n(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))

    BOOL, U8 = TensorProto.BOOL, TensorProto.UINT8

    n("Cast", ["input"], ["xb"], to=BOOL)                  # [1,10,30,30] bool

    inits.append(const("ax1", np.array([1], np.int64)))
    inits.append(const("ax23", np.array([2, 3], np.int64)))
    inits.append(const("c1s", np.array([1], np.int64)))
    inits.append(const("c10e", np.array([10], np.int64)))

    # ---- merge 6 panel blocks (sliced straight from xb) ----
    starts = [(0, 0), (0, 10), (0, 20), (10, 0), (10, 10), (10, 20)]
    blk = []
    for i, (r0, c0) in enumerate(starts):
        inits.append(const(f"bs{i}", np.array([r0, c0], np.int64)))
        inits.append(const(f"be{i}", np.array([r0 + 9, c0 + 9], np.int64)))
        n("Slice", ["xb", f"bs{i}", f"be{i}", "ax23"], [f"blk{i}"])  # [1,10,9,9]
        blk.append(f"blk{i}")
    cur = blk[0]
    for i in range(1, 6):
        out = "Sp" if i == 5 else f"Sp{i}"
        n("Or", [cur, blk[i]], [out])
        cur = out                                          # Sp [1,10,9,9] bool

    # ch0 = NOT(any color 1..9)
    n("Slice", ["Sp", "c1s", "c10e", "ax1"], ["Sc"])       # [1,9,9,9] bool
    n("Cast", ["Sc"], ["Scu"], to=U8)
    n("ReduceMax", ["Scu"], ["anyScu"], axes=[1], keepdims=1)  # [1,1,9,9] u8
    n("Cast", ["anyScu"], ["anySc"], to=BOOL)
    n("Not", ["anySc"], ["Sc0"])
    n("Concat", ["Sc0", "Sc"], ["S10"], axis=1)            # [1,10,9,9] bool

    # ---- build 10x10 unit: append a color4 separator col then row ----
    sep_col = np.zeros((1, 10, 9, 1), np.bool_)
    sep_col[0, 4] = True                                   # color4 column
    inits.append(const("sep_col", sep_col))                # 90 params
    sep_row = np.zeros((1, 10, 1, 10), np.bool_)
    sep_row[0, 4] = True                                   # color4 row
    inits.append(const("sep_row", sep_row))                # 100 params
    n("Concat", ["S10", "sep_col"], ["unit_top"], axis=3)  # [1,10,9,10]
    n("Concat", ["unit_top", "sep_row"], ["unit"], axis=2)  # [1,10,10,10]

    # ---- tile unit (period 10) to rows 0..18, cols 0..28 ----
    # (col 29 is always padding; restored by the final Pad)
    ridx = list(range(9)) + [9] + list(range(9))           # 19, uses sep row 9
    cidx = (list(range(9)) + [9]) * 2 + list(range(9))     # 29, periodic
    inits.append(const("ridx", np.array(ridx, np.int64)))
    inits.append(const("cidx", np.array(cidx, np.int64)))
    n("Gather", ["unit", "ridx"], ["tH"], axis=2)          # [1,10,19,10]
    n("Gather", ["tH", "cidx"], ["tiledU"], axis=3)        # [1,10,19,29]

    # ---- valid columns: xb[ch4,row9,0:29] (full-width separator) ----
    # one Slice on axes [1,2,3]: channel 4, row 9, cols 0..28 -> [1,1,1,29]
    inits.append(const("vstart", np.array([4, 9, 0], np.int64)))
    inits.append(const("vend", np.array([5, 10, 29], np.int64)))
    inits.append(const("vaxes", np.array([1, 2, 3], np.int64)))
    n("Slice", ["xb", "vstart", "vend", "vaxes"], ["validc"])  # [1,1,1,29]
    n("And", ["tiledU", "validc"], ["res19"])              # [1,10,19,29] bool

    # pad back to 30 rows x 30 cols -> free bool output
    inits.append(const("pads", np.array([0, 0, 0, 0, 0, 0, 11, 1], np.int64)))
    n("Pad", ["res19", "pads"], ["output"], mode="constant")

    g = helper.make_graph(
        nodes, "task306",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b4_task306.onnx"
    onnx.save(build(), out)
    print("saved", out)
