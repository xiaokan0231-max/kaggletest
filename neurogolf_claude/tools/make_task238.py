"""task238: a 4-color box frame + a separate color-8 shape. Output = the box with
its interior replaced by the 8-shape, recoloring each 8 to the unique nearest box
edge (interior distances top=ir, bottom=IH-1-ir, left=ic, right=IW-1-ic), kept 8
on ties.

Box = input with channel 8 removed, cropped to bbox. Edge colors via argmax over
each border line. Distances are positional fields; per-cell nearest edge picks
the recolor; the 8-shape is placed in the interior.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task238"):
    oh8 = np.zeros((1, 10, 1, 1), np.float32); oh8[0, 8, 0, 0] = 1.0
    notch8 = np.ones((1, 10, 1, 1), np.float32); notch8[0, 8, 0, 0] = 0.0
    kill0 = np.zeros((1, 10, 1, 1), np.float32); kill0[0, 0, 0, 0] = BIG
    inits = ck.base_inits() + [
        ck.const("d_oh8", oh8), ck.const("d_notch8", notch8), ck.const("d_kill0", kill0),
        ck.const("d_one", np.float32(1.0)), ck.const("d_two", np.float32(2.0)),
        ck.const("d_zero", np.float32(0.0)), ck.const("d_n1", np.float32(-1.0)),
        ck.const("d_ax23", np.array([2, 3], np.int64)),
        ck.const("d_ax2", np.array([2], np.int64)), ck.const("d_ax3", np.array([3], np.int64)),
        ck.const("d_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("d_wnz", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])
    # box = frame (channel 8 removed) cropped to bbox
    n("Mul", ["input", "d_notch8"], ["frame"])
    n("Conv", ["frame", "d_wnz"], ["fpres"])                    # nonzero presence (exclude bg)
    n("ReduceMax", ["fpres"], ["kr"], axes=[3], keepdims=1)
    n("ReduceMax", ["fpres"], ["kc"], axes=[2], keepdims=1)
    ck.emit_compaction(n, "kr", "kc", "frame", "box")
    n("ReduceSum", ["kr", "d_ax2"], ["H"], keepdims=1)
    n("ReduceSum", ["kc", "d_ax3"], ["W"], keepdims=1)
    n("Sub", ["H", "d_one"], ["H1"]); n("Sub", ["W", "d_one"], ["W1"])
    n("Sub", ["H", "d_two"], ["Hm2"]); n("Sub", ["W", "d_two"], ["Wm2"])

    def edge(linemask, out):
        n("Mul", ["box", linemask], [f"{out}_e"])
        n("ReduceSum", [f"{out}_e", "d_ax23"], [f"{out}_c"], keepdims=1)
        n("Sub", [f"{out}_c", "d_kill0"], [f"{out}_s"])
        n("ReduceMax", [f"{out}_s"], [f"{out}_m"], axes=[1], keepdims=1)
        n("Equal", [f"{out}_s", f"{out}_m"], [f"{out}_B"]); n("Cast", [f"{out}_B"], [out], to=1)
    n("Equal", ["ck_iota_r", "d_zero"], ["r0B"]); n("Cast", ["r0B"], ["r0m"], to=1)
    n("Equal", ["ck_iota_r", "H1"], ["rHB"]); n("Cast", ["rHB"], ["rHm"], to=1)
    n("Equal", ["ck_iota_c", "d_zero"], ["c0B"]); n("Cast", ["c0B"], ["c0m"], to=1)
    n("Equal", ["ck_iota_c", "W1"], ["cWB"]); n("Cast", ["cWB"], ["cWm"], to=1)
    edge("r0m", "topc"); edge("rHm", "botc"); edge("c0m", "leftc"); edge("cWm", "rightc")
    # 8-shape placed in interior
    n("Conv", ["input", "d_oh8"], ["m8"])
    n("ReduceMax", ["m8"], ["s8r"], axes=[3], keepdims=1)
    n("ReduceMax", ["m8"], ["s8c"], axes=[2], keepdims=1)
    ck.emit_compaction(n, "s8r", "s8c", "m8", "shp0")
    n("Equal", ["rmc", "d_one"], ["Qd1B"]); n("Cast", ["Qd1B"], ["Qd1"], to=1)
    n("Equal", ["rmc", "d_n1"], ["Qr1B"]); n("Cast", ["Qr1B"], ["Qr1"], to=1)
    n("MatMul", ["shp0", "Qr1"], ["shr"]); n("MatMul", ["Qd1", "shr"], ["shape8"])
    # distances (box coords r,c): dtop=r-1, dbot=H-2-r, dleft=c-1, dright=W-2-c
    n("Sub", ["ck_iota_r", "d_one"], ["dtop"])                  # [1,1,30,1]
    n("Sub", ["Hm2", "ck_iota_r"], ["dbot"])
    n("Sub", ["ck_iota_c", "d_one"], ["dleft"])                 # [1,1,1,30]
    n("Sub", ["Wm2", "ck_iota_c"], ["dright"])
    n("Min", ["dtop", "dbot"], ["dv"]); n("Min", ["dleft", "dright"], ["dh"])
    n("Min", ["dv", "dh"], ["mindist"])                        # [1,1,30,30]
    n("Equal", ["dtop", "mindist"], ["wtB"]); n("Cast", ["wtB"], ["wt"], to=1)
    n("Equal", ["dbot", "mindist"], ["wbB"]); n("Cast", ["wbB"], ["wb"], to=1)
    n("Equal", ["dleft", "mindist"], ["wlB"]); n("Cast", ["wlB"], ["wl"], to=1)
    n("Equal", ["dright", "mindist"], ["wrB"]); n("Cast", ["wrB"], ["wr"], to=1)
    n("Add", ["wt", "wb"], ["nw0"]); n("Add", ["nw0", "wl"], ["nw1"]); n("Add", ["nw1", "wr"], ["nwin"])
    n("Equal", ["nwin", "d_one"], ["uniqB"]); n("Cast", ["uniqB"], ["uniq"], to=1)
    # per-cell winning edge color (only meaningful where unique)
    n("Mul", ["wt", "topc"], ["ct"]); n("Mul", ["wb", "botc"], ["cb"])
    n("Mul", ["wl", "leftc"], ["cl"]); n("Mul", ["wr", "rightc"], ["cr"])
    n("Add", ["ct", "cb"], ["cc0"]); n("Add", ["cc0", "cl"], ["cc1"]); n("Add", ["cc1", "cr"], ["wincol"])
    n("Mul", ["uniq", "wincol"], ["ucol"])
    n("Sub", ["d_one", "uniq"], ["nu"]); n("Mul", ["nu", "d_oh8"], ["tiecol"])
    n("Add", ["ucol", "tiecol"], ["cellvec"])                 # [1,10,30,30]
    n("Mul", ["shape8", "cellvec"], ["recolored"])           # 8-cells -> nearest-edge color
    # output = box with its 8-cell positions replaced by the recolored shape
    n("Sub", ["d_one", "shape8"], ["notsh"]); n("Mul", ["box", "notsh"], ["boxmasked"])
    n("Add", ["boxmasked", "recolored"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
