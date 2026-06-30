"""task036 v2: single-plane pipeline (no [1,10,30,30] AvgPool).

Rule: shape color = color of the cell with the max same-color 8-neighbor count.
Output = bbox crop of that color's cells.

Pipeline (everything on [1,1,30,30] planes):
1. grid = Conv(input, [0..9])            -> [1,1,30,30] colored grid
2. gpad = Pad(grid, value=-1) by 1       -> [1,1,32,32]
3. for each of 8 dirs: eq = (slice(gpad,dir)==grid) & (grid>0); nc = sum eq
4. peak = (nc == ReduceMax(nc)); cval = ReduceMax(grid*peak)  (scalar color)
5. M = (grid == cval)                     -> [1,1,30,30] shape mask
6. compaction (cropkit) crops M's bbox to top-left -> M_comp
7. output channel cval = M_comp, channel 0 = bbox minus M_comp, built via Conv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/tools")
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

OUT = "/tmp/b4_task036.onnx"
F = TensorProto.FLOAT

# 8 neighbor directions, expressed as Slice starts on the 32x32 padded grid.
# padded index for cell (r,c) center is (r+1,c+1). neighbor (r+di,c+dj) is at
# pad index (r+1+di, c+1+dj). Slicing starts at (1+di, 1+dj), size 30x30.
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def build():
    cvec = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    inits = ck.base_inits() + [
        ck.const("cvec", cvec),
        ck.const("zero_f", np.float32(0.0)),
        ck.const("one_f", np.float32(1.0)),
        ck.const("padval", np.float32(-1.0)),
        ck.const("pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)),
        ck.const("ax23", np.array([2, 3], np.int64)),
    ]
    # Slice constants per direction
    for k, (di, dj) in enumerate(DIRS):
        inits.append(ck.const(f"st{k}", np.array([1 + di, 1 + dj], np.int64)))
        inits.append(ck.const(f"en{k}", np.array([31 + di, 31 + dj], np.int64)))
    inits.append(ck.const("slax", np.array([2, 3], np.int64)))

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # 1. colored grid
    n("Conv", ["input", "cvec"], ["grid"])                       # [1,1,30,30]
    # nonzero mask of grid
    n("Greater", ["grid", "zero_f"], ["nzB"])
    n("Cast", ["nzB"], ["nz"], to=F)                             # [1,1,30,30] 1 where nonzero
    # 2. padded grid with sentinel -1
    n("Pad", ["grid", "pads", "padval"], ["gpad"])               # [1,1,32,32]
    # 3. neighbor equality counts
    eqs = []
    for k in range(8):
        n("Slice", ["gpad", f"st{k}", f"en{k}", "slax"], [f"nb{k}"])  # [1,1,30,30]
        n("Equal", [f"nb{k}", "grid"], [f"eqB{k}"])
        n("Cast", [f"eqB{k}"], [f"eq{k}"], to=F)
        eqs.append(f"eq{k}")
    # sum equalities, then mask by nonzero
    acc = eqs[0]
    for i, e in enumerate(eqs[1:], 1):
        out = f"esum{i}"
        n("Add", [acc, e], [out]); acc = out
    n("Mul", [acc, "nz"], ["nc"])                                # [1,1,30,30] same-color-neighbor count
    # 4. peak & shape color
    n("ReduceMax", ["nc"], ["gmax"], axes=[2, 3], keepdims=1)     # [1,1,1,1]
    n("Equal", ["nc", "gmax"], ["peakB"])
    n("Cast", ["peakB"], ["peak"], to=F)
    n("Mul", ["grid", "peak"], ["gpeak"])
    n("ReduceMax", ["gpeak"], ["cval"], axes=[2, 3], keepdims=1)  # [1,1,1,1] shape color
    # 5. shape mask
    n("Equal", ["grid", "cval"], ["MB"])
    n("Cast", ["MB"], ["M"], to=F)                               # [1,1,30,30]
    # 6. compaction
    n("ReduceMax", ["M"], ["keep_r"], axes=[3], keepdims=1)       # [1,1,30,1]
    n("ReduceMax", ["M"], ["keep_c"], axes=[2], keepdims=1)       # [1,1,1,30]
    ck.emit_compaction(n, "keep_r", "keep_c", "M", "M_comp")      # [1,1,30,30]
    # output-grid mask = outer(rowmax, colmax) of M_comp
    n("ReduceMax", ["M_comp"], ["rmax"], axes=[3], keepdims=1)
    n("ReduceMax", ["M_comp"], ["cmax"], axes=[2], keepdims=1)
    n("Mul", ["rmax", "cmax"], ["G_out"])
    n("Sub", ["G_out", "M_comp"], ["bg"])                        # background within bbox
    # 7. assemble one-hot output: channel0=bg, channel cval=M_comp
    # selector one-hot over 10 channels: sel = (iota10 == cval)
    inits.append(ck.const("iota10", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    n("Equal", ["iota10", "cval"], ["selB"])                     # [1,10,1,1]
    n("Cast", ["selB"], ["selw_pre"], to=F)
    # build conv weight [10,2,1,1]: col0 = e0 (channel0=1), col1 = sel
    inits.append(ck.const("e0", (np.eye(10, dtype=np.float32)[0]).reshape(10, 1, 1, 1)))
    inits.append(ck.const("selw_shape", np.array([10, 1, 1, 1], np.int64)))
    n("Reshape", ["selw_pre", "selw_shape"], ["selw"])           # [10,1,1,1]
    n("Concat", ["e0", "selw"], ["W"], axis=1)                   # [10,2,1,1]
    n("Concat", ["bg", "M_comp"], ["pair"], axis=1)              # [1,2,30,30]
    n("Conv", ["pair", "W"], ["output"])                         # [1,10,30,30]

    ck.finalize(N, inits, "task036", OUT)


if __name__ == "__main__":
    build()
