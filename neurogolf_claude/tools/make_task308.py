"""task308: background = most-common (nonzero) color. Each other present color's
bbox crop (the contiguous bounding rectangle, internal gaps kept) is centered in
a grid sized to the largest bbox (maxH x maxW) and all crops overlaid. Output
filled with bg elsewhere.

Uses bbox-FILL (range between first/last occupied row/col) so scattered colors
keep their internal structure, then centers via dynamic down/right shifts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task308"):
    inits = ck.base_inits() + [
        ck.const("e_one", np.float32(1.0)), ck.const("e_half", np.float32(0.5)),
        ck.const("e_cax2", np.array(2, np.int64)), ck.const("e_cax3", np.array(3, np.int64)),
        ck.const("e_ax23", np.array([2, 3], np.int64)), ck.const("e_ax2", np.array([2], np.int64)),
        ck.const("e_ax3", np.array([3], np.int64)), ck.const("e_ax1", np.array([1], np.int64)),
    ]
    for c in range(10):
        oh = np.zeros((1, 10, 1, 1), np.float32); oh[0, c, 0, 0] = 1.0
        inits.append(ck.const(f"oh{c}", oh))
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    n("Sub", ["ck_iota_r", "ck_iota_c"], ["rmc"])

    def fill(band, cax, rax, out):
        n("CumSum", [band, cax], [f"{out}_cum"]); n("Sign", [f"{out}_cum"], [f"{out}_hl"])
        n("ReduceSum", [band, rax], [f"{out}_tot"], keepdims=1)
        n("Sub", [f"{out}_cum", band], [f"{out}_cex"])
        n("Sub", [f"{out}_tot", f"{out}_cex"], [f"{out}_d"]); n("Sign", [f"{out}_d"], [f"{out}_hr"])
        n("Mul", [f"{out}_hl", f"{out}_hr"], [out])

    # bg = argmax channel count; placeable channels
    n("ReduceSum", ["input", "e_ax23"], ["cnt"], keepdims=1)
    n("Sign", ["cnt"], ["present"])
    n("ReduceMax", ["cnt"], ["bgmax"], axes=[1], keepdims=1)
    n("Equal", ["cnt", "bgmax"], ["bgB"]); n("Cast", ["bgB"], ["bgsel"], to=1)
    n("Sub", ["e_one", "bgsel"], ["notbg"]); n("Mul", ["present", "notbg"], ["place"])
    # per-channel bbox height/width (filled) -> maxH, maxW
    n("ReduceMax", ["input"], ["krp"], axes=[3], keepdims=1)           # [1,10,30,1]
    n("ReduceMax", ["input"], ["kcp"], axes=[2], keepdims=1)           # [1,10,1,30]
    fill("krp", "e_cax2", "e_ax2", "krpf")
    fill("kcp", "e_cax3", "e_ax3", "kcpf")
    n("ReduceSum", ["krpf", "e_ax2"], ["hpc"], keepdims=1)             # [1,10,1,1] bbox heights
    n("ReduceSum", ["kcpf", "e_ax3"], ["wpc"], keepdims=1)
    n("Mul", ["hpc", "place"], ["hpl"]); n("ReduceMax", ["hpl"], ["maxH"], axes=[1], keepdims=1)
    n("Mul", ["wpc", "place"], ["wpl"]); n("ReduceMax", ["wpl"], ["maxW"], axes=[1], keepdims=1)
    outs = []
    for c in range(1, 10):
        p = f"o{c}"
        n("Conv", ["input", f"oh{c}"], [f"{p}_m"])
        n("ReduceMax", [f"{p}_m"], [f"{p}_mr"], axes=[3], keepdims=1)
        n("ReduceMax", [f"{p}_m"], [f"{p}_mc"], axes=[2], keepdims=1)
        fill(f"{p}_mr", "e_cax2", "e_ax2", f"{p}_kr")
        fill(f"{p}_mc", "e_cax3", "e_ax3", f"{p}_kc")
        ck.emit_compaction(n, f"{p}_kr", f"{p}_kc", f"{p}_m", f"{p}_pc")
        n("ReduceSum", [f"{p}_kr", "e_ax2"], [f"{p}_h"], keepdims=1)
        n("ReduceSum", [f"{p}_kc", "e_ax3"], [f"{p}_w"], keepdims=1)
        n("Sub", ["maxH", f"{p}_h"], [f"{p}_dh"]); n("Mul", [f"{p}_dh", "e_half"], [f"{p}_dh2"])
        n("Floor", [f"{p}_dh2"], [f"{p}_r0"])
        n("Sub", ["maxW", f"{p}_w"], [f"{p}_dw"]); n("Mul", [f"{p}_dw", "e_half"], [f"{p}_dw2"])
        n("Floor", [f"{p}_dw2"], [f"{p}_c0"]); n("Neg", [f"{p}_c0"], [f"{p}_nc0"])
        n("Equal", ["rmc", f"{p}_r0"], [f"{p}_QdB"]); n("Cast", [f"{p}_QdB"], [f"{p}_Qd"], to=1)
        n("Equal", ["rmc", f"{p}_nc0"], [f"{p}_QrB"]); n("Cast", [f"{p}_QrB"], [f"{p}_Qr"], to=1)
        n("MatMul", [f"{p}_pc", f"{p}_Qr"], [f"{p}_pr"]); n("MatMul", [f"{p}_Qd", f"{p}_pr"], [f"{p}_pl"])
        n("Mul", ["place", f"oh{c}"], [f"{p}_g0"]); n("ReduceSum", [f"{p}_g0", "e_ax1"], [f"{p}_g"], keepdims=1)
        n("Mul", [f"{p}_pl", f"{p}_g"], [f"{p}_plg"]); n("Mul", [f"{p}_plg", f"oh{c}"], [f"{p}_out"])
        outs.append(f"{p}_out")
    acc = outs[0]
    for i, o in enumerate(outs[1:], 1):
        nm = f"fa{i}"; n("Add", [acc, o], [nm]); acc = nm
    n("ReduceMax", [acc], ["fgpres"], axes=[1], keepdims=1)
    n("Less", ["ck_iota_r", "maxH"], ["rmB"]); n("Cast", ["rmB"], ["rm"], to=1)
    n("Less", ["ck_iota_c", "maxW"], ["cmB"]); n("Cast", ["cmB"], ["cm"], to=1)
    n("Mul", ["rm", "cm"], ["reg"])
    n("Sub", ["e_one", "fgpres"], ["nofg"]); n("Mul", ["reg", "nofg"], ["bgm"])
    n("Mul", ["bgm", "bgsel"], ["bgfill"])
    n("Add", [acc, "bgfill"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
