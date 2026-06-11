"""task134: two nonzero colors. The 'blob' = the dense color (max pixel_count /
bbox_area density -- a no-CC proxy for "largest connected component", verified
266/266 vs the CC rule). The 'noise' = the other color. The blob's bbox is
always a SQUARE of side in {6,9,12,15,18} (= 3*k, k in 2..6), and partitions
into a 3x3 grid of uniformly filled/empty blocks. Output (3x3): block = noise
color where the blob's block is filled, else background 0.

Pipeline (all no-CC):
  density = count / (Hbox*Wbox) per channel; blob = argmax over present
  nonzero channels (channel0 excluded). noise = the other present nonzero.
  blob mask -> compact bbox to TL -> downsample k=Hbox/3 via block-sum matrix
  Ek^T @ cb @ Ek -> P = Sign(blocksum) (3x3 macro). Emit noise at P=1,
  background channel0 in the full 3x3 region, all-zero outside.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
BIG = 100000.0


def build(tid="task134"):
    inits = ck.base_inits() + [
        ck.const("t_one", np.float32(1.0)), ck.const("t_zero", np.float32(0.0)),
        ck.const("t_three", np.float32(3.0)), ck.const("t_big", np.float32(BIG)),
        ck.const("t_ax23", np.array([2, 3], np.int64)),
        ck.const("t_ax2", np.array([2], np.int64)), ck.const("t_ax3", np.array([3], np.int64)),
        ck.const("t_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
        ck.const("t_e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # ---- per-channel count, bbox, density ----
    n("ReduceSum", ["input", "t_ax23"], ["cnt"], keepdims=1)              # [1,10,1,1]
    n("ReduceMax", ["input"], ["krp"], axes=[3], keepdims=1)              # [1,10,30,1]
    n("ReduceMax", ["input"], ["kcp"], axes=[2], keepdims=1)              # [1,10,1,30]
    n("ReduceSum", ["krp", "t_ax2"], ["Hc"], keepdims=1)                  # [1,10,1,1] bbox height
    n("ReduceSum", ["kcp", "t_ax3"], ["Wc"], keepdims=1)                  # [1,10,1,1] bbox width
    n("Mul", ["Hc", "Wc"], ["area"])
    n("Add", ["area", "t_one"], ["area1"])                               # +1 -> safe div (absent ch -> 0)
    n("Div", ["cnt", "area1"], ["density"])                             # [1,10,1,1]
    # present nonzero channels (exclude bg channel 0 and absent colors)
    n("Sign", ["cnt"], ["present"])
    n("Mul", ["present", "t_nzch"], ["presnz"])                         # [1,10,1,1] 1 iff present & ch>0
    # score = density - BIG*(1-presnz); argmax over channels -> blob
    n("Sub", ["t_one", "presnz"], ["notpn"]); n("Mul", ["notpn", "t_big"], ["pen"])
    n("Sub", ["density", "pen"], ["score"])
    n("ReduceMax", ["score"], ["maxsc"], axes=[1], keepdims=1)            # [1,1,1,1]
    n("Equal", ["score", "maxsc"], ["blobB"]); n("Cast", ["blobB"], ["blobsel"], to=1)  # [1,10,1,1]
    # noise = the other present nonzero channel
    n("Sub", ["presnz", "blobsel"], ["noisesel"])                       # [1,10,1,1] one-hot

    # ---- blob mask, compact bbox to TL, downsample to 3x3 macro ----
    n("Conv", ["input", "blobsel"], ["bmask"])                          # [1,1,30,30]
    n("ReduceMax", ["bmask"], ["krb"], axes=[3], keepdims=1)             # [1,1,30,1]
    n("ReduceMax", ["bmask"], ["kcb"], axes=[2], keepdims=1)             # [1,1,1,30]
    ck.emit_compaction(n, "krb", "kcb", "bmask", "cb")                   # blob bbox at TL [1,1,30,30]
    n("ReduceSum", ["krb", "t_ax2"], ["Hb"], keepdims=1)                 # [1,1,1,1] bbox side
    n("Div", ["Hb", "t_three"], ["k"])                                  # block size k (2..6)
    # Ek[i,j] = (floor(i/k) == j) ; downsample = Ek^T @ cb @ Ek
    n("Div", ["ck_iota_r", "k"], ["ir"]); n("Floor", ["ir"], ["irf"])
    n("Equal", ["irf", "ck_iota_c"], ["EkB"]); n("Cast", ["EkB"], ["Ek"], to=1)
    n("Transpose", ["Ek"], ["EkT"], perm=[0, 1, 3, 2])
    n("MatMul", ["EkT", "cb"], ["bs0"]); n("MatMul", ["bs0", "Ek"], ["bsum"])  # [1,1,30,30], 3x3 in TL
    n("Sign", ["bsum"], ["P"])                                          # macro filled pattern

    # ---- assemble 3x3 output: noise where P, bg where not, all-zero outside [0,3) ----
    n("Less", ["ck_iota_r", "t_three"], ["r3B"]); n("Cast", ["r3B"], ["r3"], to=1)
    n("Less", ["ck_iota_c", "t_three"], ["c3B"]); n("Cast", ["c3B"], ["c3"], to=1)
    n("Mul", ["r3", "c3"], ["reg3"])                                    # [1,1,30,30] the 3x3 block
    n("Mul", ["P", "reg3"], ["Pin"])                                    # P restricted to 3x3
    n("Mul", ["Pin", "noisesel"], ["out_fg"])                          # [1,10,30,30] noise channel
    n("Sub", ["reg3", "Pin"], ["bgcells"])                            # 3x3 cells where P==0
    n("Mul", ["bgcells", "t_e0"], ["out_bg"])                         # channel0 = 1 there
    n("Add", ["out_fg", "out_bg"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
