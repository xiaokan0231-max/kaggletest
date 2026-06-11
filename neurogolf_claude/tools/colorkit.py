"""Color-selection helpers for NeuroGolf solutions.

Produce a one-hot color selector sel [1,10,1,1] over the 10 color channels,
then a per-cell color mask via a dynamic 1x1 Conv(input, sel) -> [1,1,30,30].
"""
import numpy as np
from onnx import TensorProto, numpy_helper

BIG = 4096.0


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def color_inits():
    return [
        const("kc_big", np.float32(BIG)),
        const("kc_one", np.float32(1.0)),
        const("kc_ax23", np.array([2, 3], np.int64)),
        const("kc_bigch", (BIG * np.array([0] + [1] * 9, np.float32)).reshape(1, 10, 1, 1)),
        const("kc_kill0", (BIG * np.array([1] + [0] * 9, np.float32)).reshape(1, 10, 1, 1)),
    ]


def emit_count(n, src="input"):
    n("ReduceSum", [src, "kc_ax23"], ["kc_count"], keepdims=1)   # [1,10,1,1]
    n("Sign", ["kc_count"], ["kc_present"])


def emit_select_rarest(n, sel="sel"):
    """one-hot of the rarest *present nonzero* color."""
    emit_count(n)
    n("Mul", ["kc_bigch", "kc_present"], ["kc_pen"])
    n("Add", ["kc_count", "kc_big"], ["kc_cb"])
    n("Sub", ["kc_cb", "kc_pen"], ["kc_score"])          # count if present-nonzero else BIG
    n("Neg", ["kc_score"], ["kc_nscore"])
    n("ReduceMax", ["kc_nscore"], ["kc_nmin"], axes=[1], keepdims=1)
    n("Neg", ["kc_nmin"], ["kc_minv"])
    n("Sub", ["kc_score", "kc_minv"], ["kc_sdiff"])
    n("Sign", ["kc_sdiff"], ["kc_ssign"])
    n("Sub", ["kc_one", "kc_ssign"], [sel])


def emit_select_common(n, sel="sel"):
    """one-hot of the most common *nonzero* color (channel 0 excluded)."""
    emit_count(n)
    n("Sub", ["kc_count", "kc_kill0"], ["kc_score"])     # count, but channel0 -> very negative
    n("ReduceMax", ["kc_score"], ["kc_mx"], axes=[1], keepdims=1)
    n("Sub", ["kc_score", "kc_mx"], ["kc_sdiff"])        # 0 at max, <0 else
    n("Sign", ["kc_sdiff"], ["kc_ssign"])               # 0 at max, -1 else
    n("Add", ["kc_one", "kc_ssign"], [sel])             # 1 at max, 0 else


def emit_colormask(n, sel="sel", out="cmask"):
    """cmask[1,1,30,30] = 1 where input pixel has the selected color."""
    n("Conv", ["input", sel], [out])
