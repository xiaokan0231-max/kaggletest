"""task244: grid of NxN solid blocks separated by single colored lines.
Output = each block's color (sample top-left of each block), mirrored left-right.

stride = first separator-line row + 1; sample input[a*stride, b*stride] via two
MatMuls with a sampling matrix, then reverse columns within [0,N).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from make_crop import emit_dims  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def build(tid="task244"):
    inits = ck.base_inits() + [
        ck.const("q_one", np.float32(1.0)),
        ck.const("q_ax3", np.array([3], np.int64)),
        ck.const("q_ax2", np.array([2], np.int64)),
        ck.const("q_cax2", np.array(2, np.int64)),
        ck.const("q_nzch", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)   # Hdim, Wdim, ck_iota_r/c
    # detect first full colored separator line row -> stride
    n("ReduceSum", ["input", "q_ax3"], ["rowcount"], keepdims=1)        # [1,10,30,1]
    n("Equal", ["rowcount", "Wdim"], ["eqWB"]); n("Cast", ["eqWB"], ["eqW"], to=1)
    n("Mul", ["eqW", "q_nzch"], ["eqWc"])
    n("ReduceMax", ["eqWc"], ["fullline"], axes=[1], keepdims=1)        # [1,1,30,1]
    n("CumSum", ["fullline", "q_cax2"], ["cumfl"])
    n("Sign", ["cumfl"], ["seen"]); n("Sub", ["q_one", "seen"], ["notseen"])
    n("ReduceSum", ["notseen", "q_ax2"], ["firstsep"], keepdims=1)      # [1,1,1,1]
    n("Add", ["firstsep", "q_one"], ["stride"])
    # N = (H+1)/stride
    n("Add", ["Hdim", "q_one"], ["H1"]); n("Div", ["H1", "stride"], ["Nf"])
    n("Floor", ["Nf"], ["Nn"]); n("Sub", ["Nn", "q_one"], ["Nm1"])
    # sampling matrix S[a,r] = (r == a*stride)
    n("Mul", ["ck_iota_r", "stride"], ["astride"])                     # [1,1,30,1]
    n("Equal", ["ck_iota_c", "astride"], ["SB"]); n("Cast", ["SB"], ["S"], to=1)  # [1,1,30,30]
    n("Transpose", ["S"], ["ST"], perm=[0, 1, 3, 2])
    n("MatMul", ["S", "input"], ["samp_r"])                            # rows sampled
    n("MatMul", ["samp_r", "ST"], ["us"])                             # cols sampled -> block grid
    # fliplr within [0,N)
    n("Add", ["ck_iota_r", "ck_iota_c"], ["rpc"])
    n("Equal", ["rpc", "Nm1"], ["crB"]); n("Cast", ["crB"], ["colrev"], to=1)
    n("MatMul", ["us", "colrev"], ["output"])
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    build()
