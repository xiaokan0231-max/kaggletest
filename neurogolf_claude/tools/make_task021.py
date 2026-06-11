"""Generate task021.onnx.

Rule: input has exactly 2 colors (bg = strict majority, sep = the other).
Sep forms full horizontal/vertical lines partitioning the grid into R x C
bands (lines never at edge, never adjacent => R = full sep rows + 1,
C = full sep cols + 1). Output: R x C grid filled with bg.

Memory-lean graph (cost model = sum of all intermediate tensor sizes;
"input"/"output" are free). No 36KB intermediates at all:
  rc    = ReduceSum(input, W)  [1,10,30,1]   per-row per-color counts
  cc    = ReduceSum(input, H)  [1,10,1,30]   per-col per-color counts
  count = ReduceSum(rc, H)     [1,10,1,1]
  sgn   = Sign(count - max(count))    (0 at bg, -1 elsewhere)
  bg    = sgn + 1                     (bg one-hot, also used as 1x1 Conv weight)
  in_r  = ReduceSum(rc, ch)           in-grid cells per row
  ns_r  = Conv(rc, bg)                bg cells per row (channel contraction)
  full-sep-row count ln_r = sum(Sign(in_r)) - sum(Sign(ns_r))   (= h - bg-rows)
  rowkeep = Relu(ln_r - (iota - 0.5))   positive for rows < R = ln_r + 1
  cols symmetric; output = (bg * rowkeep) * colkeep   (judged by > 0).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task021.onnx"


def build():
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "ax2"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "ax3"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(
            (np.arange(30, dtype=np.float32) - 0.5).reshape(1, 1, 30, 1), "iota_r"),
        numpy_helper.from_array(
            (np.arange(30, dtype=np.float32) - 0.5).reshape(1, 1, 1, 30), "iota_c"),
    ]

    n = helper.make_node
    nodes = [
        # per-row / per-col color counts (the only reads of the fat input)
        n("ReduceSum", ["input", "ax3"], ["rc"], keepdims=1),    # [1,10,30,1]
        n("ReduceSum", ["input", "ax2"], ["cc"], keepdims=1),    # [1,10,1,30]
        # bg one-hot selector
        n("ReduceSum", ["rc", "ax2"], ["count"], keepdims=1),    # [1,10,1,1]
        n("ReduceMax", ["count"], ["maxc"], axes=[1], keepdims=1),
        n("Sub", ["count", "maxc"], ["dmax"]),
        n("Sign", ["dmax"], ["sgn"]),                            # 0 bg / -1 rest
        n("Add", ["sgn", "one"], ["bg"]),                        # [1,10,1,1]
        # rows: in-grid count and bg count per row
        n("ReduceSum", ["rc", "ax1"], ["in_r"], keepdims=1),     # [1,1,30,1]
        n("Conv", ["rc", "bg"], ["ns_r"]),                       # bg cells per row
        n("Sign", ["in_r"], ["a_r"]),                            # row in grid?
        n("Sign", ["ns_r"], ["b_r"]),                            # row has bg?
        n("ReduceSum", ["a_r", "ax2"], ["sa_r"], keepdims=1),    # grid height
        n("ReduceSum", ["b_r", "ax2"], ["sb_r"], keepdims=1),    # rows with bg
        n("Sub", ["sa_r", "sb_r"], ["ln_r"]),                    # full sep rows = R-1
        n("Sub", ["ln_r", "iota_r"], ["dr"]),                    # R-0.5-i
        n("Relu", ["dr"], ["rowkeep"]),                          # >0 iff i < R
        # cols symmetric
        n("ReduceSum", ["cc", "ax1"], ["in_c"], keepdims=1),     # [1,1,1,30]
        n("Conv", ["cc", "bg"], ["ns_c"]),
        n("Sign", ["in_c"], ["a_c"]),
        n("Sign", ["ns_c"], ["b_c"]),
        n("ReduceSum", ["a_c", "ax3"], ["sa_c"], keepdims=1),
        n("ReduceSum", ["b_c", "ax3"], ["sb_c"], keepdims=1),
        n("Sub", ["sa_c", "sb_c"], ["ln_c"]),
        n("Sub", ["ln_c", "iota_c"], ["dc"]),
        n("Relu", ["dc"], ["colkeep"]),
        # combine; keep the fat [1,10,30,30] only as the free "output"
        n("Mul", ["bg", "rowkeep"], ["m1"]),                     # [1,10,30,1]
        n("Mul", ["m1", "colkeep"], ["output"]),                 # [1,10,30,30]
    ]

    graph = helper.make_graph(
        nodes, "task021",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    nparams = sum(int(np.prod(t.dims)) if t.dims else 1 for t in inits)
    print(f"saved {OUT}  nodes={len(nodes)}  params={nparams}")


if __name__ == "__main__":
    build()
