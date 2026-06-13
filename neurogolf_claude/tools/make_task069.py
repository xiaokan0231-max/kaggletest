"""task069: a single multicolor "key" shape (colors 1-7,9) plus several color-8
blobs that are exact translates of the key's mask. Output: stamp the key's
color pattern onto every 8-blob, removing the key and the 8s.

Fully static, single-color-channel algorithm (no connected components):
  colorgrid = Conv(input, [0..9])                     -> integer color [1,1,30,30]
  eight     = (colorgrid == 8)
  iskey     = colorgrid in {1..7,9}
  keyc      = colorgrid * iskey                       -> color values at key cells
  Kcanon    = keyc compacted to top-left (== translation; key bbox has no empty
              interior row/col)                        [1,1,30,30]
  Kc        = top-left 3x4 of Kcanon (color kernel)    [1,1,3,4]
  M         = (Kc > 0) mask kernel; area = sum(M)
  corr      = Conv(eight, M)  (cross-correlation, padded)   [1,1,30,30]
  anchor    = (corr == area)
  outcolor  = crop( ConvTranspose(anchor, Kc) )        [1,1,30,30] color values
  output    = one-hot(outcolor) restricted to input grid extent.

corr==area alone is a sufficient anchor test (8-cells = disjoint union of key-mask
translates). Verified exact on all 262 train+test+arc-gen examples.

Everything is single-channel until the final one-hot expansion (which is the
excluded `output` tensor), keeping intermediate memory tiny.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
KH, KW = 3, 4


def build(tid="task069", out_path=None):
    out_path = out_path or f"{SOL}/{tid}.onnx"
    F16 = TensorProto.FLOAT16
    h16 = np.float16
    # key-color encoder: channel c -> color c for c in 1..7,9; ch0 and ch8 -> 0
    w_key32 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1).copy()
    w_key32[0, 8, 0, 0] = 0.0
    w_eight32 = np.zeros((1, 10, 1, 1), np.float32)
    w_eight32[0, 8, 0, 0] = 1.0
    colors = np.arange(10, dtype=h16).reshape(1, 10, 1, 1)    # one-hot decoder

    inits = [
        ck.const("w_key32", w_key32),
        ck.const("w_eight32", w_eight32),
        ck.const("colors", colors),
        ck.const("t_zero", np.array(0.0, h16)),
        ck.const("t_one", np.array(1.0, h16)),
        ck.const("ax_all", np.array([1, 2, 3], np.int64)),
        ck.const("ax3", np.array([3], np.int64)),
        # iota over kernel output rows/cols for building small gather matrices
        ck.const("iota_kh", np.arange(KH, dtype=h16).reshape(1, 1, KH, 1)),
        ck.const("iota_kw", np.arange(KW, dtype=h16).reshape(1, 1, 1, KW)),
        ck.const("sh_w", np.array([1, 1, 1, 30], np.int64)),
        ck.const("sh_h", np.array([1, 1, 30, 1], np.int64)),
    ]

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # 1) key-color grid, eight indicator, and active mask via fp32 channel-1x1 convs
    #    on the (excluded) float32 input; cast the 1-channel results to fp16 so the
    #    downstream pipeline is half-width without paying for a 10-channel fp16 cast.
    n("Conv", ["input", "w_key32"], ["keyc32"])               # key colors, fp32
    n("Cast", ["keyc32"], ["keyc"], to=F16)                   # [1,1,30,30] fp16
    n("Conv", ["input", "w_eight32"], ["eight32"])            # color-8 mask, fp32
    n("Cast", ["eight32"], ["eight"], to=F16)                 # [1,1,30,30] fp16
    n("ReduceMax", ["input"], ["active32"], axes=[1], keepdims=1)  # in-grid mask, fp32
    n("Cast", ["active32"], ["active"], to=F16)               # [1,1,30,30] fp16

    # 2) keep_row/keep_col of key as 0/1 indicators (reduce keyc first, then threshold
    #    on the small 1-D tensors to avoid a 30x30 iskey map)
    n("ReduceMax", ["keyc"], ["kc_colmax"], axes=[2], keepdims=1)  # [1,1,1,30]
    n("ReduceMax", ["keyc"], ["kc_rowmax"], axes=[3], keepdims=1)  # [1,1,30,1]
    n("Greater", ["kc_colmax", "t_zero"], ["keep_colB"])
    n("Cast", ["keep_colB"], ["keep_col"], to=F16)   # [1,1,1,30]
    n("Greater", ["kc_rowmax", "t_zero"], ["keep_rowB"])
    n("Cast", ["keep_rowB"], ["keep_row"], to=F16)   # [1,1,30,1]

    # 3) small direct gather of the key's top-left KH x KW block (no 30x30 compaction)
    #    Sr3[1,1,KH,30] gathers the first KH kept rows; Tc4[1,1,30,KW] the first KW kept cols.
    n("Reshape", ["keep_row", "sh_w"], ["kr_row"])             # [1,1,1,30]
    n("CumSum", ["kr_row", "ax3"], ["kr_ps"])
    n("Sub", ["kr_ps", "t_one"], ["kr_rank"])                  # rank among kept rows [1,1,1,30]
    n("Equal", ["iota_kh", "kr_rank"], ["SrB"])                # [1,1,KH,30]
    n("Cast", ["SrB"], ["SrE"], to=F16)
    n("Mul", ["SrE", "kr_row"], ["Sr3"])                       # [1,1,KH,30]
    n("CumSum", ["keep_col", "ax3"], ["kc_ps"])
    n("Sub", ["kc_ps", "t_one"], ["kc_rank"])                  # [1,1,1,30]
    # build Tc4 as [1,1,30,KW]: need rank as [1,1,30,1] and iota_kw [1,1,1,KW]
    n("Reshape", ["kc_rank", "sh_h"], ["kc_rank_h"])           # [1,1,30,1]
    n("Reshape", ["keep_col", "sh_h"], ["kc_col_h"])           # [1,1,30,1]
    n("Equal", ["kc_rank_h", "iota_kw"], ["TcB"])              # [1,1,30,KW]
    n("Cast", ["TcB"], ["TcE"], to=F16)
    n("Mul", ["TcE", "kc_col_h"], ["Tc4"])                     # [1,1,30,KW]
    n("MatMul", ["Sr3", "keyc"], ["KcRows"])                   # [1,1,KH,30]
    n("MatMul", ["KcRows", "Tc4"], ["Kc"])                     # [1,1,KH,KW]
    n("Greater", ["Kc", "t_zero"], ["Mb"])
    n("Cast", ["Mb"], ["M"], to=F16)                          # [1,1,3,4]
    n("ReduceSum", ["M", "ax_all"], ["area"], keepdims=1)     # [1,1,1,1]

    # 4) anchor detection via padded correlation
    n("Conv", ["eight", "M"], ["corr"], pads=[0, 0, KH - 1, KW - 1])  # [1,1,30,30]
    n("Equal", ["corr", "area"], ["anchorB"])
    n("Cast", ["anchorB"], ["anchor"], to=F16)               # [1,1,30,30]

    # 5) stamp color kernel at each anchor; pads crop the transposed-conv output
    #    from [1,1,32,33] straight to [1,1,30,30] (drop the 2 bottom rows / 3 right cols,
    #    which are always zero since anchors lie in-grid).
    n("ConvTranspose", ["anchor", "Kc"], ["outcolor"], pads=[0, 0, KH - 1, KW - 1])  # [1,1,30,30]

    # 6) mask out-of-grid cells to sentinel -1, then one-hot directly to output.
    # Stamps never fall outside the input grid extent, so just add (active-1):
    # in-grid cells unchanged, out-of-grid cells become -1 (match no color channel).
    # The Equal output is the bool `output` tensor itself (verifier does result>0),
    # so no float Cast and no 10-channel intermediate are needed.
    n("Sub", ["active", "t_one"], ["sentinel"])              # 0 in-grid, -1 out
    n("Add", ["outcolor", "sentinel"], ["outcolor_m"])       # color in-grid, -1 out
    n("Equal", ["outcolor_m", "colors"], ["output"])         # [1,10,30,30] bool == output

    _finalize_bool_output(N, inits, tid, out_path)


def _finalize_bool_output(nodes, inits, name, out_path):
    """Like cropkit.finalize but declares `output` as BOOL (excluded from memory,
    accepted by the verifier which only evaluates result > 0)."""
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
    build(out_path="/tmp/wf_task069.onnx")
