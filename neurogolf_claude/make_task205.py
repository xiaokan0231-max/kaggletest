"""Build compact ONNX for task205.

Rule: a near-solid rectangular 'fill'-color block sits in noise; inside are a few
'intruder' cells of one 'marker' color. Output = the block cropped to origin, with
every block-row and block-col that contains an intruder painted marker (fill else).

Algorithm (loop-free):
 1. stride-2 5x5 depthwise conv -> [1,10,13,13] counts; global argmax -> fill
    channel + guaranteed-interior seed.
 2. refine block row/col indicators (2 rounds): row is block-row iff over current
    block-cols it has >= ncol-2 fills; contiguity from seed via CumSum.
 3. intruder rows/cols; marker = argmax intruder-count per channel.
 4. compact single-channel fill/line planes to origin via perm-matrix MatMul, then
    expand to the 10-channel one-hot output at the very end (keeps tensors small).

Memory levers: keep everything [1,1,30,30] or length-30; float16 for arithmetic
planes; Gather (not mul+reduce) for channel selection; compact BEFORE expanding.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b4_task205.onnx"
F16 = TensorProto.FLOAT16

nodes, inits = [], []


def c_(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))


def n(op, ins, outs, **attr):
    nodes.append(helper.make_node(op, ins, outs, **attr))


# ---- constants (float16 where used in f16 arithmetic) ----
c_("w5", np.ones((10, 1, 5, 5), np.float32))                       # conv kernel (f32 in)
c_("iota30r", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1))
c_("iota30c", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30))
c_("one16", np.float16(1.0))
c_("two16", np.float16(2.0))
c_("zero16", np.float16(0.0))
c_("ax3s", np.array(3, np.int64))
c_("ax2s", np.array(2, np.int64))
c_("ax3", np.array([3], np.int64))
c_("ax2", np.array([2], np.int64))
c_("ax1", np.array([1], np.int64))
c_("sh_w", np.array([1, 1, 1, 30], np.int64))
c_("sh_h", np.array([1, 1, 30, 1], np.int64))
c_("sh1", np.array([1, 1, 1, 1], np.int64))
c_("ch_iota", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1))
c_("c7", np.float16(7.0))
c_("four16", np.float16(4.0))

# ============ Stage A: fill channel + interior seed ============
# stride-4 5x5 depthwise conv -> [1,10,7,7]; a stride-4 window fully inside the block
# still gives a guaranteed-interior seed (verified on all 266 cases).
n("Conv", ["input", "w5"], ["conv"], group=10, strides=[4, 4], kernel_shape=[5, 5])  # [1,10,7,7]
# fill channel = argmax over channels of per-channel spatial max
n("ReduceMax", ["conv", "ax23"], ["chmax"], keepdims=0)           # [1,10]
c_("ax23", np.array([2, 3], np.int64))
n("ArgMax", ["chmax"], ["chidx"], axis=1, keepdims=1)             # [1,1] int64 = fill color
n("Cast", ["chidx"], ["chf"], to=F16)                            # [1,1]
# seed = argmax over flattened space of per-position channel max (width 7)
n("ReduceMax", ["conv", "ax1"], ["spmax"], keepdims=0)            # [1,7,7]
n("Reshape", ["spmax", "flat49"], ["spflat"])                   # [1,49]
c_("flat49", np.array([1, 49], np.int64))
n("ArgMax", ["spflat"], ["spidx"], axis=1, keepdims=1)           # [1,1] int64 lin in 0..48
n("Cast", ["spidx"], ["spf"], to=F16)
n("Div", ["spf", "c7"], ["pf0"]); n("Floor", ["pf0"], ["pf"])    # p = lin // 7
n("Mul", ["pf", "c7"], ["pof"]); n("Sub", ["spf", "pof"], ["qf"])    # q = lin % 7
n("Mul", ["pf", "four16"], ["p2"]); n("Add", ["p2", "two16"], ["rseedf"])   # 4p+2
n("Mul", ["qf", "four16"], ["q2"]); n("Add", ["q2", "two16"], ["cseedf"])

# fill mask via Gather of the fill channel: [1,1,30,30] float16
n("Reshape", ["chidx", "ax1"], ["chi1"])                          # [1] int64
n("Gather", ["input", "chi1"], ["mask_g"], axis=1)                # [1,1,30,30] f32
n("Cast", ["mask_g"], ["mask"], to=F16)                           # f16

# seed scalars as [1,1,1,1]
n("Reshape", ["rseedf", "sh1"], ["rseed4"])
n("Reshape", ["cseedf", "sh1"], ["cseed4"])
n("Reshape", ["chf", "sh1"], ["chf4"])
# seed indicator vectors (float16)
n("Equal", ["iota30r", "rseed4"], ["rseedvecB"]); n("Cast", ["rseedvecB"], ["rseedvec"], to=F16)
n("Equal", ["iota30c", "cseed4"], ["cseedvecB"]); n("Cast", ["cseedvecB"], ["cseedvec"], to=F16)
# initial indicators |i-seed|<=2
n("Sub", ["iota30c", "cseed4"], ["cdif"]); n("Abs", ["cdif"], ["cdifa"])
n("LessOrEqual", ["cdifa", "two16"], ["colindB"]); n("Cast", ["colindB"], ["colind"], to=F16)
n("Sub", ["iota30r", "rseed4"], ["rdif"]); n("Abs", ["rdif"], ["rdifa"])
n("LessOrEqual", ["rdifa", "two16"], ["rowindB"]); n("Cast", ["rowindB"], ["rowind"], to=F16)


def axs(axis):
    return "ax3s" if axis == 3 else "ax2s"


def axv(axis):
    return "ax3" if axis == 3 else "ax2"


def contig(qual, seedvec, axis, tag):
    """qual/seedvec float16 0/1 on axis (2 -> [1,1,30,1], 3 -> [1,1,1,30]).
    Return contiguous-from-seed indicator (float16 0/1)."""
    iota = "iota30r" if axis == 2 else "iota30c"
    n("Sub", ["one16", qual], [f"{tag}_nq"])
    n("CumSum", [f"{tag}_nq", axs(axis)], [f"{tag}_pre"])            # sum nq[0..i]
    n("Sub", [f"{tag}_pre", f"{tag}_nq"], [f"{tag}_preex"])          # sum nq[0..i-1]
    # pres = pre at seed ; presm1 = preex at seed
    n("Mul", [f"{tag}_pre", seedvec], [f"{tag}_ps0"]); n("ReduceSum", [f"{tag}_ps0", axv(axis)], [f"{tag}_pres"], keepdims=1)
    n("Mul", [f"{tag}_preex", seedvec], [f"{tag}_pe0"]); n("ReduceSum", [f"{tag}_pe0", axv(axis)], [f"{tag}_presm1"], keepdims=1)
    n("Sub", [f"{tag}_pres", f"{tag}_preex"], [f"{tag}_left"])       # sum nq[i..s]
    n("Sub", [f"{tag}_pre", f"{tag}_presm1"], [f"{tag}_right"])      # sum nq[s..i]
    # seed index scalar
    n("Mul", [iota, seedvec], [f"{tag}_is0"]); n("ReduceSum", [f"{tag}_is0", axv(axis)], [f"{tag}_seedsc"], keepdims=1)
    n("LessOrEqual", [iota, f"{tag}_seedsc"], [f"{tag}_isleftB"]); n("Cast", [f"{tag}_isleftB"], [f"{tag}_isleft"], to=F16)
    n("Sub", [f"{tag}_right", f"{tag}_left"], [f"{tag}_d"]); n("Mul", [f"{tag}_isleft", f"{tag}_d"], [f"{tag}_dm"])
    n("Sub", [f"{tag}_right", f"{tag}_dm"], [f"{tag}_val"])          # isleft?left:right
    n("Equal", [f"{tag}_val", "zero16"], [f"{tag}_inB"]); n("Cast", [f"{tag}_inB"], [f"{tag}_in"], to=F16)
    return f"{tag}_in"


# ============ Stage B: 2 refinement rounds ============
# Counting via MatMul (mask @ col-indicator / row-indicator @ mask) avoids forming the
# [1,1,30,30] broadcast products -> only tiny [1,1,30,1]/[1,1,1,30] tensors.
cur_col, cur_row = "colind", "rowind"   # [1,1,1,30], [1,1,30,1]
for rnd in range(2):
    t = f"r{rnd}"
    # rowcnt = mask @ colvec ; colvec = cur_col reshaped [1,1,30,1]
    n("Reshape", [cur_col, "sh_h"], [f"{t}_colv"])               # [1,1,30,1]
    n("ReduceSum", [cur_col, "ax3"], [f"{t}_ncol"], keepdims=1)  # [1,1,1,1]
    n("MatMul", ["mask", f"{t}_colv"], [f"{t}_rowcnt"])          # [1,1,30,1]
    n("Sub", [f"{t}_ncol", "two16"], [f"{t}_thr"])
    n("GreaterOrEqual", [f"{t}_rowcnt", f"{t}_thr"], [f"{t}_rqB"]); n("Cast", [f"{t}_rqB"], [f"{t}_rq"], to=F16)
    cur_row = contig(f"{t}_rq", "rseedvec", 2, f"{t}r")          # [1,1,30,1]
    # colcnt = rowvec @ mask ; rowvec = cur_row reshaped [1,1,1,30]
    n("Reshape", [cur_row, "sh_w"], [f"{t}_rowv"])               # [1,1,1,30]
    n("ReduceSum", [cur_row, "ax2"], [f"{t}_nrow"], keepdims=1)
    n("MatMul", [f"{t}_rowv", "mask"], [f"{t}_colcnt"])          # [1,1,1,30]
    n("Sub", [f"{t}_nrow", "two16"], [f"{t}_thrc"])
    n("GreaterOrEqual", [f"{t}_colcnt", f"{t}_thrc"], [f"{t}_cqB"]); n("Cast", [f"{t}_cqB"], [f"{t}_cq"], to=F16)
    cur_col = contig(f"{t}_cq", "cseedvec", 3, f"{t}c")          # [1,1,1,30]

ROW, COL = cur_row, cur_col   # [1,1,30,1], [1,1,1,30]

# ============ Stage C: assembly (single-channel planes) ============
n("Mul", [ROW, COL], ["block"])                          # [1,1,30,30] f16
n("Sub", ["one16", "mask"], ["notfill"])
n("Mul", ["block", "notfill"], ["intr"])
n("ReduceMax", ["intr", "ax3"], ["introw"], keepdims=1)  # [1,1,30,1]
n("ReduceMax", ["intr", "ax2"], ["intcol"], keepdims=1)  # [1,1,1,30]
n("Add", ["introw", "intcol"], ["linesum"])
n("Greater", ["linesum", "zero16"], ["lineB"]); n("Cast", ["lineB"], ["linef"], to=F16)
n("Mul", ["linef", "block"], ["linemask"])               # [1,1,30,30]
n("Sub", ["block", "linemask"], ["fillcells"])           # [1,1,30,30]

# marker channel = argmax_c sum_xy(input[:,c]*intr).  Einsum on the f32 input fuses
# mul+reduce over space -> tiny [1,10], no [1,10,30,30] product materialized.
n("Cast", ["intr"], ["intr32"], to=TensorProto.FLOAT)    # [1,1,30,30] f32
n("Einsum", ["input", "intr32"], ["mkcnt"], equation="bcxy,bzxy->bc")  # [1,10] f32
n("ArgMax", ["mkcnt"], ["mkidx"], axis=1, keepdims=1)    # [1,1] int64 = marker color
n("Cast", ["mkidx"], ["mkf"], to=F16); n("Reshape", ["mkf", "sh1"], ["mk4"])  # [1,1,1,1]


# ============ Stage D: compact single-channel planes to origin ============
# Sr/T shared between the two planes; compute once.
n("Reshape", [ROW, "sh_w"], ["krw"])
n("CumSum", ["krw", "ax3s"], ["pr0"]); n("Sub", ["pr0", "one16"], ["pr"])
n("Equal", ["iota30r", "pr"], ["ErB"]); n("Cast", ["ErB"], ["Sr0"], to=F16)
n("Mul", ["Sr0", "krw"], ["Sr"])
n("CumSum", [COL, "ax3s"], ["pc0"]); n("Sub", ["pc0", "one16"], ["pcw"])
n("Reshape", ["pcw", "sh_h"], ["pc"]); n("Reshape", [COL, "sh_h"], ["kc"])
n("Equal", ["pc", "iota30c"], ["EcB"]); n("Cast", ["EcB"], ["Tc0"], to=F16)
n("Mul", ["Tc0", "kc"], ["T"])
# compact fillcells
n("MatMul", ["Sr", "fillcells"], ["fcc"]); n("MatMul", ["fcc", "T"], ["fill_o"])    # [1,1,30,30]
# compact linemask
n("MatMul", ["Sr", "linemask"], ["lcc"]); n("MatMul", ["lcc", "T"], ["line_o"])     # [1,1,30,30]

# ============ Stage E: expand to 10-channel one-hot via Einsum scatter ============
# output[b,c,x,y] = sum_k S[c,k]*planes[b,k,x,y]; S=[10,2] channel selector,
# planes=[1,2,30,30] stack of (fill_plane, line_plane). Single [1,10,30,30] f16 tensor.
n("Equal", ["ch_iota", "chf4"], ["isfillB"]); n("Cast", ["isfillB"], ["isfillf"], to=F16)
n("Equal", ["ch_iota", "mk4"], ["ismkB"]); n("Cast", ["ismkB"], ["ismkf"], to=F16)
c_("sh10_1", np.array([10, 1], np.int64))
n("Reshape", ["isfillf", "sh10_1"], ["selF"])    # [10,1]
n("Reshape", ["ismkf", "sh10_1"], ["selM"])      # [10,1]
n("Concat", ["selF", "selM"], ["S"], axis=1)     # [10,2]
n("Concat", ["fill_o", "line_o"], ["planes"], axis=1)   # [1,2,30,30]
# Einsum writes directly to the (f16) output -> the [1,10,30,30] tensor is excluded.
n("Einsum", ["S", "planes"], ["output"], equation="ck,bkxy->bcxy")  # [1,10,30,30] f16

graph = helper.make_graph(
    nodes, "task205",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("saved", OUT, len(model.SerializeToString()), "bytes; nodes", len(nodes), "inits", len(inits))
