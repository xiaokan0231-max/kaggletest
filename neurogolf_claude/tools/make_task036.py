"""Build task036.onnx.

Rule: target color = argmax over colors 1-9 of max 3x3-window count;
output = bbox crop of target cells (target color kept, rest 0).

Pipeline (all intermediates kept lean):
1. dens = AveragePool3x3(input, count_include_pad=1)  -> window density per channel
2. score = spatial ReduceMax; penalize channel 0 by -100; sel = 1+Sign(score-max)
   -> [1,10,1,1] one-hot selector (channel 0 always 0)
3. M = Conv(input, sel)  -> [1,1,30,30] target mask (dynamic 1x1 conv weight,
   avoids a fat [1,10,30,30]/[10,900] intermediate)
4. SHRINK: keep_r/keep_c via ReduceMax; p = CumSum(keep)*keep - 1 (non-kept rows
   get -1, so no keep-mask multiply needed on the permutation matrix);
   S = Cast(Equal(iota, p)); M_comp = S_row @ M @ S_col^T (batched MatMul on
   [1,1,30,30]).
5. G_out = outer(rowmax(M_comp), colmax(M_comp)); bg = G_out - M_comp.
6. output = Conv(Concat(bg, M_comp), Concat(e0, sel_reshaped)) -> builds the
   [1,10,30,30] one-hot directly into "output" (excluded from byte count).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task036.onnx"
F = TensorProto.FLOAT


def init_f(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    return helper.make_tensor(name, F, arr.shape, arr.flatten())


def init_i64(name, arr):
    arr = np.asarray(arr, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, arr.shape, arr.flatten())


def init_i32_scalar(name, v):
    return helper.make_tensor(name, TensorProto.INT32, [], [v])


pen = np.zeros((1, 10, 1, 1), dtype=np.float32)
pen[0, 0, 0, 0] = 100.0
e0 = np.zeros((10, 1, 1, 1), dtype=np.float32)
e0[0, 0, 0, 0] = 1.0

initializers = [
    init_f("pen", pen),                                   # channel-0 penalty
    init_f("one", np.float32(1.0)),
    init_i32_scalar("ax2", 2),
    init_i32_scalar("ax3", 3),
    init_i64("shape_row", [1, 1, 1, 30]),
    init_i64("shape_col", [1, 1, 30, 1]),
    init_i64("shape_w", [10, 1, 1, 1]),
    init_f("iota_r", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
    init_f("iota_c", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
    init_f("e0", e0),                                     # bg column of final conv weight
]

nodes = [
    # --- target color selection (window-density argmax, ch0 penalized) ---
    helper.make_node("AveragePool", ["input"], ["dens"], kernel_shape=[3, 3],
                     pads=[1, 1, 1, 1], strides=[1, 1], count_include_pad=1),
    helper.make_node("ReduceMax", ["dens"], ["score"], axes=[2, 3], keepdims=1),
    helper.make_node("Sub", ["score", "pen"], ["score_adj"]),
    helper.make_node("ReduceMax", ["score_adj"], ["smax"], axes=[1], keepdims=1),
    helper.make_node("Sub", ["score_adj", "smax"], ["sdiff"]),
    helper.make_node("Sign", ["sdiff"], ["ssign"]),
    helper.make_node("Add", ["ssign", "one"], ["sel"]),   # [1,10,1,1] one-hot
    # --- target mask via dynamic 1x1 conv ---
    helper.make_node("Conv", ["input", "sel"], ["M"]),    # [1,1,30,30]
    # --- row compaction ---
    helper.make_node("ReduceMax", ["M"], ["keep_r"], axes=[3], keepdims=1),
    helper.make_node("CumSum", ["keep_r", "ax2"], ["cum_r"]),
    helper.make_node("Mul", ["cum_r", "keep_r"], ["pk_r"]),
    helper.make_node("Sub", ["pk_r", "one"], ["p_r"]),    # kept: dest idx, else -1
    helper.make_node("Reshape", ["p_r", "shape_row"], ["p_rT"]),
    helper.make_node("Equal", ["iota_r", "p_rT"], ["eq_r"]),
    helper.make_node("Cast", ["eq_r"], ["S_row"], to=F),  # [1,1,30,30]
    # --- column compaction ---
    helper.make_node("ReduceMax", ["M"], ["keep_c"], axes=[2], keepdims=1),
    helper.make_node("CumSum", ["keep_c", "ax3"], ["cum_c"]),
    helper.make_node("Mul", ["cum_c", "keep_c"], ["pk_c"]),
    helper.make_node("Sub", ["pk_c", "one"], ["p_c"]),
    helper.make_node("Reshape", ["p_c", "shape_col"], ["p_cT"]),
    helper.make_node("Equal", ["p_cT", "iota_c"], ["eq_c"]),
    helper.make_node("Cast", ["eq_c"], ["S_colT"], to=F),  # [1,1,30,30]
    # --- compacted mask + output-grid mask ---
    helper.make_node("MatMul", ["M", "S_colT"], ["tmp"]),
    helper.make_node("MatMul", ["S_row", "tmp"], ["M_comp"]),
    helper.make_node("ReduceMax", ["M_comp"], ["rmax"], axes=[3], keepdims=1),
    helper.make_node("ReduceMax", ["M_comp"], ["cmax"], axes=[2], keepdims=1),
    helper.make_node("Mul", ["rmax", "cmax"], ["G_out"]),
    helper.make_node("Sub", ["G_out", "M_comp"], ["bg"]),
    # --- assemble one-hot output with a single conv into "output" ---
    helper.make_node("Concat", ["bg", "M_comp"], ["pair"], axis=1),  # [1,2,30,30]
    helper.make_node("Reshape", ["sel", "shape_w"], ["selw"]),       # [10,1,1,1]
    helper.make_node("Concat", ["e0", "selw"], ["W"], axis=1),       # [10,2,1,1]
    helper.make_node("Conv", ["pair", "W"], ["output"]),
]

graph = helper.make_graph(
    nodes, "task036",
    [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
    initializer=initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("saved", OUT)
