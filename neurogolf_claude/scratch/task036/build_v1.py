"""task036 v1: float16 version of deployed structure.

Rule: target color = argmax over colors 1-9 of max 3x3-window density;
output = bbox crop of target cells (target color kept, rest 0).

v1 changes vs deployed: do all spatial work in float16 (2 bytes) instead of
float32 (4 bytes) to halve the dominant 30x30 intermediate tensors.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b4_task036.onnx"
F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT


def init_f16(name, arr):
    arr = np.asarray(arr, dtype=np.float16)
    return numpy_helper.from_array(arr, name=name)


def init_i64(name, arr):
    arr = np.asarray(arr, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, arr.shape, arr.flatten())


def init_i32_scalar(name, v):
    return helper.make_tensor(name, TensorProto.INT32, [], [v])


pen = np.zeros((1, 10, 1, 1), dtype=np.float16)
pen[0, 0, 0, 0] = 100.0
e0 = np.zeros((10, 1, 1, 1), dtype=np.float16)
e0[0, 0, 0, 0] = 1.0

initializers = [
    init_f16("pen", pen),
    init_f16("one", np.float16(1.0)),
    init_i32_scalar("ax2", 2),
    init_i32_scalar("ax3", 3),
    init_i64("shape_row", [1, 1, 1, 30]),
    init_i64("shape_col", [1, 1, 30, 1]),
    init_i64("shape_w", [10, 1, 1, 1]),
    init_f16("iota_r", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)),
    init_f16("iota_c", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)),
    init_f16("e0", e0),
]

nodes = [
    # cast input one-hot float32 -> float16
    helper.make_node("Cast", ["input"], ["inp"], to=F16),
    # --- target color selection (window-density argmax, ch0 penalized) ---
    helper.make_node("AveragePool", ["inp"], ["dens"], kernel_shape=[3, 3],
                     pads=[1, 1, 1, 1], strides=[1, 1], count_include_pad=1),
    helper.make_node("ReduceMax", ["dens"], ["score"], axes=[2, 3], keepdims=1),
    helper.make_node("Sub", ["score", "pen"], ["score_adj"]),
    helper.make_node("ReduceMax", ["score_adj"], ["smax"], axes=[1], keepdims=1),
    helper.make_node("Sub", ["score_adj", "smax"], ["sdiff"]),
    helper.make_node("Sign", ["sdiff"], ["ssign"]),
    helper.make_node("Add", ["ssign", "one"], ["sel"]),
    # --- target mask via dynamic 1x1 conv ---
    helper.make_node("Conv", ["inp", "sel"], ["M"]),
    # --- row compaction ---
    helper.make_node("ReduceMax", ["M"], ["keep_r"], axes=[3], keepdims=1),
    helper.make_node("CumSum", ["keep_r", "ax2"], ["cum_r"]),
    helper.make_node("Mul", ["cum_r", "keep_r"], ["pk_r"]),
    helper.make_node("Sub", ["pk_r", "one"], ["p_r"]),
    helper.make_node("Reshape", ["p_r", "shape_row"], ["p_rT"]),
    helper.make_node("Equal", ["iota_r", "p_rT"], ["eq_r"]),
    helper.make_node("Cast", ["eq_r"], ["S_row"], to=F16),
    # --- column compaction ---
    helper.make_node("ReduceMax", ["M"], ["keep_c"], axes=[2], keepdims=1),
    helper.make_node("CumSum", ["keep_c", "ax3"], ["cum_c"]),
    helper.make_node("Mul", ["cum_c", "keep_c"], ["pk_c"]),
    helper.make_node("Sub", ["pk_c", "one"], ["p_c"]),
    helper.make_node("Reshape", ["p_c", "shape_col"], ["p_cT"]),
    helper.make_node("Equal", ["p_cT", "iota_c"], ["eq_c"]),
    helper.make_node("Cast", ["eq_c"], ["S_colT"], to=F16),
    # --- compacted mask + output-grid mask ---
    helper.make_node("MatMul", ["M", "S_colT"], ["tmp"]),
    helper.make_node("MatMul", ["S_row", "tmp"], ["M_comp"]),
    helper.make_node("ReduceMax", ["M_comp"], ["rmax"], axes=[3], keepdims=1),
    helper.make_node("ReduceMax", ["M_comp"], ["cmax"], axes=[2], keepdims=1),
    helper.make_node("Mul", ["rmax", "cmax"], ["G_out"]),
    helper.make_node("Sub", ["G_out", "M_comp"], ["bg"]),
    # --- assemble one-hot output with a single conv, cast to float32 ---
    helper.make_node("Concat", ["bg", "M_comp"], ["pair"], axis=1),
    helper.make_node("Reshape", ["sel", "shape_w"], ["selw"]),
    helper.make_node("Concat", ["e0", "selw"], ["W"], axis=1),
    helper.make_node("Conv", ["pair", "W"], ["out16"]),
    helper.make_node("Cast", ["out16"], ["output"], to=F32),
]

graph = helper.make_graph(
    nodes, "task036",
    [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
    initializer=initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("saved", OUT)
