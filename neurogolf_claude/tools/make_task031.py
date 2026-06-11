"""task031: crop to bounding box of nonzero cells.

Approach (SHRINK permutation matrices, opset 17):
  1. M = Conv(input, W[1,10,1,1]) with W = [0,1,1,...,1]  -> colored-cell mask [1,1,30,30]
  2. any_r = ReduceMax(M, cols), any_c = ReduceMax(M, rows)
  3. keep = Sign(CumSum(any) * CumSum(any, reverse))   (true bbox rows/cols)
  4. p = CumSum(keep) - 1
     Sr [12,30]  = Cast(Equal(iota12[12,1], p_r[1,30])) * keep_r[1,30]
     ScT [30,12] = Cast(Equal(p_c[30,1], iota12T[1,12])) * keep_c[30,1]
     (inputs are <=12x12 so the bbox never exceeds 12 -> compact matrices are exact)
  5. Z = MatMul(Sr, MatMul(input, ScT))  -> [1,10,12,12]; output = Pad(Z) to [1,10,30,30]
Permuted one-hot is exact 0/1, so it directly satisfies the >0 criterion.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task031.onnx"
K = 12  # max bbox extent (inputs are at most 12x12)

def init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT if arr.dtype == np.float32
                              else TensorProto.INT64, arr.shape, arr.flatten().tolist())

inits = [
    init("Wsel",   np.array([0] + [1]*9, dtype=np.float32).reshape(1, 10, 1, 1)),
    init("iota",   np.arange(K, dtype=np.float32).reshape(K, 1)),
    init("iotaT",  np.arange(K, dtype=np.float32).reshape(1, K)),
    init("one",    np.array(1.0, dtype=np.float32).reshape(())),
    init("ax2",    np.array(2, dtype=np.int64).reshape(())),
    init("ax3",    np.array(3, dtype=np.int64).reshape(())),
    init("shp_r",  np.array([1, 30], dtype=np.int64)),
    init("shp_c",  np.array([30, 1], dtype=np.int64)),
    init("pads",   np.array([0, 0, 0, 0, 0, 0, 30 - K, 30 - K], dtype=np.int64)),
]

nodes = [helper.make_node("Conv", ["input", "Wsel"], ["M"])]

def axis_chain(tag, reduce_axis, cum_ax, shp, eq_inputs):
    """any -> keep -> p -> compact selection matrix S<tag>."""
    return [
        helper.make_node("ReduceMax", ["M"], [f"any{tag}"], axes=[reduce_axis], keepdims=1),
        helper.make_node("CumSum", [f"any{tag}", cum_ax], [f"f{tag}"]),
        helper.make_node("CumSum", [f"any{tag}", cum_ax], [f"b{tag}"], reverse=1),
        helper.make_node("Mul", [f"f{tag}", f"b{tag}"], [f"m{tag}"]),
        helper.make_node("Sign", [f"m{tag}"], [f"keep{tag}"]),
        helper.make_node("CumSum", [f"keep{tag}", cum_ax], [f"c{tag}"]),
        helper.make_node("Sub", [f"c{tag}", "one"], [f"p{tag}"]),
        helper.make_node("Reshape", [f"p{tag}", shp], [f"p{tag}2"]),
        helper.make_node("Reshape", [f"keep{tag}", shp], [f"keep{tag}2"]),
        helper.make_node("Equal", eq_inputs, [f"eq{tag}"]),
        helper.make_node("Cast", [f"eq{tag}"], [f"eqf{tag}"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [f"eqf{tag}", f"keep{tag}2"], [f"S{tag}"]),
    ]

# Sr[i,r] = (p_r[r]==i)*keep_r[r]  -> [12,30]
nodes += axis_chain("r", 3, "ax2", "shp_r", ["iota", "pr2"])
# Sc[c,j] = (p_c[c]==j)*keep_c[c]  -> [30,12]  (this is S_c^T directly)
nodes += axis_chain("c", 2, "ax3", "shp_c", ["pc2", "iotaT"])
nodes += [
    helper.make_node("MatMul", ["input", "Sc"], ["Y"]),   # [1,10,30,12]
    helper.make_node("MatMul", ["Sr", "Y"], ["Z"]),       # [1,10,12,12]
    helper.make_node("Pad", ["Z", "pads"], ["output"]),   # [1,10,30,30]
]

graph = helper.make_graph(
    nodes, "task031",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("saved", OUT, len(model.SerializeToString()), "bytes")
