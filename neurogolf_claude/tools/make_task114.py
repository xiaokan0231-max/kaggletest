"""task114: EXPAND (h+2)x(w+2) = edge-replicate pad 1 ring, four corners forced to background 0.

Construction (all dynamic via row/col selection matrices, opset 13, no Equal needed):
  G  = ReduceMax(input, ch)            -> [1,1,30,30] grid mask
  g  = ReduceMax(G, axis=3)            -> row indicator (colvec), reshaped to rowvec
  v  = ReduceMax(G, axis=2)            -> col indicator (rowvec)
  last = g - shift_up(g)               (one-hot at h-1; shift via const sub-diag D1)
  u    = shift_down2(last)             (one-hot at h+1; via const D2[i,i+2])
  R  = E00 + D1*g_row + u_col @ last_row      [30,30] row selection: out row i <- src row
  CT = E00 + D1T*v_col + lastc_col @ uc_row   transposed col selection
  Y  = R @ X @ CT                      (two batched MatMuls, broadcast over 10 channels)
  corner mask M = (e0+u)_col @ (e0+uc)_row ; out = Y + M*K, K=[+1,-2,...,-2] per channel
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

F = TensorProto.FLOAT

def const(name, arr):
    arr = np.asarray(arr)
    return helper.make_tensor(name, TensorProto.INT64 if arr.dtype == np.int64 else F,
                              arr.shape, arr.flatten().tolist())

# constant matrices
D1 = np.zeros((30, 30), np.float32)   # D1[i, i-1] = 1  (sub-diagonal)
for i in range(1, 30):
    D1[i, i - 1] = 1.0
D1T = D1.T.copy()                      # D1T[i, i+1] = 1
D2 = np.zeros((30, 30), np.float32)   # D2[i, i+2] = 1
for i in range(28):
    D2[i, i + 2] = 1.0
E00 = np.zeros((30, 30), np.float32)
E00[0, 0] = 1.0
e0_col = np.zeros((30, 1), np.float32); e0_col[0, 0] = 1.0
e0_row = np.zeros((1, 30), np.float32); e0_row[0, 0] = 1.0
K = np.full((10, 1, 1), -2.0, np.float32); K[0] = 1.0

inits = [
    const("D1", D1), const("D1T", D1T), const("D2", D2), const("E00", E00),
    const("e0_col", e0_col), const("e0_row", e0_row), const("K", K),
    const("shape_row", np.array([1, 1, 1, 30], np.int64)),
    const("shape_col", np.array([1, 1, 30, 1], np.int64)),
]

n = helper.make_node
nodes = [
    # masks / indicators
    n("ReduceMax", ["input"], ["G"], axes=[1], keepdims=1),          # [1,1,30,30]
    n("ReduceMax", ["G"], ["g4"], axes=[3], keepdims=1),             # [1,1,30,1] rows
    n("ReduceMax", ["G"], ["v_row"], axes=[2], keepdims=1),          # [1,1,1,30] cols
    n("Reshape", ["g4", "shape_row"], ["g_row"]),                    # [1,1,1,30]
    # rows: last (one-hot h-1), u (one-hot h+1)
    n("MatMul", ["g_row", "D1"], ["g_shift"]),                       # g_{j+1}
    n("Sub", ["g_row", "g_shift"], ["last_row"]),
    n("MatMul", ["last_row", "D2"], ["u_row"]),
    n("Reshape", ["u_row", "shape_col"], ["u_col"]),
    # R = E00 + D1*g + u@last
    n("Mul", ["D1", "g_row"], ["mid_R"]),                            # [1,1,30,30]
    n("MatMul", ["u_col", "last_row"], ["corner_R"]),
    n("Sum", ["mid_R", "corner_R", "E00"], ["R"]),
    # cols: lastc (one-hot w-1), uc (one-hot w+1)
    n("MatMul", ["v_row", "D1"], ["v_shift"]),
    n("Sub", ["v_row", "v_shift"], ["lastc_row"]),
    n("MatMul", ["lastc_row", "D2"], ["uc_row"]),
    n("Reshape", ["lastc_row", "shape_col"], ["lastc_col"]),
    n("Reshape", ["v_row", "shape_col"], ["v_col"]),
    # CT = E00 + D1T*v_col + lastc@uc
    n("Mul", ["D1T", "v_col"], ["mid_CT"]),
    n("MatMul", ["lastc_col", "uc_row"], ["corner_CT"]),
    n("Sum", ["mid_CT", "corner_CT", "E00"], ["CT"]),
    # Y = R @ X @ CT
    n("MatMul", ["R", "input"], ["T1"]),                             # [1,10,30,30]
    n("MatMul", ["T1", "CT"], ["Y"]),                                # [1,10,30,30]
    # corner fix: out = Y + M*K
    n("Add", ["e0_col", "u_col"], ["a_col"]),                        # rows 0 & h+1
    n("Add", ["e0_row", "uc_row"], ["b_row"]),                       # cols 0 & w+1
    n("MatMul", ["a_col", "b_row"], ["M"]),                          # 4 corners
    n("Mul", ["M", "K"], ["adj"]),                                   # [1,10,30,30]
    n("Add", ["Y", "adj"], ["output"]),
]

graph = helper.make_graph(
    nodes, "task114",
    [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 10
onnx.checker.check_model(model)
out_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task114.onnx"
onnx.save(model, out_path)
print("saved", out_path, "params:", sum(np.prod(t.dims or [1]) for t in inits))
