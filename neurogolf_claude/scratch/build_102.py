"""task102: enclosed solid-square holes (NxN all-zero blocks bounded by walls) -> color 2.

Single-conv detector per N: kernel on (N+2)x(N+2) window, interior NxN = +1, frame ring = -BIG.
Conv(b) == N^2  iff  interior all-zero (N^2 ones) AND frame all walls (no zeros) => perfect enclosed square.
Spread detector over its NxN block via MaxPool dilation; union over N; recolor those 0-cells to 2.
Final output: Where(cover, e2 one-hot, input) -> only [1,10,30,30] tensor is the excluded output.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

H = W = 30
OPSET = 18
IR = 10
BIG = 100.0  # frame penalty; any frame zero drives sum below N^2

nodes = []
inits = []


def C(name, arr):
    inits.append(nh.from_array(np.asarray(arr), name))
    return name


# 1. collapse one-hot -> color grid g [1,1,30,30] f32 (irreducible input cost)
C("Wc", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Conv", ["input", "Wc"], ["g"], kernel_shape=[1, 1]))

# 2. zero-mask b as f32 (1.0 where color 0). g in {0..9}; b = (g < 0.5)
C("half", np.array(0.5, np.float32))
nodes.append(helper.make_node("Less", ["g", "half"], ["b_bool"]))
nodes.append(helper.make_node("Cast", ["b_bool"], ["b"], to=TensorProto.FLOAT))

cover_terms = []
for N in [1, 2, 3, 4]:
    K = N + 2
    ker = -BIG * np.ones((1, 1, K, K), np.float32)
    ker[0, 0, 1:1 + N, 1:1 + N] = 1.0   # interior +1, frame -BIG
    C(f"k{N}", ker)
    # anchor detector at inner top-left (r,c): window covers rows[r-1..r+N], cols[c-1..c+N];
    # window top-left = (r-1,c-1) -> pad top/left 1, bottom/right N.
    nodes.append(helper.make_node(
        "Conv", ["b", f"k{N}"], [f"s{N}"],
        kernel_shape=[K, K], pads=[1, 1, N, N]))
    C(f"n2_{N}", np.array(float(N * N), np.float32))
    nodes.append(helper.make_node("Equal", [f"s{N}", f"n2_{N}"], [f"det{N}"]))  # bool at inner TL
    nodes.append(helper.make_node("Cast", [f"det{N}"], [f"detu{N}"], to=TensorProto.UINT8))
    # spread to NxN block (uint8 MaxPool, 900B): pad top/left N-1
    nodes.append(helper.make_node(
        "MaxPool", [f"detu{N}"], [f"cov{N}"],
        kernel_shape=[N, N], pads=[N - 1, N - 1, 0, 0], strides=[1, 1]))
    cover_terms.append(f"cov{N}")

# union of covers (uint8 in {0,1}); ==1 -> bool covered
nodes.append(helper.make_node("Max", cover_terms, ["cover"]))  # uint8 900B
C("one_u8", np.array(1, np.uint8))
nodes.append(helper.make_node("Equal", ["cover", "one_u8"], ["cover_bool"]))

# 3. output: keep input except covered cells -> color-2 one-hot
e2 = np.zeros((1, 10, 1, 1), np.float32)
e2[0, 2, 0, 0] = 1.0
C("e2", e2)
nodes.append(helper.make_node("Where", ["cover_bool", "e2", "input"], ["output"]))

graph = helper.make_graph(
    nodes, "task102",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H, W])],
    inits)
model = helper.make_model(graph, ir_version=IR,
                          opset_imports=[helper.make_opsetid("", OPSET)])
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b14_task102.onnx")
print("saved nodes=", len(nodes))
