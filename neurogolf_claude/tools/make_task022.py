"""Build task022.onnx.

Rule: input 11x11 has 3-4 small shapes each containing exactly one gray (5)
anchor; output 3x3 overlays the 3x3 input neighborhoods centered at every
anchor (non-zero contributions never conflict; center is always 5).

ONNX realization:
  xs = input[:, :, :11, :11]  (grid is always 11x11; everything nonzero
       lives there, so all sums below are exact on the slice)
  m  = xs channel 5  (anchor indicator, [1,1,11,11])
  M9 = Conv(m, K) with K[k,0,1-dr,1-dc]=1 for offset k=(dr+1)*3+(dc+1),
       pads=1  ->  M9[k][u,v] = m[u-dr, v-dc]            ([1,9,11,11])
  S  = Gemm(X=[10,121], M=[9,121], transB=1) = [10,9]
       S[ch,k] = sum_{anchors (r,c)} x[ch, r+dr, c+dc]   (counts >= 0)
  channel-0 must be rebuilt: ch0 = 1 - sum_{ch=1..9} S[ch,:]
       (colored cell -> <=0, background cell -> 1 > 0)
  out = Concat(ch0, S[1:10]) -> Reshape [1,10,3,3] -> zero-Pad to [1,10,30,30]
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OUT_PATH = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task022.onnx"


def build():
    # 9 delta kernels: k = (dr+1)*3 + (dc+1), delta at (1-dr, 1-dc)
    K = np.zeros((9, 1, 3, 3), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            k = (dr + 1) * 3 + (dc + 1)
            K[k, 0, 1 - dr, 1 - dc] = 1.0

    inits = [
        numpy_helper.from_array(K, "K"),
        numpy_helper.from_array(np.array([0, 0], dtype=np.int64), "st00"),
        numpy_helper.from_array(np.array([11, 11], dtype=np.int64), "en1111"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "ax23"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), "st5"),
        numpy_helper.from_array(np.array([6], dtype=np.int64), "en6"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
        numpy_helper.from_array(np.array([10, 121], dtype=np.int64), "shpX"),
        numpy_helper.from_array(np.array([9, 121], dtype=np.int64), "shpM"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "st1"),
        numpy_helper.from_array(np.array([10], dtype=np.int64), "en10"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "ax0"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(np.array([1, 10, 3, 3], dtype=np.int64), "shpO"),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64), "pads"
        ),
    ]

    nodes = [
        # xs = input[:, :, :11, :11]  (the actual grid region)
        helper.make_node("Slice", ["input", "st00", "en1111", "ax23"], ["xs"]),
        # m = xs[:, 5:6]  (anchor indicator)
        helper.make_node("Slice", ["xs", "st5", "en6", "ax1"], ["m"]),
        # 9 shifted copies of the anchor mask
        helper.make_node("Conv", ["m", "K"], ["M9"], pads=[1, 1, 1, 1]),
        helper.make_node("Reshape", ["xs", "shpX"], ["X"]),
        helper.make_node("Reshape", ["M9", "shpM"], ["M"]),
        # S[ch,k] = #anchors whose offset-k neighbor has color ch
        helper.make_node("Gemm", ["X", "M"], ["S"], transB=1),
        helper.make_node("Slice", ["S", "st1", "en10", "ax0"], ["Scol"]),
        helper.make_node("ReduceSum", ["Scol", "ax0"], ["colsum"], keepdims=1),
        # background channel: >0 iff no colored contribution
        helper.make_node("Sub", ["one", "colsum"], ["ch0"]),
        helper.make_node("Concat", ["ch0", "Scol"], ["out10"], axis=0),
        helper.make_node("Reshape", ["out10", "shpO"], ["out4"]),
        helper.make_node("Pad", ["out4", "pads"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task022",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT_PATH)
    n_params = sum(int(np.prod(t.dims)) if t.dims else 1 for t in model.graph.initializer)
    print(f"saved {OUT_PATH}, initializer elements = {n_params}")


if __name__ == "__main__":
    build()
