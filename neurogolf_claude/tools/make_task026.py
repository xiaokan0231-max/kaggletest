"""Build ONNX network for task026.

Rule: input 5x7, col 3 is a separator of 1s. Left half = cols 0..2,
right half = cols 4..6, both in {0,9}. Output 5x3: cell = 8 iff both
halves are 0 at that position (NOR of 9-masks), else 0.

Graph (4 nodes):
  1. Slice   : input[:, 9:10, 0:5, 0:7]            -> s   [1,1,5,7]
  2. Conv    : kernel [1,1,1,5] = [1,0,0,0,1]      -> sum [1,1,5,3]
               sum[r,c] = s[r,c] + s[r,c+4]  (left 9-mask + right 9-mask)
  3. Conv1x1 : W [9,1,1,1], B [9]; ch0 = sum, ch8 = 1 - sum -> m [1,9,5,3]
               (no Sign needed: sum in {0,1,2})
  4. Pad     : embed at top-left of [1,10,30,30], appending zero ch9.

Verdict semantics check (judgement is elementwise output > 0):
  8-cell (NOR): sum = 0 -> ch0 = 0 <= 0, ch8 = 1 > 0, others 0.
  0-cell      : sum in {1,2} -> ch0 > 0, ch8 = 1 - sum in {0,-1} <= 0.
  outside 5x3 region / channel 9: all 0 <= 0 (Pad fills 0).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task026.onnx"


def main():
    inits = []

    def init(name, arr):
        inits.append(onnx.numpy_helper.from_array(arr, name=name))
        return name

    # Slice channel 9, rows 0:5, cols 0:7
    init("sl_starts", np.array([9, 0, 0], dtype=np.int64))
    init("sl_ends", np.array([10, 5, 7], dtype=np.int64))
    init("sl_axes", np.array([1, 2, 3], dtype=np.int64))

    # Conv1: width-5 kernel summing col c and col c+4
    w1 = np.zeros((1, 1, 1, 5), dtype=np.float32)
    w1[0, 0, 0, 0] = 1.0
    w1[0, 0, 0, 4] = 1.0
    init("w1", w1)

    # Conv2: 1x1, scatter into channels: ch0 = sum, ch8 = 1 - sum
    w2 = np.zeros((9, 1, 1, 1), dtype=np.float32)
    w2[0, 0, 0, 0] = 1.0
    w2[8, 0, 0, 0] = -1.0
    init("w2", w2)
    b2 = np.zeros(9, dtype=np.float32)
    b2[8] = 1.0
    init("b2", b2)

    # Pad [1,9,5,3] -> [1,10,30,30] at top-left (1 extra zero channel at end)
    init("pads", np.array([0, 0, 0, 0, 0, 1, 25, 27], dtype=np.int64))

    nodes = [
        helper.make_node("Slice", ["input", "sl_starts", "sl_ends", "sl_axes"], ["s"]),
        helper.make_node("Conv", ["s", "w1"], ["sum"]),
        helper.make_node("Conv", ["sum", "w2", "b2"], ["m"]),
        helper.make_node("Pad", ["m", "pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(
        nodes,
        "task026",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    n_params = sum(int(np.prod(t.dims)) if t.dims else 1 for t in inits)
    print(f"saved {OUT}, initializer elements = {n_params}")


if __name__ == "__main__":
    main()
