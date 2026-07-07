"""task006 variant V4: opset 9, attribute-based Slice/Pad, no Conv.

  A   = Slice(input, ch1, rows 0:3, cols 0:3)   [1,1,3,3]
  B   = Slice(input, ch1, rows 0:3, cols 4:7)   [1,1,3,3]
  M   = Mul(A, B)                               AND
  ch0 = Sub(1.0, M)                             background
  P1  = Pad(M, ch begin 1)                      [1,2,3,3] = [0, M]
  C   = Concat([ch0, P1], axis=1)               [1,3,3,3] = [1-M, 0, M]
  output = Pad(C, end ch+7 H+27 W+27)           [1,10,30,30]

Only initializer: scalar 1.0 -> 1 param.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/scratch/task006/task006_v4.onnx"

inits = [helper.make_tensor("one", TensorProto.FLOAT, [], np.array([1.0], dtype=np.float32))]

nodes = [
    helper.make_node("Slice", ["input"], ["A"], axes=[1, 2, 3], starts=[1, 0, 0], ends=[2, 3, 3]),
    helper.make_node("Slice", ["input"], ["B"], axes=[1, 2, 3], starts=[1, 0, 4], ends=[2, 3, 7]),
    helper.make_node("Mul", ["A", "B"], ["M"]),
    helper.make_node("Sub", ["one", "M"], ["ch0"]),
    helper.make_node("Pad", ["M"], ["P1"], pads=[0, 1, 0, 0, 0, 0, 0, 0]),
    helper.make_node("Concat", ["ch0", "P1"], ["C"], axis=1),
    helper.make_node("Pad", ["C"], ["output"], pads=[0, 0, 0, 0, 0, 7, 27, 27]),
]

graph = helper.make_graph(
    nodes,
    "task006",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("saved", OUT)
