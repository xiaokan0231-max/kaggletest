"""task006 variant V2: V1 graph (Mul+Conv) with opset 9 attribute Slice/Pad.

  A = Slice(input, ch1, rows 0:3, cols 0:3)   [1,1,3,3]
  B = Slice(input, ch1, rows 0:3, cols 4:7)   [1,1,3,3]
  M = Mul(A, B)                               AND
  C = Conv(M, W=[-1,0,1], B=[1,0,0])          [1,3,3,3] = [1-M, 0, M]
  output = Pad(C, end ch+7 H+27 W+27)         [1,10,30,30]

Initializers: Conv W (3) + B (3) -> 6 params.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/scratch/task006/task006_v2.onnx"

inits = [
    helper.make_tensor("W", TensorProto.FLOAT, [3, 1, 1, 1], np.array([-1.0, 0.0, 1.0], dtype=np.float32)),
    helper.make_tensor("Bb", TensorProto.FLOAT, [3], np.array([1.0, 0.0, 0.0], dtype=np.float32)),
]

nodes = [
    helper.make_node("Slice", ["input"], ["A"], axes=[1, 2, 3], starts=[1, 0, 0], ends=[2, 3, 3]),
    helper.make_node("Slice", ["input"], ["B"], axes=[1, 2, 3], starts=[1, 0, 4], ends=[2, 3, 7]),
    helper.make_node("Mul", ["A", "B"], ["M"]),
    helper.make_node("Conv", ["M", "W", "Bb"], ["C"]),
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
