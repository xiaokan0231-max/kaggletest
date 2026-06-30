"""task006 variant V5: opset 9, single Slice + dilated Conv + Pad (3 nodes).

  S = Slice(input, ch1, rows 0:3, cols 0:7)   [1,1,3,7]  (0/1 left+right masks)
  C = Conv(S, kernel [3,1,1,2], dilation (1,4), bias)    [1,3,3,3]
      taps: A = S[r,c], B = S[r,c+4]
      ch0 = 1.5 - A - B   (>0 iff A+B<=1, i.e. not AND)
      ch1 = 0
      ch2 = A + B - 1     (>0 iff A=B=1, i.e. AND)
  output = Pad(C, end ch+7 H+27 W+27)         [1,10,30,30]

Initializers: W (6) + bias (3) -> 9 params.
Strict >0 check: AND cell ch2=1>0, ch0=-0.5<=0; bg cell in 3x3 ch0>=0.5>0,
ch2<=0; outside 3x3 all 0<=0.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/scratch/task006/task006_v5.onnx"

W = np.array([[[[-1.0, -1.0]]],   # ch0: -(A+B)
              [[[0.0, 0.0]]],     # ch1: 0
              [[[1.0, 1.0]]]],    # ch2: A+B
             dtype=np.float32)
Bb = np.array([1.5, 0.0, -1.0], dtype=np.float32)

inits = [
    helper.make_tensor("W", TensorProto.FLOAT, [3, 1, 1, 2], W),
    helper.make_tensor("Bb", TensorProto.FLOAT, [3], Bb),
]

nodes = [
    helper.make_node("Slice", ["input"], ["S"], axes=[1, 2, 3], starts=[1, 0, 0], ends=[2, 3, 7]),
    helper.make_node("Conv", ["S", "W", "Bb"], ["C"], dilations=[1, 4]),
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
