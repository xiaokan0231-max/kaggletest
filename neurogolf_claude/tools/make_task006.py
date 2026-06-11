"""Build task006.onnx (final: variant V5, official 192 bytes + 9 params = 19.697 pts).

Rule (fully static): input is 3x7, col 3 = separator (color 5).
Left half cols 0:3 and right half cols 4:7 are 0/1 masks (color 1).
Output 3x3: cell = color 2 where both halves nonzero (AND), else 0.

Key insight: with binary masks the AND is linear under the strict >0 readout:
  ch2 = A + B - 1   (>0 iff A=B=1)
  ch0 = 1.5 - A - B (>0 iff A+B<=1)
so one dilated Conv replaces Slice+Slice+Mul+Conv.

Graph (3 nodes, opset 9 so Slice/Pad use attributes -> zero int64 params):
  S = Slice(input, ch1, rows 0:3, cols 0:7)            [1,1,3,7]
  C = Conv(S, W [3,1,1,2], dilation (1,4), bias [3])   [1,3,3,3]
      taps A=S[r,c], B=S[r,c+4]; ch0=1.5-A-B, ch1=0, ch2=A+B-1
  output = Pad(C, end pads ch+7, H+27, W+27)           [1,10,30,30]

Strict >0 check: AND cell ch2=1>0, ch0=-0.5<=0; bg cell in 3x3 ch0>=0.5>0,
ch2<=0; outside 3x3 all channels 0<=0.

Variant history (all official-verified):
  V1 opset13 Slice+Slice+Mul+Conv+Pad, int64 initializers: 216B+30p = 19.495
  V2 opset9 same graph, attrs instead of initializers:      216B+6p  = 19.597
  V4 opset9 Mul+Sub+Pad+Concat+Pad (1 param):               324B+1p  = 19.216
  V5 opset9 single Slice + dilated Conv + Pad:              192B+9p  = 19.697  <- this
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task006.onnx"


def main():
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


if __name__ == "__main__":
    main()
