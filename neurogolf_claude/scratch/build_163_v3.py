"""task163 v3: 4D-throughout to drop the squeeze duplicate.

Rule (verified 267/267): 3x3 array of 3x3 panels (sep color-5 at rows/cols 3,7).
One cell holds color 4 in the KEY panel at within-panel pos (tr,tc); output = the
KEY panel copied into panel SLOT (tr,tc), zeros elsewhere, separators kept.

Separable placement: content = Pr @ (Er @ g @ Ec^T) @ Pc^T, all batched [1,1,*,*].
Er/Ec extract the key panel's 3 rows/cols; Pr/Pc place them at the target slot.
Final one-hot emitted by Equal(padded-color-grid, colors) AS the graph output.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 11


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    EConst = np.zeros((3, 3, S), np.float16)
    for k in range(3):
        for ir in range(3):
            EConst[k, ir, 4 * k + ir] = 1
    PConst = np.zeros((3, S, 3), np.float16)
    for t in range(3):
        for ir in range(3):
            PConst[t, 4 * t + ir, ir] = 1
    KR = np.zeros((S, 3), np.float16)
    TR = np.zeros((S, 3), np.float16)
    for R in (0, 1, 2, 4, 5, 6, 8, 9, 10):
        KR[R, R // 4] = 1
        TR[R, R % 4] = 1
    inits += [
        numpy_helper.from_array(EConst.reshape(3, 3 * S), "EConst"),  # [3,33]
        numpy_helper.from_array(PConst.reshape(3, S * 3), "PConst"),  # [3,33]
        c16("KR", KR), c16("TR", TR),                                 # [11,3]
    ]

    # color grid -> crop 11x11 -> f16, all kept 4D [1,1,*,*]
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))   # f32 [1,1,30,30]
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g11f"]))  # f32 [1,1,11,11]
    nodes.append(helper.make_node("Cast", ["g11f"], ["g"], to=F16))      # f16 [1,1,11,11]

    # find the 4 -> row/col onehots (mask kept uint8: 121B vs 242B for f16)
    inits.append(c16("four", 4.0))
    nodes.append(helper.make_node("Equal", ["g", "four"], ["m4b"]))      # bool [1,1,11,11]
    nodes.append(helper.make_node("Cast", ["m4b"], ["m4"], to=U8))       # u8 [1,1,11,11]
    nodes.append(helper.make_node("ReduceMax", ["m4"], ["rmcolu"], axes=[3], keepdims=1))  # u8 [1,1,11,1]
    nodes.append(helper.make_node("ReduceMax", ["m4"], ["rmrowu"], axes=[2], keepdims=1))  # u8 [1,1,1,11]
    # cast the tiny onehot vectors to f16 for the MatMuls
    nodes.append(helper.make_node("Cast", ["rmcolu"], ["rmcol"], to=F16))   # [1,1,11,1]
    nodes.append(helper.make_node("Cast", ["rmrowu"], ["rmrow"], to=F16))   # [1,1,1,11]
    nodes.append(helper.make_node("Transpose", ["rmcol"], ["rowv"], perm=[0, 1, 3, 2]))  # [1,1,1,11]

    # selector vectors [1,1,1,3]
    nodes.append(helper.make_node("MatMul", ["rowv", "KR"], ["kR"]))     # [1,1,1,3]
    nodes.append(helper.make_node("MatMul", ["rowv", "TR"], ["tR"]))
    nodes.append(helper.make_node("MatMul", ["rmrow", "KR"], ["kC"]))
    nodes.append(helper.make_node("MatMul", ["rmrow", "TR"], ["tC"]))

    # Er/Ec [1,1,3,11], Pr/Pc [1,1,11,3]
    inits.append(ci("e4n", [1, 1, 3, S]))
    inits.append(ci("p4n", [1, 1, S, 3]))
    nodes.append(helper.make_node("MatMul", ["kR", "EConst"], ["Er_f"]))   # [1,1,1,33]
    nodes.append(helper.make_node("Reshape", ["Er_f", "e4n"], ["Er"]))     # [1,1,3,11]
    nodes.append(helper.make_node("MatMul", ["kC", "EConst"], ["Ec_f"]))
    nodes.append(helper.make_node("Reshape", ["Ec_f", "e4n"], ["Ec"]))     # [1,1,3,11]
    nodes.append(helper.make_node("MatMul", ["tR", "PConst"], ["Pr_f"]))
    nodes.append(helper.make_node("Reshape", ["Pr_f", "p4n"], ["Pr"]))     # [1,1,11,3]
    nodes.append(helper.make_node("MatMul", ["tC", "PConst"], ["Pc_f"]))
    nodes.append(helper.make_node("Reshape", ["Pc_f", "p4n"], ["Pc"]))     # [1,1,11,3]

    # panel = Er @ g @ Ec^T ; content = Pr @ panel @ Pc^T
    nodes.append(helper.make_node("MatMul", ["Er", "g"], ["keyrows"]))     # [1,1,3,11]
    nodes.append(helper.make_node("Transpose", ["Ec"], ["EcT"], perm=[0, 1, 3, 2]))  # [1,1,11,3]
    nodes.append(helper.make_node("MatMul", ["keyrows", "EcT"], ["panel"]))   # [1,1,3,3]
    nodes.append(helper.make_node("MatMul", ["Pr", "panel"], ["pc1"]))        # [1,1,11,3]
    nodes.append(helper.make_node("Transpose", ["Pc"], ["PcT"], perm=[0, 1, 3, 2]))  # [1,1,3,11]
    nodes.append(helper.make_node("MatMul", ["pc1", "PcT"], ["content"]))     # [1,1,11,11]

    # content -> uint8, overlay separators (g==5 -> 5)
    nodes.append(helper.make_node("Cast", ["content"], ["contu"], to=U8))     # u8 [1,1,11,11]
    inits.append(c16("five", 5.0))
    inits.append(cu("five8", 5))
    nodes.append(helper.make_node("Equal", ["g", "five"], ["sepb"]))          # bool [1,1,11,11]
    nodes.append(helper.make_node("Where", ["sepb", "five8", "contu"], ["gmu"]))  # u8 [1,1,11,11]

    # pad 30x30 sentinel 255, Equal -> one-hot output
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Pad", ["gmu", "pads", "sent"], ["gm30"]))  # u8 [1,1,30,30]
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))   # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t163", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task163_v3.onnx")
