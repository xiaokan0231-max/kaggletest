"""task163 v5: Einsum-fused selection (3-input Einsum drops matmul/reshape/transpose
intermediates) on a 2D [11,11] grid.

Rule (verified 267/267): 3x3 array of 3x3 panels (sep color-5 at rows/cols 3,7).
One cell holds color 4 in the KEY panel at within-panel pos (tr,tc); output = the
KEY panel copied into panel SLOT (tr,tc), zeros elsewhere, separators kept.

content = Pr @ (Er @ g @ Ec^T) @ Pc^T where the small selection matrices come from
fused Einsums of the 4-onehot. Final one-hot emitted by Equal(pad(color), colors)
AS the graph output (excluded from memory).
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
    inits += [c16("EConst", EConst), c16("PConst", PConst), c16("KR", KR), c16("TR", TR)]

    # color grid -> crop 11x11 -> f16 2D
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))   # f32 [1,1,30,30]
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g11f"]))  # f32 [1,1,11,11]
    inits.append(ci("ax01", [0, 1]))
    nodes.append(helper.make_node("Squeeze", ["g11f", "ax01"], ["g2f"]))  # f32 [11,11]
    nodes.append(helper.make_node("Cast", ["g2f"], ["g"], to=F16))        # f16 [11,11]

    # find the 4 -> row/col onehots
    inits.append(c16("four", 4.0))
    nodes.append(helper.make_node("Equal", ["g", "four"], ["m4b"]))       # bool [11,11]
    nodes.append(helper.make_node("Cast", ["m4b"], ["m4"], to=F16))       # f16 [11,11]
    nodes.append(helper.make_node("ReduceMax", ["m4"], ["rowv"], axes=[1], keepdims=0))  # [11]
    nodes.append(helper.make_node("ReduceMax", ["m4"], ["colv"], axes=[0], keepdims=0))  # [11]

    # fused selection matrices (one Einsum each, no intermediates)
    nodes.append(helper.make_node("Einsum", ["rowv", "KR", "EConst"], ["Er"], equation="r,rk,kia->ia"))  # [3,11]
    nodes.append(helper.make_node("Einsum", ["rowv", "TR", "PConst"], ["Pr"], equation="r,rt,tAi->Ai"))  # [11,3]
    nodes.append(helper.make_node("Einsum", ["colv", "KR", "EConst"], ["Ec"], equation="r,rk,kia->ia"))  # [3,11]
    nodes.append(helper.make_node("Einsum", ["colv", "TR", "PConst"], ["Pc"], equation="r,rt,tAi->Ai"))  # [11,3]

    # panel = Er @ g @ Ec^T ; content = Pr @ panel @ Pc^T (each a single Einsum)
    nodes.append(helper.make_node("Einsum", ["Er", "g", "Ec"], ["panel"], equation="ia,ab,jb->ij"))  # [3,3]
    nodes.append(helper.make_node("Einsum", ["Pr", "panel", "Pc"], ["content"], equation="Ai,ij,Bj->AB"))  # [11,11]

    # content -> uint8, overlay separators (g==5 -> 5)
    nodes.append(helper.make_node("Cast", ["content"], ["contu"], to=U8))   # u8 [11,11]
    inits.append(c16("five", 5.0))
    inits.append(cu("five8", 5))
    nodes.append(helper.make_node("Equal", ["g", "five"], ["sepb"]))        # bool [11,11]
    nodes.append(helper.make_node("Where", ["sepb", "five8", "contu"], ["gmu"]))  # u8 [11,11]

    # reshape [1,1,11,11], pad 30x30 sentinel 255, Equal -> one-hot output
    inits.append(ci("r4n", [1, 1, S, S]))
    nodes.append(helper.make_node("Reshape", ["gmu", "r4n"], ["gm4"]))      # u8 [1,1,11,11]
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Pad", ["gm4", "pads", "sent"], ["gm30"]))  # u8 [1,1,30,30]
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
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task163_v5.onnx")
