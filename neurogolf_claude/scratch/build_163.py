"""Build compact ONNX for task163.

Rule (verified 267/267): the 11x11 grid is a 3x3 array of 3x3 panels separated by
color-5 lines (rows/cols 3,7). Exactly one cell holds color 4; it lives in some
panel (the KEY panel) at within-panel position (tr,tc). The output is all-zero
(separators kept) except the KEY panel's content copied into the panel SLOT (tr,tc).

Implementation: collapse one-hot input to a color grid g (excluded), crop to 11x11,
find the 4 with Equal, reduce to row/col onehots, build small separable selection
operators, place the key panel via two MatMuls, overlay separators, then emit the
one-hot via Equal(gm, colors) AS the graph output (excluded).
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

    # ---- constant selection tables (tiny) ----
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

    # ---- color grid g30 (excluded) -> crop 11x11 -> f16 ----
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))  # f32 [1,1,30,30]
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g11f"]))
    nodes.append(helper.make_node("Cast", ["g11f"], ["gf"], to=F16))    # f16 [1,1,11,11]
    # squeeze to [11,11] for matmuls
    inits.append(ci("ax01", [0, 1]))
    nodes.append(helper.make_node("Squeeze", ["gf", "ax01"], ["g"]))    # f16 [11,11]

    # ---- find the 4: row / col onehots ----
    inits.append(c16("four", 4.0))
    nodes.append(helper.make_node("Equal", ["g", "four"], ["m4b"]))     # bool [11,11]
    nodes.append(helper.make_node("Cast", ["m4b"], ["m4"], to=F16))     # f16 [11,11]
    nodes.append(helper.make_node("ReduceMax", ["m4"], ["rowmask"], axes=[1], keepdims=0))  # [11]
    nodes.append(helper.make_node("ReduceMax", ["m4"], ["colmask"], axes=[0], keepdims=0))  # [11]

    # ---- selector vectors (each [3]) ----
    # need rowmask as [1,11] for matmul -> reshape
    inits.append(ci("r1n", [1, S]))
    nodes.append(helper.make_node("Reshape", ["rowmask", "r1n"], ["rm2"]))  # [1,11]
    nodes.append(helper.make_node("Reshape", ["colmask", "r1n"], ["cm2"]))  # [1,11]
    nodes.append(helper.make_node("MatMul", ["rm2", "KR"], ["kR2"]))  # [1,3]
    nodes.append(helper.make_node("MatMul", ["rm2", "TR"], ["tR2"]))  # [1,3]
    nodes.append(helper.make_node("MatMul", ["cm2", "KR"], ["kC2"]))  # [1,3]
    nodes.append(helper.make_node("MatMul", ["cm2", "TR"], ["tC2"]))  # [1,3]

    # ---- build E_r/E_c [3,11] and P_r/P_c [11,3] from selector vectors ----
    # E_r = (kR2[1,3] @ EConst[3,33]) -> [1,33] -> reshape [3,11]
    inits.append(ci("e3n", [3, S]))
    inits.append(ci("p3n", [S, 3]))
    nodes.append(helper.make_node("MatMul", ["kR2", "EConst"], ["Er_f"]))
    nodes.append(helper.make_node("Reshape", ["Er_f", "e3n"], ["Er"]))   # [3,11]
    nodes.append(helper.make_node("MatMul", ["kC2", "EConst"], ["Ec_f"]))
    nodes.append(helper.make_node("Reshape", ["Ec_f", "e3n"], ["Ec"]))   # [3,11]
    nodes.append(helper.make_node("MatMul", ["tR2", "PConst"], ["Pr_f"]))
    nodes.append(helper.make_node("Reshape", ["Pr_f", "p3n"], ["Pr"]))   # [11,3]
    nodes.append(helper.make_node("MatMul", ["tC2", "PConst"], ["Pc_f"]))
    nodes.append(helper.make_node("Reshape", ["Pc_f", "p3n"], ["Pc"]))   # [11,3]

    # ---- panel = Er @ g @ Ec^T ; content = Pr @ panel @ Pc^T ----
    nodes.append(helper.make_node("MatMul", ["Er", "g"], ["keyrows"]))   # [3,11]
    # Ec^T: transpose Ec[3,11] -> [11,3]
    nodes.append(helper.make_node("Transpose", ["Ec"], ["EcT"], perm=[1, 0]))  # [11,3]
    nodes.append(helper.make_node("MatMul", ["keyrows", "EcT"], ["panel"]))    # [3,3]
    nodes.append(helper.make_node("MatMul", ["Pr", "panel"], ["pc1"]))         # [11,3]
    nodes.append(helper.make_node("Transpose", ["Pc"], ["PcT"], perm=[1, 0]))  # [3,11]
    nodes.append(helper.make_node("MatMul", ["pc1", "PcT"], ["content"]))      # [11,11]

    # ---- content -> uint8, overlay separators in uint8 (where g==5 -> 5) ----
    nodes.append(helper.make_node("Cast", ["content"], ["contu"], to=U8))  # u8 [11,11]
    inits.append(c16("five", 5.0))
    inits.append(cu("five8", 5))
    nodes.append(helper.make_node("Equal", ["g", "five"], ["sepb"]))    # bool [11,11]
    nodes.append(helper.make_node("Where", ["sepb", "five8", "contu"], ["gmu"]))  # u8 [11,11]

    # ---- reshape to [1,1,11,11], pad 30x30 sentinel 255, Equal->one-hot output ----
    inits.append(ci("r4n", [1, 1, S, S]))
    nodes.append(helper.make_node("Reshape", ["gmu", "r4n"], ["gm4"]))  # u8 [1,1,11,11]
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
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task163.onnx")
