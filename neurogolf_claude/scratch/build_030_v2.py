"""Build compact ONNX for task030.

Rule (verified 266/266): exactly 3 colored objects, colors {1,2,4}, in disjoint
columns. Color 1 is the anchor and does NOT move. Each object shifts vertically
(rigidly) so its topmost row aligns with color-1's topmost row.

All grids fit in the top-left 10x10, so interior work is cropped to S=10x10.

Compact ONNX:
  1. ONE Conv collapses one-hot input -> value grid gs [1,1,30,30] f32 with
     weight [1,2,...,10]: out-of-grid -> 0, in-grid color k -> k+1. (input excl.)
  2. Slice gs to 10x10, cast uint8 (gv). ingrid = gv>=1 (fixed grid rectangle).
  3. For shifted-color sc in {2,3,5}: objtop = min row containing sc; atop =
     color-1's objtop; coltop[j] = sum objtop_c*colhas_c[j].
  4. offset[j] = M - atop + coltop[j] in [1,15]; src[r,j]=r+offset in [0,HP).
     Gather from gp = gv zero-padded to height HP=26 -> no validity ops.
  5. gm = max(gathered,1) (padding->bg 1); where(ingrid,gm,0); pad to 30x30 with
     sentinel 0. Equal(gm30,[1..10]) IS the output one-hot (excluded).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
S = 10
M = 8
HP = S + 2 * M  # 26


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci32(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int32), name=name)


def ci64(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # 1. Conv collapse -> gs f32 [1,1,30,30]
    inits.append(cf("Wcol", np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["gs"]))
    # 2. crop to 10x10 and cast uint8
    inits += [ci64("cs", [0, 0]), ci64("ce", [S, S]), ci64("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["gs", "cs", "ce", "ca"], ["gsc"]))  # f32 [1,1,10,10]
    nodes.append(helper.make_node("Cast", ["gsc"], ["gv"], to=U8))               # u8  [1,1,10,10]
    inits.append(cu("one_u8", 1))
    nodes.append(helper.make_node("GreaterOrEqual", ["gv", "one_u8"], ["ingrid"]))  # bool

    # 3. per-color objtop / coltop (tiny tensors)
    inits.append(ci32("rows1d", np.arange(S, dtype=np.int32)))
    inits.append(ci64("rshape", [1, 1, S, 1]))
    nodes.append(helper.make_node("Reshape", ["rows1d", "rshape"], ["rrow"]))  # i32 [1,1,10,1]
    inits.append(ci32("BIG", 1000))

    coltop_prev = None
    for color, sc in ((1, 2), (2, 3), (4, 5)):
        cn = str(color)
        inits.append(cu("sc" + cn, sc))
        nodes.append(helper.make_node("Equal", ["gv", "sc" + cn], ["mc" + cn]))   # bool [1,1,10,10]
        nodes.append(helper.make_node("Cast", ["mc" + cn], ["mc_u" + cn], to=U8))  # u8
        nodes.append(helper.make_node("ReduceMax", ["mc_u" + cn], ["rowhas" + cn], axes=[3], keepdims=1))
        nodes.append(helper.make_node("Cast", ["rowhas" + cn], ["rowhasb" + cn], to=BOOL))
        nodes.append(helper.make_node("Where", ["rowhasb" + cn, "rrow", "BIG"], ["rsel" + cn]))
        nodes.append(helper.make_node("ReduceMin", ["rsel" + cn], ["objtop" + cn], axes=[2], keepdims=1))  # [1,1,1,1]
        nodes.append(helper.make_node("ReduceMax", ["mc_u" + cn], ["colhasu" + cn], axes=[2], keepdims=1))
        nodes.append(helper.make_node("Cast", ["colhasu" + cn], ["colhas" + cn], to=I32))
        nodes.append(helper.make_node("Mul", ["objtop" + cn, "colhas" + cn], ["contrib" + cn]))  # [1,1,1,10]
        if coltop_prev is None:
            coltop_prev = "contrib" + cn
        else:
            nodes.append(helper.make_node("Add", [coltop_prev, "contrib" + cn], ["acc" + cn]))
            coltop_prev = "acc" + cn
    coltop = coltop_prev

    # 4. offset = M - atop + coltop -> [1,1,1,10]
    inits.append(ci32("Mc", M))
    nodes.append(helper.make_node("Sub", ["Mc", "objtop1"], ["m_atop"]))
    nodes.append(helper.make_node("Add", [coltop, "m_atop"], ["offset"]))
    nodes.append(helper.make_node("Add", ["rrow", "offset"], ["src"]))  # i32 [1,1,10,10]

    # padded source gp = gv zero-padded vertically to HP
    inits.append(ci64("pads", [0, 0, M, 0, 0, 0, M, 0]))
    inits.append(cu("zero_u8", 0))
    nodes.append(helper.make_node("Pad", ["gv", "pads", "zero_u8"], ["gp"]))  # u8 [1,1,26,10]
    nodes.append(helper.make_node("GatherElements", ["gp", "src"], ["gathered"], axis=2))  # u8 [1,1,10,10]

    # 5. gm = max(gathered,1); out-of-grid -> 0
    nodes.append(helper.make_node("Max", ["gathered", "one_u8"], ["gm1"]))
    nodes.append(helper.make_node("Where", ["ingrid", "gm1", "zero_u8"], ["gm"]))  # u8 [1,1,10,10]
    # one-hot at 10x10 then Pad to [1,10,30,30] as the graph output (excluded).
    inits.append(cu("colors", np.arange(1, 11, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["oh10"]))  # bool [1,10,10,10] interior
    inits.append(ci64("pad30", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "false_b"))
    nodes.append(helper.make_node("Pad", ["oh10", "pad30", "false_b"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t030", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1d_task030.onnx")
