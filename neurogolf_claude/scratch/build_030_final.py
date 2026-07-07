"""Build compact ONNX for task030 (matmul-shift, bool-assembled).

Rule (verified 266/266): colors {1,2,4} in disjoint columns. Color 1 is the
anchor (does not move). Each object shifts vertically so its topmost row aligns
with color-1's topmost row.

Strategy (gather-free, all interior tensors f16/bool, cropped to 10x10):
  * Slice the input one-hot per needed channel (0,1,2,4) at 10x10 -> f32 [1,1,10,10].
  * Per object channel c: top_c = min row containing the object; delta = top_c - top_1.
  * Vertical shift via MatMul with a one-hot shift matrix S = (DIFF == delta),
    DIFF[i,j]=j-i, so (S @ ch)[i] = ch[i+delta] (shift up by delta).
  * painted = OR of shifted object channels; background out0 = ingrid AND NOT painted.
  * Assemble channels [0,1,2,3(zero),4] as bool, Concat, Pad to [1,10,30,30] = output.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL
S = 10


def cf16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci64(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def ci32(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int32), name=name)


def build(path):
    nodes, inits = [], []

    # DIFF[i,j] = j - i  (int64 [10,10]); row index column; BIG sentinel
    DIFF = (np.arange(S)[None, :] - np.arange(S)[:, None]).astype(np.int32)
    inits.append(ci32("DIFF", DIFF))
    inits.append(ci32("ROWIDX", np.arange(S, dtype=np.int32).reshape(1, 1, S, 1)))
    inits.append(ci32("BIG", 99))
    inits.append(ci64("axes4", [0, 1, 2, 3]))

    # slice the 4 needed channels at 10x10
    def slice_ch(c):
        inits.append(ci64("st" + str(c), [0, c, 0, 0]))
        inits.append(ci64("en" + str(c), [1, c + 1, S, S]))
        nodes.append(helper.make_node("Slice", ["input", "st" + str(c), "en" + str(c), "axes4"], ["ch" + str(c)]))  # f32 [1,1,10,10]

    for c in (0, 1, 2, 4):
        slice_ch(c)

    # per-object-channel top row (on f32) and shifted channel via matmul
    tops = {}
    for c in (1, 2, 4):
        cn = str(c)
        nodes.append(helper.make_node("ReduceMax", ["ch" + cn], ["rh" + cn], axes=[3], keepdims=1))  # f32 [1,1,10,1]
        nodes.append(helper.make_node("Cast", ["rh" + cn], ["rhb" + cn], to=BOOL))
        nodes.append(helper.make_node("Where", ["rhb" + cn, "ROWIDX", "BIG"], ["rows" + cn]))  # i64 [1,1,10,1]
        nodes.append(helper.make_node("ReduceMin", ["rows" + cn], ["top" + cn], axes=[2], keepdims=1))  # i64 [1,1,1,1]
        tops[c] = "top" + cn

    # f16 copies of object channels for matmul
    for c in (1, 2, 4):
        nodes.append(helper.make_node("Cast", ["ch" + str(c)], ["chf" + str(c)], to=F16))  # f16 [1,1,10,10]

    inits.append(cf16("zf16", 0.0))
    painted_prev = None
    out_chan = {}
    for c in (1, 2, 4):
        cn = str(c)
        nodes.append(helper.make_node("Sub", [tops[c], tops[1]], ["delta" + cn]))   # i64 [1,1,1,1]
        nodes.append(helper.make_node("Equal", ["DIFF", "delta" + cn], ["Sb" + cn]))  # bool [10,10]
        nodes.append(helper.make_node("Cast", ["Sb" + cn], ["Sf" + cn], to=F16))      # f16 [10,10]
        nodes.append(helper.make_node("MatMul", ["Sf" + cn, "chf" + cn], ["outp" + cn]))  # f16 [1,1,10,10]
        # bool channel
        nodes.append(helper.make_node("Greater", ["outp" + cn, "zf16"], ["ob" + cn]))  # bool [1,1,10,10]
        out_chan[c] = "ob" + cn
        if painted_prev is None:
            painted_prev = "ob" + cn
        else:
            nodes.append(helper.make_node("Or", [painted_prev, "ob" + cn], ["pacc" + cn]))
            painted_prev = "pacc" + cn
    painted = painted_prev  # bool [1,1,10,10]

    # ingrid = (ch0+ch1+ch2+ch4) > 0
    nodes.append(helper.make_node("Sum", ["ch0", "ch1", "ch2", "ch4"], ["insidef"]))  # f32 [1,1,10,10]
    inits.append(numpy_helper.from_array(np.asarray(0.0, np.float32), "zf32"))
    nodes.append(helper.make_node("Greater", ["insidef", "zf32"], ["ingrid"]))  # bool
    # out0 = ingrid XOR painted  (painted subset of ingrid) -> ingrid AND NOT painted
    nodes.append(helper.make_node("Xor", ["ingrid", painted], ["ob0"]))  # bool [1,1,10,10]

    # assemble channels 0,1,2,3(zero),4 as bool -> [1,5,10,10]
    inits.append(numpy_helper.from_array(np.zeros((1, 1, S, S), np.bool_), "zero_ch"))
    nodes.append(helper.make_node("Concat", ["ob0", out_chan[1], out_chan[2], "zero_ch", out_chan[4]],
                                  ["out5"], axis=1))  # bool [1,5,10,10]
    # pad: channel dim +5 (to 10), spatial +20 each -> [1,10,30,30]
    inits.append(ci64("pads30", [0, 0, 0, 0, 0, 5, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "false_b"))
    nodes.append(helper.make_node("Pad", ["out5", "pads30", "false_b"], ["output"]))  # bool [1,10,30,30]

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
