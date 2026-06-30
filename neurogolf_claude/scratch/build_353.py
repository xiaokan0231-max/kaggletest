"""Build compact ONNX for task353.

Rule (verified 271/271): grid has exactly one cell of color 3 and one of color 4.
The 3 moves ONE step toward the 4: displacement = (sign(y4-y3), sign(x4-x3)).
The 4 stays. The old 3 cell becomes background 0; the new cell becomes 3.

Strategy: collapse the one-hot input to a color grid (excluded input cost) and
the masks via 1x1 Convs, reduce to row/col one-hot vectors (tiny), compute the
sign displacement, shift the 3 row/col one-hots by +/-1/0 via static Slice+Pad and
a scalar selection, outer-product to the new-3 mask, edit the uint8 color grid,
mark out-of-grid cells with sentinel 255, and let a single Equal vs colors 0..9 be
the graph 'output' (excluded one-hot [1,10,30,30]).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
B1 = TensorProto.INT8  # not used
H = W = 30


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # ---- collapse one-hot input to color grid + masks via 1x1 Convs (input excluded) ----
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["gin_f"]))   # f32 [1,1,30,30]

    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["valsum"]))  # f32 [1,1,30,30]

    w3 = np.zeros((1, 10, 1, 1), np.float32); w3[0, 3, 0, 0] = 1.0
    inits.append(cf("W3", w3))
    nodes.append(helper.make_node("Conv", ["input", "W3"], ["m3_f"]))      # f32 [1,1,30,30]

    w4 = np.zeros((1, 10, 1, 1), np.float32); w4[0, 4, 0, 0] = 1.0
    inits.append(cf("W4", w4))
    nodes.append(helper.make_node("Conv", ["input", "W4"], ["m4_f"]))      # f32 [1,1,30,30]

    # ---- row/col one-hot vectors via ReduceMax (f32, tiny). opset13: axes attr ----
    nodes.append(helper.make_node("ReduceMax", ["m3_f"], ["r3"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", ["m3_f"], ["c3"], axes=[2], keepdims=1))  # [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", ["m4_f"], ["r4"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", ["m4_f"], ["c4"], axes=[2], keepdims=1))  # [1,1,1,30]

    # ---- positions: y3=sum(r3*idxH), etc. (scalars) ----
    idxH = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)
    idxW = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
    inits.append(cf("idxH", idxH))
    inits.append(cf("idxW", idxW))
    nodes.append(helper.make_node("Mul", ["r3", "idxH"], ["r3i"]))
    nodes.append(helper.make_node("Mul", ["c3", "idxW"], ["c3i"]))
    nodes.append(helper.make_node("Mul", ["r4", "idxH"], ["r4i"]))
    nodes.append(helper.make_node("Mul", ["c4", "idxW"], ["c4i"]))
    nodes.append(helper.make_node("ReduceSum", ["r3i"], ["y3"], axes=[2, 3], keepdims=1))  # [1,1,1,1]
    nodes.append(helper.make_node("ReduceSum", ["c3i"], ["x3"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["r4i"], ["y4"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["c4i"], ["x4"], axes=[2, 3], keepdims=1))

    nodes.append(helper.make_node("Sub", ["y4", "y3"], ["dy"]))
    nodes.append(helper.make_node("Sub", ["x4", "x3"], ["dx"]))
    nodes.append(helper.make_node("Sign", ["dy"], ["sy"]))  # [1,1,1,1] in {-1,0,1}
    nodes.append(helper.make_node("Sign", ["dx"], ["sx"]))

    # ---- shift the 3 row one-hot r3 by sy (-1/0/+1) ----
    # r3 is [1,1,30,1]. up = new[i]=r3[i+1]; down = new[i]=r3[i-1].
    inits.append(ci("s0", [0, 0, 1, 0]))   # start for "up": axis2 from 1
    inits.append(ci("e_big", [1, 1, 30, 1]))
    inits.append(ci("axall", [0, 1, 2, 3]))
    inits.append(ci("padBot", [0, 0, 0, 0, 0, 0, 1, 0]))  # pad end of axis2 by 1
    inits.append(ci("padTop", [0, 0, 1, 0, 0, 0, 0, 0]))  # pad begin of axis2 by 1
    inits.append(cf("zpad", 0.0))

    # up shift (move toward index 0): drop first row, pad bottom
    inits.append(ci("s1", [1]))
    inits.append(ci("e30", [30]))
    inits.append(ci("ax2only", [2]))
    nodes.append(helper.make_node("Slice", ["r3", "s1", "e30", "ax2only"], ["r3_drop0"]))   # [1,1,29,1]
    nodes.append(helper.make_node("Pad", ["r3_drop0", "padBot", "zpad"], ["r3_up"]))        # [1,1,30,1]
    # down shift (move +1): drop last row, pad top
    inits.append(ci("s0a", [0]))
    inits.append(ci("e29", [29]))
    nodes.append(helper.make_node("Slice", ["r3", "s0a", "e29", "ax2only"], ["r3_drop_last"]))  # [1,1,29,1]
    nodes.append(helper.make_node("Pad", ["r3_drop_last", "padTop", "zpad"], ["r3_down"]))   # [1,1,30,1]

    # selection coefficients from sy
    inits.append(cf("zero", 0.0))
    nodes.append(helper.make_node("Less", ["sy", "zero"], ["sy_neg"]))      # bool
    nodes.append(helper.make_node("Greater", ["sy", "zero"], ["sy_pos"]))   # bool
    nodes.append(helper.make_node("Equal", ["sy", "zero"], ["sy_zero"]))    # bool
    nodes.append(helper.make_node("Cast", ["sy_neg"], ["syn"], to=F32))
    nodes.append(helper.make_node("Cast", ["sy_pos"], ["syp"], to=F32))
    nodes.append(helper.make_node("Cast", ["sy_zero"], ["syz"], to=F32))
    # nr3 = syn*up + syz*r3 + syp*down
    nodes.append(helper.make_node("Mul", ["r3_up", "syn"], ["t_up_r"]))
    nodes.append(helper.make_node("Mul", ["r3", "syz"], ["t_sm_r"]))
    nodes.append(helper.make_node("Mul", ["r3_down", "syp"], ["t_dn_r"]))
    nodes.append(helper.make_node("Add", ["t_up_r", "t_sm_r"], ["t_a_r"]))
    nodes.append(helper.make_node("Add", ["t_a_r", "t_dn_r"], ["nr3"]))     # [1,1,30,1]

    # ---- shift the 3 col one-hot c3 by sx ----
    inits.append(ci("padR", [0, 0, 0, 0, 0, 0, 0, 1]))  # pad end of axis3
    inits.append(ci("padL", [0, 0, 0, 1, 0, 0, 0, 0]))  # pad begin of axis3
    inits.append(ci("ax3only", [3]))
    nodes.append(helper.make_node("Slice", ["c3", "s1", "e30", "ax3only"], ["c3_drop0"]))   # [1,1,1,29]
    nodes.append(helper.make_node("Pad", ["c3_drop0", "padR", "zpad"], ["c3_left"]))        # [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["c3", "s0a", "e29", "ax3only"], ["c3_drop_last"]))
    nodes.append(helper.make_node("Pad", ["c3_drop_last", "padL", "zpad"], ["c3_right"]))   # [1,1,1,30]
    nodes.append(helper.make_node("Less", ["sx", "zero"], ["sx_neg"]))
    nodes.append(helper.make_node("Greater", ["sx", "zero"], ["sx_pos"]))
    nodes.append(helper.make_node("Equal", ["sx", "zero"], ["sx_zero"]))
    nodes.append(helper.make_node("Cast", ["sx_neg"], ["sxn"], to=F32))
    nodes.append(helper.make_node("Cast", ["sx_pos"], ["sxp"], to=F32))
    nodes.append(helper.make_node("Cast", ["sx_zero"], ["sxz"], to=F32))
    nodes.append(helper.make_node("Mul", ["c3_left", "sxn"], ["t_up_c"]))
    nodes.append(helper.make_node("Mul", ["c3", "sxz"], ["t_sm_c"]))
    nodes.append(helper.make_node("Mul", ["c3_right", "sxp"], ["t_dn_c"]))
    nodes.append(helper.make_node("Add", ["t_up_c", "t_sm_c"], ["t_a_c"]))
    nodes.append(helper.make_node("Add", ["t_a_c", "t_dn_c"], ["nc3"]))     # [1,1,1,30]

    # ---- outer product => new3 mask [1,1,30,30] ----
    nodes.append(helper.make_node("Mul", ["nr3", "nc3"], ["new3_f"]))       # f32 [1,1,30,30]

    # ---- build the output color grid in uint8 ----
    # gin (uint8), clear old-3 (m3), set new-3 to color 3.
    nodes.append(helper.make_node("Cast", ["gin_f"], ["gin_u"], to=U8))     # u8 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["m3_f"], ["m3_u"], to=U8))
    nodes.append(helper.make_node("Cast", ["new3_f"], ["new3_u"], to=U8))
    # cleared = gin * (1 - m3)
    inits.append(cu("one_u", 1))
    nodes.append(helper.make_node("Sub", ["one_u", "m3_u"], ["notm3"]))
    nodes.append(helper.make_node("Mul", ["gin_u", "notm3"], ["g_clear"]))  # u8
    # set new3 -> add 3*new3 (new3 cell was background 0 so safe)
    inits.append(cu("three_u", 3))
    nodes.append(helper.make_node("Mul", ["new3_u", "three_u"], ["add3"]))
    nodes.append(helper.make_node("Add", ["g_clear", "add3"], ["gout_u"]))  # u8 [1,1,30,30]

    # ---- sentinel for out-of-grid cells ----
    inits.append(cf("half", 0.5))
    nodes.append(helper.make_node("Greater", ["valsum", "half"], ["valid_b"]))  # bool [1,1,30,30]
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Where", ["valid_b", "gout_u", "sent"], ["gsent"]))  # u8 [1,1,30,30]

    # ---- final one-hot = Equal vs colors -> graph output (excluded) ----
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gsent", "colors"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t353", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1d_task353.onnx")
