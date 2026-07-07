"""task340 v2: same algorithm, uint8/bool everywhere to shrink the [1,1,30,30]
intermediates from f16 (1800B) to uint8/bool (900B).

g is built as uint8 directly. Walls, matches, placements, frame, total all uint8.
Masks are bool. Per-line vectors are tiny [1,1,1,30]/[1,1,30,1]. Output = Equal(...).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cb(name, v):
    return numpy_helper.from_array(np.asarray(v, np.bool_), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    # color grid g[1,1,30,30] uint8 from one-hot input (input excluded)
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))  # f32 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))       # u8 [1,1,30,30]

    inits.append(cu("u0", 0))
    # nz = g>0  bool
    nodes.append(helper.make_node("Greater", ["g", "u0"], ["nz"]))      # bool [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["nz"], ["nzu"], to=U8))      # u8 0/1 [1,1,30,30]

    # colany [1,1,1,30], rowany [1,1,30,1]  (max over the other axis)
    nodes.append(helper.make_node("ReduceMax", ["nzu"], ["colany"], axes=[2], keepdims=1))  # u8 [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", ["nzu"], ["rowany"], axes=[3], keepdims=1))  # u8 [1,1,30,1]
    nodes.append(helper.make_node("Cast", ["colany"], ["colany_b"], to=BOOL))
    nodes.append(helper.make_node("Cast", ["rowany"], ["rowany_b"], to=BOOL))

    # shifted: colany[c+1], rowany[r+1]
    inits += [ci("s1", [1]), ci("sBig", [30])]
    nodes.append(helper.make_node("Slice", ["colany_b", "s1", "sBig", "ax3"], ["colanyL_t"]))
    inits.append(ci("padC", [0, 0, 0, 0, 0, 0, 0, 1]))
    inits.append(cb("falseC", False))
    nodes.append(helper.make_node("Pad", ["colanyL_t", "padC", "falseC"], ["colanyL_b"]))  # bool [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["rowany_b", "s1", "sBig", "ax2"], ["rowanyU_t"]))
    inits.append(ci("padR", [0, 0, 0, 0, 0, 0, 1, 0]))
    nodes.append(helper.make_node("Pad", ["rowanyU_t", "padR", "falseC"], ["rowanyU_b"]))  # bool [1,1,30,1]

    # is_rightcol = colany_b & ~colanyL_b ; is_botrow = rowany_b & ~rowanyU_b
    nodes.append(helper.make_node("Not", ["colanyL_b"], ["colanyL_n"]))
    nodes.append(helper.make_node("And", ["colany_b", "colanyL_n"], ["is_rightcol_b"]))  # bool [1,1,1,30]
    nodes.append(helper.make_node("Not", ["rowanyU_b"], ["rowanyU_n"]))
    nodes.append(helper.make_node("And", ["rowany_b", "rowanyU_n"], ["is_botrow_b"]))    # bool [1,1,30,1]

    # axes consts
    inits += [ci("ax2", [2]), ci("ax3", [3]), ci("s0", [0])]

    # wall colors: topc=g row0 (u8 [1,1,1,30]); leftc=g col0 (u8 [1,1,30,1])
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "ax2"], ["topc"]))   # u8 [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "ax3"], ["leftc"]))  # u8 [1,1,30,1]
    # botc = sum over rows of g*is_botrow ; rightc = sum over cols of g*is_rightcol
    nodes.append(helper.make_node("Cast", ["is_botrow_b"], ["is_botrow_u"], to=U8))
    nodes.append(helper.make_node("Cast", ["is_rightcol_b"], ["is_rightcol_u"], to=U8))
    nodes.append(helper.make_node("Mul", ["g", "is_botrow_u"], ["g_bot"]))           # u8 [1,1,30,30]
    nodes.append(helper.make_node("ReduceSum", ["g_bot", "ax2"], ["botc"], keepdims=1))   # u8 [1,1,1,30]
    nodes.append(helper.make_node("Mul", ["g", "is_rightcol_u"], ["g_right"]))       # u8 [1,1,30,30]
    nodes.append(helper.make_node("ReduceSum", ["g_right", "ax3"], ["rightc"], keepdims=1))  # u8 [1,1,30,1]

    # interior masks (bool)
    notcol0 = np.zeros((1, 1, 1, 30), np.bool_); notcol0[..., 1:] = True
    notrow0 = np.zeros((1, 1, 30, 1), np.bool_); notrow0[:, :, 1:, :] = True
    inits.append(cb("notcol0", notcol0))
    inits.append(cb("notrow0", notrow0))
    nodes.append(helper.make_node("Not", ["is_rightcol_b"], ["n_rightcol"]))
    nodes.append(helper.make_node("And", ["colany_b", "n_rightcol"], ["ic_a"]))
    nodes.append(helper.make_node("And", ["ic_a", "notcol0"], ["interior_c"]))  # bool [1,1,1,30]
    nodes.append(helper.make_node("Not", ["is_botrow_b"], ["n_botrow"]))
    nodes.append(helper.make_node("And", ["rowany_b", "n_botrow"], ["ir_a"]))
    nodes.append(helper.make_node("And", ["ir_a", "notrow0"], ["interior_r"]))  # bool [1,1,30,1]
    nodes.append(helper.make_node("And", ["interior_r", "interior_c"], ["interior"]))  # bool [1,1,30,30]
    nodes.append(helper.make_node("And", ["nz", "interior"], ["particles"]))           # bool [1,1,30,30]

    # matches per wall
    def wall_match(name, wallc_t):
        nodes.append(helper.make_node("Equal", ["g", wallc_t], [name + "_eq"]))
        nodes.append(helper.make_node("And", [name + "_eq", "particles"], [name + "_m"]))  # bool
        nodes.append(helper.make_node("Cast", [name + "_m"], [name + "_mu"], to=U8))        # u8
        return name + "_mu"

    mt = wall_match("top", "topc")
    mb = wall_match("bot", "botc")
    ml = wall_match("left", "leftc")
    mr = wall_match("right", "rightc")
    # per-line OR via ReduceMax (u8)
    nodes.append(helper.make_node("ReduceMax", [mt], ["col_has_top"], axes=[2], keepdims=1))   # u8 [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", [mb], ["col_has_bot"], axes=[2], keepdims=1))   # u8 [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", [ml], ["row_has_left"], axes=[3], keepdims=1))  # u8 [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", [mr], ["row_has_right"], axes=[3], keepdims=1)) # u8 [1,1,30,1]

    # one-hot row/col selectors (u8)
    is_row1 = np.zeros((1, 1, 30, 1), np.uint8); is_row1[:, :, 1, :] = 1
    is_col1 = np.zeros((1, 1, 1, 30), np.uint8); is_col1[..., 1] = 1
    inits.append(cu("is_row1", is_row1))
    inits.append(cu("is_col1", is_col1))
    # is_botm1 = shiftU(is_botrow_u); is_rightm1 = shiftL(is_rightcol_u)
    nodes.append(helper.make_node("Slice", ["is_botrow_u", "s1", "sBig", "ax2"], ["is_botm1_t"]))
    inits.append(cu("u0pad", 0))
    nodes.append(helper.make_node("Pad", ["is_botm1_t", "padR", "u0pad"], ["is_botm1"]))   # u8 [1,1,30,1]
    nodes.append(helper.make_node("Slice", ["is_rightcol_u", "s1", "sBig", "ax3"], ["is_rightm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_rightm1_t", "padC", "u0pad"], ["is_rightm1"]))  # u8 [1,1,1,30]

    # placement color vectors: wallcolor * has_line  (u8, tiny)
    nodes.append(helper.make_node("Mul", ["topc", "col_has_top"], ["topv"]))
    nodes.append(helper.make_node("Mul", ["botc", "col_has_bot"], ["botv"]))
    nodes.append(helper.make_node("Mul", ["leftc", "row_has_left"], ["leftv"]))
    nodes.append(helper.make_node("Mul", ["rightc", "row_has_right"], ["rightv"]))
    # placement grids via outer product (broadcast u8)
    nodes.append(helper.make_node("Mul", ["is_row1", "topv"], ["top_grid"]))
    nodes.append(helper.make_node("Mul", ["is_botm1", "botv"], ["bot_grid"]))
    nodes.append(helper.make_node("Mul", ["leftv", "is_col1"], ["left_grid"]))
    nodes.append(helper.make_node("Mul", ["rightv", "is_rightm1"], ["right_grid"]))

    # frame grid = g * (~interior)
    nodes.append(helper.make_node("Not", ["interior"], ["frame_b"]))
    nodes.append(helper.make_node("Cast", ["frame_b"], ["frame_u"], to=U8))
    nodes.append(helper.make_node("Mul", ["g", "frame_u"], ["frame_grid"]))  # u8

    # total = sum of five disjoint u8 grids
    nodes.append(helper.make_node("Sum", ["top_grid", "bot_grid", "left_grid", "right_grid", "frame_grid"], ["total"]))

    # mask outside grid -> sentinel 255
    nodes.append(helper.make_node("And", ["rowany_b", "colany_b"], ["inside"]))  # bool [1,1,30,30]
    inits.append(cu("sent255", np.asarray(255, np.uint8)))
    nodes.append(helper.make_node("Where", ["inside", "total", "sent255"], ["gm30"]))  # u8

    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))  # bool [1,10,30,30] = output

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t340", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b10_task340_v2.onnx")
