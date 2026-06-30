"""task340 v3: all interior [1,1,30,30] grids are uint8/bool (900B), no f16/Mul/Sum.

Uses Where (uint8) instead of Mul, Max (uint8) instead of Sum, ReduceMax (uint8)
instead of ReduceSum. Only the unavoidable Conv output g32 is f32 (input floor).

Algorithm (verified 266/266 in numpy):
  g = color grid (uint8). nz = g>0.
  colany = Max_rows(nz), rowany = Max_cols(nz).
  is_rightcol = colany & ~shiftL(colany); is_botrow = rowany & ~shiftU(rowany).
  topc=g[row0]; leftc=g[col0]; botc=Max_rows(Where(is_botrow,g,0)); rightc=Max_cols(Where(is_rightcol,g,0)).
  interior = (rowany & ~is_botrow & row>=1) x (colany & ~is_rightcol & col>=1).
  particles = nz & interior.
  per wall: has_line = Max(particles & (g==wallc)); v = Where(has_line, wallc, 0).
  placement grids via Where(onehot_line, v_broadcast, 0); combine 4 disjoint via Max.
  final = Where(interior, place, g); then Where(inside, final, 255) sentinel; output = Equal(final, colors).
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

    # color grid g (uint8) from one-hot input (input excluded)
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))  # f32 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))       # u8 [1,1,30,30]

    inits.append(cu("u0", 0))
    nodes.append(helper.make_node("Greater", ["g", "u0"], ["nz"]))      # bool
    nodes.append(helper.make_node("Cast", ["nz"], ["nzu"], to=U8))      # u8 0/1

    nodes.append(helper.make_node("ReduceMax", ["nzu"], ["colanyu"], axes=[2], keepdims=1))  # u8 [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", ["nzu"], ["rowanyu"], axes=[3], keepdims=1))  # u8 [1,1,30,1]
    nodes.append(helper.make_node("Cast", ["colanyu"], ["colany_b"], to=BOOL))
    nodes.append(helper.make_node("Cast", ["rowanyu"], ["rowany_b"], to=BOOL))

    # axes/slice consts
    inits += [ci("s0", [0]), ci("s1", [1]), ci("sBig", [30]), ci("ax2", [2]), ci("ax3", [3])]
    inits.append(ci("padC", [0, 0, 0, 0, 0, 0, 0, 1]))
    inits.append(ci("padR", [0, 0, 0, 0, 0, 0, 1, 0]))
    inits.append(cb("falseC", False))

    # shifted colany/rowany
    nodes.append(helper.make_node("Slice", ["colany_b", "s1", "sBig", "ax3"], ["colanyL_t"]))
    nodes.append(helper.make_node("Pad", ["colanyL_t", "padC", "falseC"], ["colanyL_b"]))
    nodes.append(helper.make_node("Slice", ["rowany_b", "s1", "sBig", "ax2"], ["rowanyU_t"]))
    nodes.append(helper.make_node("Pad", ["rowanyU_t", "padR", "falseC"], ["rowanyU_b"]))

    # is_rightcol / is_botrow (bool)
    nodes.append(helper.make_node("Not", ["colanyL_b"], ["colanyL_n"]))
    nodes.append(helper.make_node("And", ["colany_b", "colanyL_n"], ["is_rightcol_b"]))  # [1,1,1,30]
    nodes.append(helper.make_node("Not", ["rowanyU_b"], ["rowanyU_n"]))
    nodes.append(helper.make_node("And", ["rowany_b", "rowanyU_n"], ["is_botrow_b"]))    # [1,1,30,1]

    # wall colors
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "ax2"], ["topc"]))   # u8 [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "ax3"], ["leftc"]))  # u8 [1,1,30,1]
    # botc = Max_rows(Where(is_botrow, g, 0)); rightc = Max_cols(Where(is_rightcol, g, 0))
    nodes.append(helper.make_node("Where", ["is_botrow_b", "g", "u0"], ["g_bot"]))      # u8 [1,1,30,30]
    nodes.append(helper.make_node("ReduceMax", ["g_bot"], ["botc"], axes=[2], keepdims=1))  # u8 [1,1,1,30]
    nodes.append(helper.make_node("Where", ["is_rightcol_b", "g", "u0"], ["g_right"]))  # u8 [1,1,30,30]
    nodes.append(helper.make_node("ReduceMax", ["g_right"], ["rightc"], axes=[3], keepdims=1))  # u8 [1,1,30,1]

    # interior masks
    notcol0 = np.zeros((1, 1, 1, 30), np.bool_); notcol0[..., 1:] = True
    notrow0 = np.zeros((1, 1, 30, 1), np.bool_); notrow0[:, :, 1:, :] = True
    inits.append(cb("notcol0", notcol0))
    inits.append(cb("notrow0", notrow0))
    nodes.append(helper.make_node("Not", ["is_rightcol_b"], ["n_rightcol"]))
    nodes.append(helper.make_node("And", ["colany_b", "n_rightcol"], ["ic_a"]))
    nodes.append(helper.make_node("And", ["ic_a", "notcol0"], ["interior_c"]))  # [1,1,1,30]
    nodes.append(helper.make_node("Not", ["is_botrow_b"], ["n_botrow"]))
    nodes.append(helper.make_node("And", ["rowany_b", "n_botrow"], ["ir_a"]))
    nodes.append(helper.make_node("And", ["ir_a", "notrow0"], ["interior_r"]))  # [1,1,30,1]
    nodes.append(helper.make_node("And", ["interior_r", "interior_c"], ["interior"]))  # [1,1,30,30]
    nodes.append(helper.make_node("And", ["nz", "interior"], ["particles"]))           # [1,1,30,30]

    # matches per wall -> has_line (u8 [1,1,1,30] or [1,1,30,1])
    def wall(name, wallc, axis):
        nodes.append(helper.make_node("Equal", ["g", wallc], [name + "_eq"]))
        nodes.append(helper.make_node("And", [name + "_eq", "particles"], [name + "_m"]))
        nodes.append(helper.make_node("Cast", [name + "_m"], [name + "_mu"], to=U8))
        nodes.append(helper.make_node("ReduceMax", [name + "_mu"], [name + "_has"], axes=[axis], keepdims=1))
        # value vector = Where(has>0, wallc, 0)
        nodes.append(helper.make_node("Cast", [name + "_has"], [name + "_hasb"], to=BOOL))
        nodes.append(helper.make_node("Where", [name + "_hasb", wallc, "u0"], [name + "_v"]))
        return name + "_v"

    topv = wall("top", "topc", 2)      # [1,1,1,30]
    botv = wall("bot", "botc", 2)      # [1,1,1,30]
    leftv = wall("left", "leftc", 3)   # [1,1,30,1]
    rightv = wall("right", "rightc", 3) # [1,1,30,1]

    # one-hot line selectors (bool)
    is_row1 = np.zeros((1, 1, 30, 1), np.bool_); is_row1[:, :, 1, :] = True
    is_col1 = np.zeros((1, 1, 1, 30), np.bool_); is_col1[..., 1] = True
    inits.append(cb("is_row1", is_row1))
    inits.append(cb("is_col1", is_col1))
    # is_botm1 = shiftU(is_botrow); is_rightm1 = shiftL(is_rightcol)
    nodes.append(helper.make_node("Slice", ["is_botrow_b", "s1", "sBig", "ax2"], ["is_botm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_botm1_t", "padR", "falseC"], ["is_botm1"]))   # bool [1,1,30,1]
    nodes.append(helper.make_node("Slice", ["is_rightcol_b", "s1", "sBig", "ax3"], ["is_rightm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_rightm1_t", "padC", "falseC"], ["is_rightm1"]))  # bool [1,1,1,30]

    # placement grids via Where(onehot_line, v_broadcast, 0)  -> u8 [1,1,30,30]
    nodes.append(helper.make_node("Where", ["is_row1", "top_v", "u0"], ["top_grid"]))
    nodes.append(helper.make_node("Where", ["is_botm1", "bot_v", "u0"], ["bot_grid"]))
    nodes.append(helper.make_node("Where", ["is_col1", "left_v", "u0"], ["left_grid"]))
    nodes.append(helper.make_node("Where", ["is_rightm1", "right_v", "u0"], ["right_grid"]))
    # combine 4 disjoint via Max
    nodes.append(helper.make_node("Max", ["top_grid", "bot_grid", "left_grid", "right_grid"], ["place"]))

    # final = Where(interior, place, g); then sentinel outside grid
    nodes.append(helper.make_node("Where", ["interior", "place", "g"], ["final0"]))  # u8 [1,1,30,30]
    nodes.append(helper.make_node("And", ["rowany_b", "colany_b"], ["inside"]))      # bool [1,1,30,30]
    inits.append(cu("sent255", np.asarray(255, np.uint8)))
    nodes.append(helper.make_node("Where", ["inside", "final0", "sent255"], ["gm30"]))  # u8

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
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b10_task340_v3.onnx")
