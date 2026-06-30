"""task340 v5: drop nz/nzu grids.

colany/rowany come from ReduceMax(g)>0 (max color per col/row). gp = Where(interior, g, 0);
background interior cells are 0 and never equal a wall color>0, so &nz is unnecessary.
All grids uint8/bool; only Conv g32 is f32.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8


def cf(n, v): return numpy_helper.from_array(np.asarray(v, np.float32), name=n)
def ci(n, v): return numpy_helper.from_array(np.asarray(v, np.int64), name=n)
def cb(n, v): return numpy_helper.from_array(np.asarray(v, np.bool_), name=n)
def cu(n, v): return numpy_helper.from_array(np.asarray(v, np.uint8), name=n)


def build(path):
    nodes, inits = [], []

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))

    inits.append(cu("u0", 0))
    # colany/rowany via ReduceMax(g) > 0
    nodes.append(helper.make_node("ReduceMax", ["g"], ["colmaxu"], axes=[2], keepdims=1))  # u8 [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", ["g"], ["rowmaxu"], axes=[3], keepdims=1))  # u8 [1,1,30,1]
    nodes.append(helper.make_node("Greater", ["colmaxu", "u0"], ["colany_b"]))
    nodes.append(helper.make_node("Greater", ["rowmaxu", "u0"], ["rowany_b"]))

    inits += [ci("s0", [0]), ci("s1", [1]), ci("sBig", [30]), ci("ax2", [2]), ci("ax3", [3])]
    inits.append(ci("padC", [0, 0, 0, 0, 0, 0, 0, 1]))
    inits.append(ci("padR", [0, 0, 0, 0, 0, 0, 1, 0]))
    inits.append(cb("falseC", False))

    nodes.append(helper.make_node("Slice", ["colany_b", "s1", "sBig", "ax3"], ["colanyL_t"]))
    nodes.append(helper.make_node("Pad", ["colanyL_t", "padC", "falseC"], ["colanyL_b"]))
    nodes.append(helper.make_node("Slice", ["rowany_b", "s1", "sBig", "ax2"], ["rowanyU_t"]))
    nodes.append(helper.make_node("Pad", ["rowanyU_t", "padR", "falseC"], ["rowanyU_b"]))

    nodes.append(helper.make_node("Not", ["colanyL_b"], ["colanyL_n"]))
    nodes.append(helper.make_node("And", ["colany_b", "colanyL_n"], ["is_rightcol_b"]))
    nodes.append(helper.make_node("Not", ["rowanyU_b"], ["rowanyU_n"]))
    nodes.append(helper.make_node("And", ["rowany_b", "rowanyU_n"], ["is_botrow_b"]))

    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "ax2"], ["topc"]))
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "ax3"], ["leftc"]))
    nodes.append(helper.make_node("Where", ["is_botrow_b", "g", "u0"], ["g_bot"]))
    nodes.append(helper.make_node("ReduceMax", ["g_bot"], ["botc"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Where", ["is_rightcol_b", "g", "u0"], ["g_right"]))
    nodes.append(helper.make_node("ReduceMax", ["g_right"], ["rightc"], axes=[3], keepdims=1))

    notcol0 = np.zeros((1, 1, 1, 30), np.bool_); notcol0[..., 1:] = True
    notrow0 = np.zeros((1, 1, 30, 1), np.bool_); notrow0[:, :, 1:, :] = True
    inits.append(cb("notcol0", notcol0))
    inits.append(cb("notrow0", notrow0))
    nodes.append(helper.make_node("Not", ["is_rightcol_b"], ["n_rightcol"]))
    nodes.append(helper.make_node("And", ["colany_b", "n_rightcol"], ["ic_a"]))
    nodes.append(helper.make_node("And", ["ic_a", "notcol0"], ["interior_c"]))
    nodes.append(helper.make_node("Not", ["is_botrow_b"], ["n_botrow"]))
    nodes.append(helper.make_node("And", ["rowany_b", "n_botrow"], ["ir_a"]))
    nodes.append(helper.make_node("And", ["ir_a", "notrow0"], ["interior_r"]))
    nodes.append(helper.make_node("And", ["interior_r", "interior_c"], ["interior"]))
    nodes.append(helper.make_node("Where", ["interior", "g", "u0"], ["gp"]))  # u8

    def wall(name, wallc, axis):
        nodes.append(helper.make_node("Equal", ["gp", wallc], [name + "_eq"]))
        nodes.append(helper.make_node("Cast", [name + "_eq"], [name + "_equ"], to=U8))
        nodes.append(helper.make_node("ReduceMax", [name + "_equ"], [name + "_has"], axes=[axis], keepdims=1))
        nodes.append(helper.make_node("Cast", [name + "_has"], [name + "_hasb"], to=BOOL))
        nodes.append(helper.make_node("Where", [name + "_hasb", wallc, "u0"], [name + "_v"]))

    wall("top", "topc", 2)
    wall("bot", "botc", 2)
    wall("left", "leftc", 3)
    wall("right", "rightc", 3)

    is_row1 = np.zeros((1, 1, 30, 1), np.bool_); is_row1[:, :, 1, :] = True
    is_col1 = np.zeros((1, 1, 1, 30), np.bool_); is_col1[..., 1] = True
    inits.append(cb("is_row1", is_row1))
    inits.append(cb("is_col1", is_col1))
    nodes.append(helper.make_node("Slice", ["is_botrow_b", "s1", "sBig", "ax2"], ["is_botm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_botm1_t", "padR", "falseC"], ["is_botm1"]))
    nodes.append(helper.make_node("Slice", ["is_rightcol_b", "s1", "sBig", "ax3"], ["is_rightm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_rightm1_t", "padC", "falseC"], ["is_rightm1"]))

    nodes.append(helper.make_node("Where", ["is_row1", "top_v", "u0"], ["top_grid"]))
    nodes.append(helper.make_node("Where", ["is_botm1", "bot_v", "u0"], ["bot_grid"]))
    nodes.append(helper.make_node("Where", ["is_col1", "left_v", "u0"], ["left_grid"]))
    nodes.append(helper.make_node("Where", ["is_rightm1", "right_v", "u0"], ["right_grid"]))
    nodes.append(helper.make_node("Max", ["top_grid", "bot_grid", "left_grid", "right_grid"], ["place"]))

    nodes.append(helper.make_node("Where", ["interior", "place", "g"], ["final0"]))
    nodes.append(helper.make_node("And", ["rowany_b", "colany_b"], ["inside"]))
    inits.append(cu("sent255", np.asarray(255, np.uint8)))
    nodes.append(helper.make_node("Where", ["inside", "final0", "sent255"], ["gm30"]))

    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t340", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b10_task340_v5.onnx")
