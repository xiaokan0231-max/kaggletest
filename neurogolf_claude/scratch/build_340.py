"""Build compact ONNX for task340.

Rule (verified 262/262): the grid is a rectangle with a 1-cell colored frame
(four distinct border colors: top/bottom/left/right). Each interior particle
slides along its row or column to the frame whose color it matches, landing in
the cell adjacent to that wall (row1 for top, H-2 for bottom, col1 for left,
W-2 for right). Particles matching no border vanish. No collisions occur.

Tensor algorithm (all on a single [1,1,30,30] color grid g):
  nz=g>0 ; colany=Max over rows ; rowany=Max over cols
  is_rightcol = colany & ~shiftL(colany) ; is_botrow = rowany & ~shiftU(rowany)
  topc=row0 of g ; leftc=col0 of g ; botc=Sum(g*is_botrow,rows) ; rightc=Sum(g*is_rightcol,cols)
  interior = (rowany&~is_botrow, drop row0) x (colany&~is_rightcol, drop col0)
  particles = nz & interior
  For each wall: match where particle color == wall color; per-line OR -> has_line;
  build placement grid = onehot(target row/col) (x) (wallcolor*has_line); add the four
  placement grids + frame (g where ~interior); positions are disjoint so sum is exact.
  Mask outside-grid (rowany&colany) to sentinel 255, then Equal vs colors[0..9] = output.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


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

    # --- color grid g[1,1,30,30] f16 from one-hot input (input excluded from memory)
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))  # f32 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=F16))      # f16 [1,1,30,30]

    inits.append(c16("half", 0.5))
    # nz = g>0.5  bool [1,1,30,30]
    nodes.append(helper.make_node("Greater", ["g", "half"], ["nz"]))
    nodes.append(helper.make_node("Cast", ["nz"], ["nzf"], to=F16))  # f16 for reductions

    # colany [1,1,1,30], rowany [1,1,30,1]  (max over the other axis)
    nodes.append(helper.make_node("ReduceMax", ["nzf"], ["colany"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["nzf"], ["rowany"], axes=[3], keepdims=1))

    # shifted versions: colany shifted LEFT by 1 (colany[c+1]); rowany shifted UP by 1
    # Slice off first element then pad a 0 at the end.
    inits += [ci("s1", [1]), ci("sBig", [30]), ci("axCol", [3]), ci("axRow", [2])]
    nodes.append(helper.make_node("Slice", ["colany", "s1", "sBig", "axCol"], ["colany_t"]))  # [1,1,1,29]
    inits.append(ci("padC", [0, 0, 0, 0, 0, 0, 0, 1]))
    inits.append(c16("zero", 0.0))
    nodes.append(helper.make_node("Pad", ["colany_t", "padC", "zero"], ["colanyL"]))  # [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["rowany", "s1", "sBig", "axRow"], ["rowany_t"]))  # [1,1,29,1]
    inits.append(ci("padR", [0, 0, 0, 0, 0, 0, 1, 0]))
    nodes.append(helper.make_node("Pad", ["rowany_t", "padR", "zero"], ["rowanyU"]))  # [1,1,30,1]

    # is_rightcol = colany>0.5 AND colanyL<0.5  ; is_botrow = rowany>0.5 AND rowanyU<0.5
    nodes.append(helper.make_node("Greater", ["colany", "half"], ["colany_b"]))
    nodes.append(helper.make_node("Greater", ["colanyL", "half"], ["colanyL_b"]))
    nodes.append(helper.make_node("Not", ["colanyL_b"], ["colanyL_n"]))
    nodes.append(helper.make_node("And", ["colany_b", "colanyL_n"], ["is_rightcol_b"]))  # bool [1,1,1,30]
    nodes.append(helper.make_node("Cast", ["is_rightcol_b"], ["is_rightcol"], to=F16))

    nodes.append(helper.make_node("Greater", ["rowany", "half"], ["rowany_b"]))
    nodes.append(helper.make_node("Greater", ["rowanyU", "half"], ["rowanyU_b"]))
    nodes.append(helper.make_node("Not", ["rowanyU_b"], ["rowanyU_n"]))
    nodes.append(helper.make_node("And", ["rowany_b", "rowanyU_n"], ["is_botrow_b"]))   # bool [1,1,30,1]
    nodes.append(helper.make_node("Cast", ["is_botrow_b"], ["is_botrow"], to=F16))

    # wall colors:
    # topc [1,1,1,30] = g row0 ; leftc [1,1,30,1] = g col0
    inits += [ci("s0", [0]), ci("axR2", [2]), ci("axC2", [3])]
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "axR2"], ["topc"]))   # [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["g", "s0", "s1", "axC2"], ["leftc"]))  # [1,1,30,1]
    # botc [1,1,1,30] = sum over rows of g*is_botrow ; rightc [1,1,30,1] = sum over cols of g*is_rightcol
    inits += [ci("ax2", [2]), ci("ax3", [3])]
    nodes.append(helper.make_node("Mul", ["g", "is_botrow"], ["g_bot"]))
    nodes.append(helper.make_node("ReduceSum", ["g_bot", "ax2"], ["botc"], keepdims=1))
    nodes.append(helper.make_node("Mul", ["g", "is_rightcol"], ["g_right"]))
    nodes.append(helper.make_node("ReduceSum", ["g_right", "ax3"], ["rightc"], keepdims=1))

    # interior masks (bool). col index>=1 mask and row index>=1 mask as constants.
    notcol0 = np.zeros((1, 1, 1, 30), np.bool_); notcol0[..., 1:] = True
    notrow0 = np.zeros((1, 1, 30, 1), np.bool_); notrow0[:, :, 1:, :] = True
    inits.append(cb("notcol0", notcol0))
    inits.append(cb("notrow0", notrow0))
    # interior_c = colany_b & ~is_rightcol_b & notcol0  ; interior_r = rowany_b & ~is_botrow_b & notrow0
    nodes.append(helper.make_node("Not", ["is_rightcol_b"], ["n_rightcol"]))
    nodes.append(helper.make_node("And", ["colany_b", "n_rightcol"], ["ic_a"]))
    nodes.append(helper.make_node("And", ["ic_a", "notcol0"], ["interior_c"]))  # bool [1,1,1,30]
    nodes.append(helper.make_node("Not", ["is_botrow_b"], ["n_botrow"]))
    nodes.append(helper.make_node("And", ["rowany_b", "n_botrow"], ["ir_a"]))
    nodes.append(helper.make_node("And", ["ir_a", "notrow0"], ["interior_r"]))  # bool [1,1,30,1]
    # interior = interior_r & interior_c (broadcast) -> [1,1,30,30]
    nodes.append(helper.make_node("And", ["interior_r", "interior_c"], ["interior"]))
    # particles = nz & interior
    nodes.append(helper.make_node("And", ["nz", "interior"], ["particles"]))    # bool [1,1,30,30]

    # matches per wall: (g==wallc) & particles & (wallc>0)
    def wall_match(name, wallc_t, wallc_gt0_needed=True):
        # eqb = Equal(g, wallc) ; m = eqb & particles  (g>0 already ensured by particles ⊂ nz; wall>0 ensured because g[match]>0)
        nodes.append(helper.make_node("Equal", ["g", wallc_t], [name + "_eq"]))
        nodes.append(helper.make_node("And", [name + "_eq", "particles"], [name + "_m"]))
        return name + "_m"

    mt = wall_match("top", "topc")
    mb = wall_match("bot", "botc")
    ml = wall_match("left", "leftc")
    mr = wall_match("right", "rightc")
    # cast matches to f16 for per-line reduction
    nodes.append(helper.make_node("Cast", [mt], ["mt_f"], to=F16))
    nodes.append(helper.make_node("Cast", [mb], ["mb_f"], to=F16))
    nodes.append(helper.make_node("Cast", [ml], ["ml_f"], to=F16))
    nodes.append(helper.make_node("Cast", [mr], ["mr_f"], to=F16))
    # per-line OR via ReduceMax
    nodes.append(helper.make_node("ReduceMax", ["mt_f"], ["col_has_top"], axes=[2], keepdims=1))   # [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", ["mb_f"], ["col_has_bot"], axes=[2], keepdims=1))   # [1,1,1,30]
    nodes.append(helper.make_node("ReduceMax", ["ml_f"], ["row_has_left"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(helper.make_node("ReduceMax", ["mr_f"], ["row_has_right"], axes=[3], keepdims=1)) # [1,1,30,1]

    # one-hot row/col selectors:
    is_row1 = np.zeros((1, 1, 30, 1), np.float16); is_row1[:, :, 1, :] = 1
    is_col1 = np.zeros((1, 1, 1, 30), np.float16); is_col1[..., 1] = 1
    inits.append(c16("is_row1", is_row1))
    inits.append(c16("is_col1", is_col1))
    # is_botm1 = shiftU(is_botrow) one more (row H-2); is_rightm1 = shiftL(is_rightcol)
    nodes.append(helper.make_node("Slice", ["is_botrow", "s1", "sBig", "axRow"], ["is_botm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_botm1_t", "padR", "zero"], ["is_botm1"]))   # [1,1,30,1]
    nodes.append(helper.make_node("Slice", ["is_rightcol", "s1", "sBig", "axCol"], ["is_rightm1_t"]))
    nodes.append(helper.make_node("Pad", ["is_rightm1_t", "padC", "zero"], ["is_rightm1"]))  # [1,1,1,30]

    # placement color vectors (per line): wallcolor * has_line
    nodes.append(helper.make_node("Mul", ["topc", "col_has_top"], ["topv"]))      # [1,1,1,30]
    nodes.append(helper.make_node("Mul", ["botc", "col_has_bot"], ["botv"]))      # [1,1,1,30]
    nodes.append(helper.make_node("Mul", ["leftc", "row_has_left"], ["leftv"]))   # [1,1,30,1]
    nodes.append(helper.make_node("Mul", ["rightc", "row_has_right"], ["rightv"]))# [1,1,30,1]
    # placement grids via outer product (broadcast)
    nodes.append(helper.make_node("Mul", ["is_row1", "topv"], ["top_grid"]))       # [1,1,30,30]
    nodes.append(helper.make_node("Mul", ["is_botm1", "botv"], ["bot_grid"]))      # [1,1,30,30]
    nodes.append(helper.make_node("Mul", ["leftv", "is_col1"], ["left_grid"]))     # [1,1,30,30]
    nodes.append(helper.make_node("Mul", ["rightv", "is_rightm1"], ["right_grid"]))# [1,1,30,30]

    # frame grid = g where ~interior else 0
    nodes.append(helper.make_node("Not", ["interior"], ["frame_b"]))
    nodes.append(helper.make_node("Cast", ["frame_b"], ["frame_f"], to=F16))
    nodes.append(helper.make_node("Mul", ["g", "frame_f"], ["frame_grid"]))         # [1,1,30,30]

    # total = sum of the five disjoint grids
    nodes.append(helper.make_node("Sum", ["top_grid", "bot_grid", "left_grid", "right_grid", "frame_grid"], ["total"]))

    # mask outside grid (rowany_b & colany_b) -> inside; else sentinel 255
    nodes.append(helper.make_node("And", ["rowany_b", "colany_b"], ["inside"]))   # bool [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["total"], ["total_u"], to=U8))         # u8 0..9
    inits.append(cu("sent255", np.asarray(255, np.uint8)))
    nodes.append(helper.make_node("Where", ["inside", "total_u", "sent255"], ["gm30"]))  # u8 [1,1,30,30]

    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))  # bool [1,10,30,30] = graph output

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t340", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b10_task340.onnx")
