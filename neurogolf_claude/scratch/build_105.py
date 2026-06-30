"""Build compact ONNX for task105.

Rule (verified 266/266): the foreground (color 1) draws dashed rectangular
lattice lines inside a bounding box. Complete every lattice line by filling its
in-box background (color 0) cells with color 2.

A row r (within bbox [r0,r1]x[c0,c1]) is a *line* iff it is the top/bottom edge,
OR rowcount>=4, OR (rowcount==3 AND it touches at most one of the two side
walls c0,c1). Columns are symmetric (touching top/bottom edges r0,r1).
A background cell on a line-row or line-col (and inside the bbox) becomes 2.

All grids are <=14 rows x 13 cols, so we crop the color grid to 14x13 and run
the whole pipeline on tiny [1,1,14,13] tensors (uint8/int32/bool). The graph
output is Equal(color, colors0..9) one-hot bool [1,10,30,30] (excluded from
memory); off-grid cells carry sentinel 255 so they match nothing -> all-zero.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I32 = TensorProto.INT32
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
H, W = 14, 13


def ci64(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def ci32(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int32), name=name)


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cf16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def cu8(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path):
    nodes, inits = [], []

    def add(op, i, o, **kw):
        nodes.append(helper.make_node(op, i, o, **kw))

    # --- Slice the input one-hot channels directly at crop size 14x13.
    #     Inputs use only colors 0,1: channel 1 = foreground, channel 0 = background.
    #     Each single-channel crop is f32 [1,1,14,13]=728 B (far cheaper than a
    #     30x30 color conv). on-grid = ch0|ch1 ; foreground f = ch1.
    inits += [ci64("s1", [1, 0, 0]), ci64("e1", [2, H, W]), ci64("a1", [1, 2, 3])]
    inits += [ci64("s0", [0, 0, 0]), ci64("e0", [1, H, W])]
    add("Slice", ["input", "s1", "e1", "a1"], ["ch1"])   # f32 [1,1,14,13] foreground
    add("Slice", ["input", "s0", "e0", "a1"], ["ch0"])   # f32 [1,1,14,13] background

    inits.append(cf("zf", [0.0]))
    add("Greater", ["ch1", "zf"], ["f_b"])               # bool foreground
    add("Greater", ["ch0", "zf"], ["bg_b"])              # bool on-grid background

    # --- rowcount [1,1,14,1], colcount [1,1,1,13]  directly from ch1 (f32 0/1).
    inits.append(ci64("ax3", [3]))
    inits.append(ci64("ax2", [2]))
    add("ReduceSum", ["ch1", "ax3"], ["rowcount"], keepdims=1)   # f32 [1,1,14,1]
    add("ReduceSum", ["ch1", "ax2"], ["colcount"], keepdims=1)   # f32 [1,1,1,13]

    # rows_any / cols_any (int32 0/1 for CumSum)
    inits.append(cf("zf16", [0.0]))
    add("Greater", ["rowcount", "zf16"], ["rows_any_b"])
    add("Greater", ["colcount", "zf16"], ["cols_any_b"])
    add("Cast", ["rows_any_b"], ["rows_any"], to=I32)
    add("Cast", ["cols_any_b"], ["cols_any"], to=I32)
    inits.append(ci32("z32", 0))

    # cumulative sums along the spatial axis (CumSum needs an axis scalar input)
    inits.append(ci64("c2", 2))
    inits.append(ci64("c3", 3))
    add("CumSum", ["rows_any", "c2"], ["cs_top"])                 # along rows fwd
    add("CumSum", ["rows_any", "c2"], ["cs_bot"], reverse=1)      # along rows rev
    add("CumSum", ["cols_any", "c3"], ["cs_l"])                   # along cols fwd
    add("CumSum", ["cols_any", "c3"], ["cs_r"], reverse=1)        # along cols rev

    # inbox_row = cs_top>0 & cs_bot>0 ; inbox_col similar
    add("Greater", ["cs_top", "z32"], ["ibr_t"])
    add("Greater", ["cs_bot", "z32"], ["ibr_b"])
    add("And", ["ibr_t", "ibr_b"], ["inbox_row_b"])              # bool [1,1,14,1]
    add("Greater", ["cs_l", "z32"], ["ibc_l"])
    add("Greater", ["cs_r", "z32"], ["ibc_r"])
    add("And", ["ibc_l", "ibc_r"], ["inbox_col_b"])              # bool [1,1,1,13]
    add("And", ["inbox_row_b", "inbox_col_b"], ["inbox_b"])      # bool [1,1,14,13]

    # is_r0/r1/c0/c1 : (any) & (cs==1).  rows_any_b/cols_any_b reused for (any==1).
    inits.append(ci32("one32", 1))
    add("Equal", ["cs_top", "one32"], ["cst1"])
    add("And", ["rows_any_b", "cst1"], ["is_r0_b"])
    add("Equal", ["cs_bot", "one32"], ["csb1"])
    add("And", ["rows_any_b", "csb1"], ["is_r1_b"])
    add("Equal", ["cs_l", "one32"], ["csl1"])
    add("And", ["cols_any_b", "csl1"], ["is_c0_b"])
    add("Equal", ["cs_r", "one32"], ["csr1"])
    add("And", ["cols_any_b", "csr1"], ["is_c1_b"])

    # te via MatMul to avoid materializing big [14,13] products.
    # isc = (is_c0|is_c1) as int32 column [1,1,13,1]; te_row = f @ isc -> [1,1,14,1]
    add("Or", ["is_c0_b", "is_c1_b"], ["isc_b"])                # bool [1,1,1,13]
    add("Cast", ["isc_b"], ["isc_r"], to=F32)                   # f32 [1,1,1,13]
    inits.append(ci64("shp_c", [1, 1, 13, 1]))
    add("Reshape", ["isc_r", "shp_c"], ["isc"])                 # f32 [1,1,13,1]
    add("MatMul", ["ch1", "isc"], ["te_row"])                   # f32 [1,1,14,1]
    # isr = (is_r0|is_r1) as f32 row [1,1,1,14]; te_col = isr @ ch1 -> [1,1,1,13]
    add("Or", ["is_r0_b", "is_r1_b"], ["isr_b"])                # bool [1,1,14,1]
    add("Cast", ["isr_b"], ["isr_c"], to=F32)                   # f32 [1,1,14,1]
    inits.append(ci64("shp_r", [1, 1, 1, 14]))
    add("Reshape", ["isr_c", "shp_r"], ["isr"])                 # f32 [1,1,1,14]
    add("MatMul", ["isr", "ch1"], ["te_col"])                   # f32 [1,1,1,13]

    # line_row = is_r0|is_r1 | rowcount>=4 | (rowcount==3 & te_row<=1)
    inits.append(cf("four", [4.0]))
    inits.append(cf("three", [3.0]))
    inits.append(cf("one16", [1.0]))
    add("GreaterOrEqual", ["rowcount", "four"], ["lr_c4"])
    add("Equal", ["rowcount", "three"], ["lr_c3"])
    add("LessOrEqual", ["te_row", "one16"], ["lr_te"])
    add("And", ["lr_c3", "lr_te"], ["lr_c3te"])
    add("Or", ["isr_b", "lr_c4"], ["lr_a"])
    add("Or", ["lr_a", "lr_c3te"], ["line_row_b"])              # bool [1,1,14,1]

    add("GreaterOrEqual", ["colcount", "four"], ["lc_c4"])
    add("Equal", ["colcount", "three"], ["lc_c3"])
    add("LessOrEqual", ["te_col", "one16"], ["lc_te"])
    add("And", ["lc_c3", "lc_te"], ["lc_c3te"])
    add("Or", ["isc_b", "lc_c4"], ["lc_a"])
    add("Or", ["lc_a", "lc_c3te"], ["line_col_b"])              # bool [1,1,1,13]

    # online = line_row | line_col (broadcast); lc = inbox & online (line cell)
    add("Or", ["line_row_b", "line_col_b"], ["online_b"])      # bool [1,1,14,13]
    add("And", ["inbox_b", "online_b"], ["lc_b"])             # bool [1,1,14,13]
    # color (priority): fg->1 ; else line cell->2 ; else on-grid bg->0 ; else off-grid->255.
    inits.append(cu8("two_u8", 2))
    inits.append(cu8("zero_u8", 0))
    inits.append(cu8("one_u8", 1))
    inits.append(cu8("sent", 255))
    add("Where", ["bg_b", "zero_u8", "sent"], ["lvl0"])       # uint8 bg->0 else 255
    add("Where", ["lc_b", "two_u8", "lvl0"], ["lvl1"])        # uint8 line->2 else lvl0
    add("Where", ["f_b", "one_u8", "lvl1"], ["color_og"])     # uint8 fg->1 else lvl1

    # pad back to 30x30 with sentinel 255
    inits.append(ci64("pads", [0, 0, 0, 0, 0, 0, 30 - H, 30 - W]))
    add("Pad", ["color_og", "pads", "sent"], ["color30"])     # uint8 [1,1,30,30]

    # one-hot output = (color30 == c) for c in 0..9  -> graph output (excluded)
    inits.append(cu8("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    add("Equal", ["color30", "colors"], ["output"])           # bool [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t105", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1d_task105.onnx")
