"""Build compact ONNX for task301: staircase sort of horizontal bars.

Rule (verified 266/266): the input has W bars (horizontal runs), one per nonzero
row, of lengths exactly 1..W and distinct colors; the bottom-most bar is full
width W. The output sorts them by length and stacks them right-aligned in the
bottom W rows of the grid (shortest on top, longest = base at the bottom row),
forming a bottom-right staircase. Outside the bars everything is background 0;
cells outside the actual HxW grid are all-zero in every channel.

Canvas formulation (crop to HHxWW which covers every example):
  g    = color grid [1,1,HH,WW]  (Conv collapse of one-hot input, cropped, uint8)
  nz   = g>0
  Lr   = sum_c nz   (bar length per row; 0 for empty rows)
  Cr   = max_c g    (bar color per row)
  H1   = max row index with Lr>0   (bottom grid row)
  Wd   = max(Lr)                   (grid width = base bar length)
  T    = H1 - Wd + Lr              (target out-row per input row)
  OCr  = Cr where T==r             (out-row color, via Equal-match + ReduceSum)
  cell (r,c) filled iff OCr[r]>0 AND (H1-r) <= c < Wd ; cells with r<=H1,c<Wd are in-grid
  outside in-grid -> sentinel 255 so Equal yields all-false (matches GT exactly)
Final one-hot via Equal(gm == colors) is the graph OUTPUT (excluded from memory).

All interior value tensors are uint8 (colors/lengths/indices 0..11) and masks are
bool -> tiny. Only unavoidable big interior tensor is the f32 color-grid collapse.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
HH = 12   # crop height (max example H = 12)
WW = 9    # crop width  (max example W = 9)


F16 = TensorProto.FLOAT16


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

    # color grid from one-hot input (input excluded) -> g30 f32 [1,1,30,30]
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    # crop to HHxWW (f32), then cast straight to uint8 (colors 0..9 exact)
    inits += [ci("cs", [0, 0]), ci("ce", [HH, WW]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["gf"]))  # f32 [1,1,HH,WW]
    nodes.append(helper.make_node("Cast", ["gf"], ["g"], to=U8))                # u8 [1,1,HH,WW]

    inits.append(cu("zero", 0))
    inits.append(ci("ax3", [3]))
    inits.append(ci("axall", [2, 3]))
    inits.append(ci("ax2", [2]))

    # nz = g>0   (bool)  -> cast to f16 only for the ReduceSum (ORT lacks uint8 ReduceSum)
    nodes.append(helper.make_node("Greater", ["g", "zero"], ["nzb"]))           # bool [1,1,HH,WW]
    nodes.append(helper.make_node("Cast", ["nzb"], ["nzf"], to=F16))           # f16 [1,1,HH,WW]

    # Lr = sum over cols (f16) -> cast u8 ; Cr = max over cols (u8) -> [1,1,HH,1]
    nodes.append(helper.make_node("ReduceSum", ["nzf", "ax3"], ["Lrf"], keepdims=1))  # f16 [1,1,HH,1]
    nodes.append(helper.make_node("Cast", ["Lrf"], ["Lr"], to=U8))            # u8 [1,1,HH,1]
    nodes.append(helper.make_node("ReduceMax", ["g", "ax3"], ["Cr"], keepdims=1))     # u8 [1,1,HH,1]

    # rowhas = Lr>0
    nodes.append(helper.make_node("Greater", ["Lr", "zero"], ["rowhasb"]))      # bool [1,1,HH,1]
    nodes.append(helper.make_node("Cast", ["rowhasb"], ["rowhas"], to=U8))

    # H1 = max row index where rowhas ; Wd = max(Lr)
    rowidx = np.arange(HH, dtype=np.uint8).reshape(1, 1, HH, 1)
    inits.append(cu("rowidx", rowidx))
    nodes.append(helper.make_node("Mul", ["rowhas", "rowidx"], ["rowidx_m"]))   # u8 [1,1,HH,1]
    nodes.append(helper.make_node("ReduceMax", ["rowidx_m", "axall"], ["H1"], keepdims=1))  # u8 [1,1,1,1]
    nodes.append(helper.make_node("ReduceMax", ["Lr", "axall"], ["Wd"], keepdims=1))        # u8 [1,1,1,1]

    # T = H1 + Lr - Wd  (compute as H1 + Lr then subtract Wd; all small nonneg uint8)
    nodes.append(helper.make_node("Add", ["H1", "Lr"], ["H1pLr"]))             # u8 [1,1,HH,1]
    nodes.append(helper.make_node("Sub", ["H1pLr", "Wd"], ["T"]))             # u8 [1,1,HH,1]

    # match[i,r] = (T_i == r). outrow [1,1,1,HH]
    outrow = np.arange(HH, dtype=np.uint8).reshape(1, 1, 1, HH)
    inits.append(cu("outrow", outrow))
    nodes.append(helper.make_node("Equal", ["T", "outrow"], ["matchb"]))        # bool [1,1,HH,HH]
    # wCr = rowhas * Cr  (drops empty rows)  [1,1,HH,1]  -> f16
    nodes.append(helper.make_node("Mul", ["rowhas", "Cr"], ["wCru"]))         # u8
    nodes.append(helper.make_node("Cast", ["wCru"], ["wCr"], to=F16))         # f16 [1,1,HH,1]
    # contrib = where(matchb, wCr, 0)  -> picks each bar's color into its target out-row
    inits.append(c16("zf", 0.0))
    nodes.append(helper.make_node("Where", ["matchb", "wCr", "zf"], ["contrib"]))  # f16 [1,1,HH,HH]
    nodes.append(helper.make_node("ReduceSum", ["contrib", "ax2"], ["OCr_rf"], keepdims=1))  # f16 [1,1,1,HH]
    nodes.append(helper.make_node("Cast", ["OCr_rf"], ["OCr_r"], to=U8))      # u8 [1,1,1,HH]
    nodes.append(helper.make_node("Transpose", ["OCr_r"], ["OCr"], perm=[0, 1, 3, 2]))      # u8 [1,1,HH,1]

    # staircase: (H1-r) <= c < Wd  AND  OCr[r]>0
    colidx = np.arange(WW, dtype=np.uint8).reshape(1, 1, 1, WW)
    inits.append(cu("colidx", colidx))
    nodes.append(helper.make_node("Sub", ["H1", "rowidx"], ["thr"]))           # u8 [1,1,HH,1]
    nodes.append(helper.make_node("GreaterOrEqual", ["colidx", "thr"], ["geo"]))   # bool [1,1,HH,WW]
    nodes.append(helper.make_node("Less", ["colidx", "Wd"], ["cltW"]))          # bool [1,1,1,WW]
    nodes.append(helper.make_node("And", ["geo", "cltW"], ["geo2"]))            # bool [1,1,HH,WW]
    nodes.append(helper.make_node("Greater", ["OCr", "zero"], ["ocnz"]))        # bool [1,1,HH,1]
    nodes.append(helper.make_node("And", ["geo2", "ocnz"], ["stairb"]))         # bool [1,1,HH,WW]
    nodes.append(helper.make_node("Cast", ["stairb"], ["stair"], to=U8))       # u8

    # gm = stair * OCr  -> color grid [1,1,HH,WW] uint8
    nodes.append(helper.make_node("Mul", ["stair", "OCr"], ["gmu"]))           # u8 [1,1,HH,WW]

    # in-grid mask: r <= H1  AND  c < Wd ; outside -> sentinel 255
    nodes.append(helper.make_node("LessOrEqual", ["rowidx", "H1"], ["rle"]))    # bool [1,1,HH,1]
    nodes.append(helper.make_node("And", ["rle", "cltW"], ["ingridb"]))         # bool [1,1,HH,WW]
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Where", ["ingridb", "gmu", "sent"], ["gms"]))  # u8 [1,1,HH,WW]

    # pad to 30x30 with sentinel 255
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - HH, 30 - WW]))
    nodes.append(helper.make_node("Pad", ["gms", "pads", "sent"], ["gm30"]))    # u8 [1,1,30,30]

    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))     # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t301", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b15_task301.onnx")
