"""Build compact ONNX for task004.

Rule (verified 265/265, 3x3 cellular automaton, generalizes with 0 unseen windows):
The colored shape is a right-leaning parallelogram outline. Output = shift the shape
right by 1 with edge corrections. Decomposed as two independent, exact local rules:

  COLOR:  oc(r,c) = g(r,c-1) if g(r,c-1)!=0 else g(r,c)
  MASK :  3x3 cellular-automaton on the binary (g!=0) mask.

We collapse the one-hot input to a single color grid g [1,1,30,30] (excluded input),
optionally crop to SxS, run the whole pipeline in uint8/bool, build the final color
grid fc (uint8, 0 = background), and emit  Equal(fc, colors[1,10,1,1])  as the graph
output (excluded) so no interior [1,10,*] tensor is ever materialized.

Two membership variants for the MASK rule:
  - "gather": one Conv -> 3x3 code, Cast to int, Gather a 512-entry {0,1} table.
  - "logic" : closed-form boolean   mo = W & ~sub | add   over 8 neighbor masks.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
I32 = TensorProto.INT32


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu8(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cb(name, v):
    return numpy_helper.from_array(np.asarray(v, np.bool_), name=name)


# 3x3 ON codes (row-major place values 256..1), verified.
ON_CODES = [34, 35, 49, 56, 60, 98, 99, 120, 176, 278, 280, 482, 483]


def build(path, S=16, variant="gather"):
    nodes, inits = [], []

    # 1) Single conv encodes color AND in-grid membership: weight channel c by (c+1).
    #    out-of-grid (all channels 0) -> 0 ; in-grid color c -> c+1 (range 1..10).
    inits.append(cf("Wcol", (np.arange(10) + 1).astype(np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["v30"]))  # f32 [1,1,30,30], values 0..10

    # Cast the only 30x30 f32 tensor straight to uint8 (values 0..10 exact), then crop.
    nodes.append(helper.make_node("Cast", ["v30"], ["v30u"], to=U8))  # u8 [1,1,30,30]
    if S < 30:
        inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
        nodes.append(helper.make_node("Slice", ["v30u", "cs", "ce", "ca"], ["g"]))  # u8 [1,1,S,S]
    else:
        nodes.append(helper.make_node("Identity", ["v30u"], ["g"]))
    # Work entirely in "color+1" space (value v = color+1; bg=1, color c -> c+1, oog=0).
    inits.append(cu8("u0", 0))
    inits.append(cu8("u1", 1))
    # in-grid mask: g>0
    nodes.append(helper.make_node("Greater", ["g", "u0"], ["ingrid"]))  # bool [1,1,S,S]
    # foreground mask mi: color != 0  <=>  value > 1
    nodes.append(helper.make_node("Greater", ["g", "u1"], ["mi"]))  # bool [1,1,S,S]

    # ---- MASK rule ----
    if variant in ("gather", "gather4"):
        # mask -> float (f16 keeps the code tensors small).
        nodes.append(helper.make_node("Cast", ["mi"], ["mif"], to=F16))  # f16 [1,1,S,S]
        if variant == "gather":
            # full 3x3 code (place values 256..1), 512-entry LUT.
            pv = np.array([[256, 128, 64], [32, 16, 8], [4, 2, 1]], np.float16).reshape(1, 1, 3, 3)
            inits.append(numpy_helper.from_array(pv, "Wpv"))
            nodes.append(helper.make_node("Conv", ["mif", "Wpv"], ["codef"], pads=[1, 1, 1, 1]))  # f16
            table = np.zeros(512, np.bool_)
            for c in ON_CODES:
                table[c] = True
        else:
            # 4-neighbour code {W,C,E,S} (place values 8,4,2,1) -> 16-entry LUT.
            # kernel (3x3), taps at W=(0,-1),C=(0,0),E=(0,1),S=(1,0):
            #   row0: 0 0 0 ; row1: W C E = 8 4 2 ; row2: 0 S 0 = 0 1 0
            pv = np.array([[0, 0, 0], [8, 4, 2], [0, 1, 0]], np.float16).reshape(1, 1, 3, 3)
            inits.append(numpy_helper.from_array(pv, "Wpv"))
            nodes.append(helper.make_node("Conv", ["mif", "Wpv"], ["codef"], pads=[1, 1, 1, 1]))  # f16
            # ON 4-bit codes {5,6,9,12,14}; unseen {10,11,13,15} filled per geometry:
            #   10:W,E   11:W,E,S   13:W,C,S   15:all  -> set unseen to match shift-right base (W=1 => 1)
            table = np.zeros(16, np.bool_)
            for c in [5, 6, 9, 12, 14]:
                table[c] = True
            for c in [10, 11, 13, 15]:   # all have W=1 -> base shift keeps them ON
                table[c] = True
        inits.append(cb("LUT", table))
        nodes.append(helper.make_node("Cast", ["codef"], ["code"], to=I32))  # int32 [1,1,S,S]
        nodes.append(helper.make_node("Gather", ["LUT", "code"], ["mo"], axis=0))  # bool [1,1,S,S]
    else:  # logic
        # pad mask by 1 then slice 8 neighbors + center
        inits.append(ci("p1", [0, 0, 1, 1, 0, 0, 1, 1]))
        inits.append(cb("pf", False))
        nodes.append(helper.make_node("Pad", ["mi", "p1", "pf"], ["miP"]))  # bool [1,1,S+2,S+2]

        def sl(name, r0, c0):
            inits.append(ci(name + "_s", [r0, c0]))
            inits.append(ci(name + "_e", [r0 + S, c0 + S]))
            inits.append(ci(name + "_a", [2, 3]))
            nodes.append(helper.make_node("Slice", ["miP", name + "_s", name + "_e", name + "_a"], [name]))

        # neighbor at (dr,dc): slice top-left = (1+dr, 1+dc) ... but pad index:
        # center C = slice starting (1,1). W (left, dc=-1) = start (1,0). etc.
        sl("NW", 0, 0); sl("N", 0, 1); sl("NE", 0, 2)
        sl("W", 1, 0); sl("C", 1, 1); sl("E", 1, 2)
        sl("SW", 2, 0); sl("S", 2, 1); sl("SE", 2, 2)

        # not-masks needed
        def NOT(a):
            nodes.append(helper.make_node("Not", [a], ["n_" + a]))
            return "n_" + a

        # sub = W & ~C & ~N & ~NE & ~E & ~S & ~SE & (SW xor NW)
        nodes.append(helper.make_node("Xor", ["SW", "NW"], ["swnw"]))
        # AND chain of: W, ~C, ~N, ~NE, ~E, ~S, ~SE, swnw
        nC = NOT("C"); nN = NOT("N"); nNE = NOT("NE"); nE = NOT("E"); nS = NOT("S"); nSE = NOT("SE")
        nodes.append(helper.make_node("And", ["W", nC], ["a1"]))
        nodes.append(helper.make_node("And", ["a1", nN], ["a2"]))
        nodes.append(helper.make_node("And", ["a2", nNE], ["a3"]))
        nodes.append(helper.make_node("And", ["a3", nE], ["a4"]))
        nodes.append(helper.make_node("And", ["a4", nS], ["a5"]))
        nodes.append(helper.make_node("And", ["a5", nSE], ["a6"]))
        nodes.append(helper.make_node("And", ["a6", "swnw"], ["sub"]))

        # add = ~W & C & ~N & ~NE & NW & ((SW&S&~E)|(E&~SW&~S&~SE))
        nW = NOT("W"); nSW = NOT("SW")
        nodes.append(helper.make_node("And", ["SW", "S"], ["b1"]))
        nodes.append(helper.make_node("And", ["b1", nE], ["b2"]))      # SW&S&~E
        nodes.append(helper.make_node("And", ["E", nSW], ["c1"]))
        nodes.append(helper.make_node("And", ["c1", nS], ["c2"]))
        nodes.append(helper.make_node("And", ["c2", nSE], ["c3"]))     # E&~SW&~S&~SE
        nodes.append(helper.make_node("Or", ["b2", "c3"], ["d1"]))
        nodes.append(helper.make_node("And", [nW, "C"], ["e1"]))
        nodes.append(helper.make_node("And", ["e1", nN], ["e2"]))
        nodes.append(helper.make_node("And", ["e2", nNE], ["e3"]))
        nodes.append(helper.make_node("And", ["e3", "NW"], ["e4"]))
        nodes.append(helper.make_node("And", ["e4", "d1"], ["add"]))

        nodes.append(helper.make_node("Not", ["sub"], ["nsub"]))
        nodes.append(helper.make_node("And", ["W", "nsub"], ["m1"]))
        nodes.append(helper.make_node("Or", ["m1", "add"], ["mo"]))    # bool [1,1,S,S]

    # ---- COLOR rule (in color+1 space) ----
    # gW = shift g right by 1 (pad left with 0 -> oog/edge value 0, drop last col)
    inits.append(ci("pgl", [0, 0, 0, 1, 0, 0, 0, 0]))   # pad 1 on left (W dim)
    nodes.append(helper.make_node("Pad", ["g", "pgl", "u0"], ["gWpad"]))  # u8 [1,1,S,S+1]
    inits += [ci("gws", [0]), ci("gwe", [S]), ci("gwa", [3])]
    nodes.append(helper.make_node("Slice", ["gWpad", "gws", "gwe", "gwa"], ["gW"]))  # u8 [1,1,S,S]
    # left is foreground  <=>  gW value > 1 ; oc = Where(leftfg, gW, g)
    nodes.append(helper.make_node("Greater", ["gW", "u1"], ["leftfg"]))  # bool [1,1,S,S]
    nodes.append(helper.make_node("Where", ["leftfg", "gW", "g"], ["oc"]))  # u8 value [1,1,S,S]
    # selected = mo ? oc : background(value 1)
    nodes.append(helper.make_node("Where", ["mo", "oc", "u1"], ["fc_c"]))  # u8 [1,1,S,S]
    # out-of-grid -> sentinel 255 (matches no channel)
    inits.append(cu8("sent", 255))
    nodes.append(helper.make_node("Where", ["ingrid", "fc_c", "sent"], ["fc_s"]))  # u8 [1,1,S,S]

    # pad back to 30x30 with sentinel 255 (outside crop is out-of-grid)
    if S < 30:
        inits.append(ci("pback", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
        nodes.append(helper.make_node("Pad", ["fc_s", "pback", "sent"], ["fc"]))  # u8 [1,1,30,30]
    else:
        nodes.append(helper.make_node("Identity", ["fc_s"], ["fc"]))

    # output one-hot: channel ch matches value==ch+1. Equal IS the graph output (excluded).
    inits.append(cu8("colors1", (np.arange(10) + 1).astype(np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["fc", "colors1"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t004", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "S", S, "variant", variant, "nodes", len(nodes))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task004.onnx"
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    variant = sys.argv[3] if len(sys.argv) > 3 else "gather"
    build(path, S, variant)
