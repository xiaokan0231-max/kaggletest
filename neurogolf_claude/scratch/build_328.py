"""Build compact ONNX for task328.

Rule (verified 262/262 arc-gen + 4 train + 1 test):
  Colored pixels appear only at the (<=4) grid corners. Grid is NxN at top-left.
  For each cell, owner = nearest corner by Chebyshev dist, tie-broken by Manhattan;
  if still tied -> background. Within owner, with corner-relative (rr down, cc side):
    even rr: filled iff (cc even) or (cc<=rr)
    odd  rr: filled iff (cc even) and (cc> rr)
  Filled cell takes owner corner color; else background 0.

ONNX strategy:
  - Conv collapse one-hot input -> color grid g f32 [1,1,30,30] (input excluded).
  - valid mask m = (sum of channels) >0  (Conv ones).
  - distances: r,c index constants; R=rr_bottom, C=cc_right via reverse CumSum of m.
  - 4 corners = (r|R) x (c|C). For each: cheb, manh, key=cheb*64+manh (inactive->BIG),
    ideal pattern (uint8). winner_i = (key_i==minkey) and (nwin==1).
    color accumulation gm = sum_i winner_i*pat_i*col_i, restricted to valid; out-of-grid=255.
  - Equal(gm, colors[0..9]) IS the graph output (excluded) -> bool [1,10,30,30].
All interior [1,1,30,30] tensors are uint8/bool to minimize memory.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
BOOL = TensorProto.BOOL
N = 30


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def ci32(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int32), name=name)


def build(path):
    nodes, inits = [], []

    # --- color grid g (f32, input excluded as it's the input tensor) ---
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))   # f32 [1,1,30,30]
    # valid mask via ones conv -> >0
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["msum"]))  # f32
    inits.append(cf("zero_f", 0.0))
    nodes.append(helper.make_node("Greater", ["msum", "zero_f"], ["validb"]))  # bool
    nodes.append(helper.make_node("Cast", ["validb"], ["validu"], to=U8))      # u8 valid 0/1

    # g as uint8 (colors exact 0..9)
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))  # u8 color grid

    # --- distance grids (uint8) ---
    # r index, c index constants
    r_idx = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, N, 1), (1, 1, N, N)).copy()
    c_idx = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, 1, N), (1, 1, N, N)).copy()
    inits.append(cu("rI", r_idx))
    inits.append(cu("cI", c_idx))
    # reverse cumsum of valid over width -> N..1 within grid, then -1 -> C = cc_right
    # use int32 for cumsum
    nodes.append(helper.make_node("Cast", ["validu"], ["valid32"], to=I32))
    inits.append(ci("axW", 3))
    inits.append(ci("axH", 2))
    nodes.append(helper.make_node("CumSum", ["valid32", "axW"], ["rincl"], reverse=1))   # i32
    nodes.append(helper.make_node("CumSum", ["valid32", "axH"], ["bincl"], reverse=1))   # i32
    inits.append(ci32("one32", 1))
    nodes.append(helper.make_node("Sub", ["rincl", "one32"], ["Ci"]))   # i32 cc_right
    nodes.append(helper.make_node("Sub", ["bincl", "one32"], ["Ri"]))   # i32 rr_bottom
    nodes.append(helper.make_node("Cast", ["Ci"], ["C"], to=U8))
    nodes.append(helper.make_node("Cast", ["Ri"], ["R"], to=U8))

    # --- corner colors broadcast (uint8 scalar->grid via ReduceSum of g*indicator) ---
    # indicators: TL r==0&c==0 ; TR r==0&C==0 ; BL c==0&R==0 ; BR R==0&C==0  (within valid)
    inits.append(cu("zerou", 0))
    nodes.append(helper.make_node("Equal", ["rI", "zerou"], ["r0"]))   # bool
    nodes.append(helper.make_node("Equal", ["cI", "zerou"], ["c0"]))
    nodes.append(helper.make_node("Equal", ["R", "zerou"], ["R0b"]))
    nodes.append(helper.make_node("Equal", ["C", "zerou"], ["C0b"]))
    # combine with valid for the dynamic ones
    nodes.append(helper.make_node("And", ["R0b", "validb"], ["R0"]))
    nodes.append(helper.make_node("And", ["C0b", "validb"], ["C0"]))

    nodes.append(helper.make_node("Cast", ["g"], ["g32i"], to=I32))  # color grid int32

    def corner_color(name, m1, m2):
        nodes.append(helper.make_node("And", [m1, m2], [name + "_ib"]))
        nodes.append(helper.make_node("Cast", [name + "_ib"], [name + "_i32"], to=I32))
        nodes.append(helper.make_node("Mul", ["g32i", name + "_i32"], [name + "_gi"]))   # i32
        nodes.append(helper.make_node("ReduceSum", [name + "_gi", "axHW"], [name + "_col32"], keepdims=1))  # i32 [1,1,1,1]
        nodes.append(helper.make_node("Cast", [name + "_col32"], [name + "_col"], to=U8))  # u8 scalar color

    inits.append(ci("axHW", [2, 3]))
    corner_color("TL", "r0", "c0")
    corner_color("TR", "r0", "C0")
    corner_color("BL", "c0", "R0")
    corner_color("BR", "R0", "C0")

    # --- per corner: key + pattern; combine ---
    # constants
    inits.append(cu("BIGk", 250))
    inits.append(cu("k64", 64))
    inits.append(cu("twou", 2))
    # need key in a type that supports cheb*64+manh up to ~1914 -> use int32 to be safe.
    inits.append(ci32("k64_32", 64))
    inits.append(ci32("BIG32", 60000))

    corners = [("TL", "rI", "cI"), ("TR", "rI", "C"), ("BL", "R", "cI"), ("BR", "R", "C")]
    keynames = []
    patnames = []
    for nm, rr, cc in corners:
        # cheb = max(rr,cc) ; manh = rr+cc   (uint8 ok: max 29+29=58)
        nodes.append(helper.make_node("Max", [rr, cc], [nm + "_ch"]))   # u8
        nodes.append(helper.make_node("Add", [rr, cc], [nm + "_mn"]))   # u8
        # key32 = cheb*64 + manh
        nodes.append(helper.make_node("Cast", [nm + "_ch"], [nm + "_ch32"], to=I32))
        nodes.append(helper.make_node("Cast", [nm + "_mn"], [nm + "_mn32"], to=I32))
        nodes.append(helper.make_node("Mul", [nm + "_ch32", "k64_32"], [nm + "_chk"]))
        nodes.append(helper.make_node("Add", [nm + "_chk", nm + "_mn32"], [nm + "_key0"]))
        # active = corner color != 0 ; if inactive set key=BIG. broadcast color scalar.
        nodes.append(helper.make_node("Equal", [nm + "_col", "zerou"], [nm + "_offb"]))  # bool [1,1,1,1]
        # where off -> BIG32 else key0
        nodes.append(helper.make_node("Where", [nm + "_offb", "BIG32", nm + "_key0"], [nm + "_key"]))  # i32
        keynames.append(nm + "_key")
        # ideal pattern (uint8 0/1):
        # ev_c = cc%2==0 ; ev_r = rr%2==0 ; even_case = ev_c or cc<=rr ; odd_case = ev_c and cc>rr
        nodes.append(helper.make_node("Mod", [cc, "twou"], [nm + "_ccm"]))  # u8
        nodes.append(helper.make_node("Mod", [rr, "twou"], [nm + "_rrm"]))  # u8
        nodes.append(helper.make_node("Equal", [nm + "_ccm", "zerou"], [nm + "_evc"]))  # bool
        nodes.append(helper.make_node("Equal", [nm + "_rrm", "zerou"], [nm + "_evr"]))  # bool
        # cc<=rr  -> Not(cc>rr)
        nodes.append(helper.make_node("Greater", [cc, rr], [nm + "_ccgt"]))  # cc>rr bool
        nodes.append(helper.make_node("Not", [nm + "_ccgt"], [nm + "_ccle"]))  # cc<=rr
        nodes.append(helper.make_node("Or", [nm + "_evc", nm + "_ccle"], [nm + "_evencase"]))
        nodes.append(helper.make_node("And", [nm + "_evc", nm + "_ccgt"], [nm + "_oddcase"]))
        # pat = (evr AND evencase) OR ((NOT evr) AND oddcase)
        nodes.append(helper.make_node("And", [nm + "_evr", nm + "_evencase"], [nm + "_p1"]))
        nodes.append(helper.make_node("Not", [nm + "_evr"], [nm + "_oddr"]))
        nodes.append(helper.make_node("And", [nm + "_oddr", nm + "_oddcase"], [nm + "_p2"]))
        nodes.append(helper.make_node("Or", [nm + "_p1", nm + "_p2"], [nm + "_patb"]))  # bool
        patnames.append(nm + "_patb")

    # minkey over 4 keys
    nodes.append(helper.make_node("Min", keynames, ["minkey"]))  # i32
    # nwin = sum_i (key_i==minkey)
    eqnames = []
    for i, kn in enumerate(keynames):
        nodes.append(helper.make_node("Equal", [kn, "minkey"], [f"iseq{i}b"]))  # bool
        nodes.append(helper.make_node("Cast", [f"iseq{i}b"], [f"iseq{i}u"], to=I32))
        eqnames.append((f"iseq{i}b", f"iseq{i}u"))
    nodes.append(helper.make_node("Add", [eqnames[0][1], eqnames[1][1]], ["nwin01"]))
    nodes.append(helper.make_node("Add", [eqnames[2][1], eqnames[3][1]], ["nwin23"]))
    nodes.append(helper.make_node("Add", ["nwin01", "nwin23"], ["nwin"]))  # i32
    nodes.append(helper.make_node("Equal", ["nwin", "one32"], ["uniqb"]))  # bool (exactly one winner)

    # accumulate color: sum_i (winner_i and pat_i) * col_i
    accparts = []
    for i, (nm, _, _) in enumerate(corners):
        eqb = eqnames[i][0]
        nodes.append(helper.make_node("And", [eqb, "uniqb"], [f"win{i}"]))       # winner unique
        nodes.append(helper.make_node("And", [f"win{i}", patnames[i]], [f"fill{i}"]))  # bool
        nodes.append(helper.make_node("Cast", [f"fill{i}"], [f"fill{i}u"], to=I32))
        nodes.append(helper.make_node("Mul", [f"fill{i}u", nm + "_col32"], [f"part{i}"]))  # i32
        accparts.append(f"part{i}")
    nodes.append(helper.make_node("Add", [accparts[0], accparts[1]], ["acc01"]))
    nodes.append(helper.make_node("Add", [accparts[2], accparts[3]], ["acc23"]))
    nodes.append(helper.make_node("Add", ["acc01", "acc23"], ["gm_in32"]))  # i32 color grid
    nodes.append(helper.make_node("Cast", ["gm_in32"], ["gm_in"], to=U8))  # u8

    # restrict to valid (already only valid cells get colored, but bg color 0 in-grid must stay 0
    # and out-of-grid must become 255 sentinel).
    # out-of-grid = not valid -> set 255
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Where", ["validb", "gm_in", "sent"], ["gm"]))  # u8 [1,1,30,30]

    # Equal-as-output
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t328", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b20_task328.onnx")