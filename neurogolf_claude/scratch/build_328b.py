"""Compact ONNX for task328 — v2, all interior tensors uint8/bool (no int32 grids).

Two-level Voronoi (cheb then manh, tie->bg). Pattern per corner:
  pat = (E and (F or gt)) or (notE and F and notgt)
  where E=cc even, F=rr even, gt=cc>rr.
Output via Equal(gm_uint8, colors).
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


def build(path):
    nodes, inits = [], []

    # color grid g (uint8) and valid mask (bool)
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))          # u8 color grid
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["msum"]))
    inits.append(cf("zf", 0.0))
    nodes.append(helper.make_node("Greater", ["msum", "zf"], ["valid"]))    # bool valid
    nodes.append(helper.make_node("Cast", ["valid"], ["validu"], to=U8))

    # index grids (uint8 constants)
    rI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, N, 1), (1, 1, N, N)).copy()
    cI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, 1, N), (1, 1, N, N)).copy()
    inits.append(cu("rI", rI))
    inits.append(cu("cI", cI))
    # parity constants for fixed axes (checkerboard), as bool: even index?
    evrI = (rI % 2 == 0)
    evcI = (cI % 2 == 0)
    inits.append(numpy_helper.from_array(evrI.astype(np.bool_), "evrI"))
    inits.append(numpy_helper.from_array(evcI.astype(np.bool_), "evcI"))

    # R = rr_bottom, C = cc_right via reverse cumsum of valid (int32) then -1, cast u8
    nodes.append(helper.make_node("Cast", ["validu"], ["v32"], to=I32))
    inits.append(ci("axW", 3))
    inits.append(ci("axH", 2))
    nodes.append(helper.make_node("CumSum", ["v32", "axW"], ["rincl"], reverse=1))
    nodes.append(helper.make_node("CumSum", ["v32", "axH"], ["bincl"], reverse=1))
    nodes.append(helper.make_node("Cast", ["rincl"], ["rinclu"], to=U8))
    nodes.append(helper.make_node("Cast", ["bincl"], ["binclu"], to=U8))
    inits.append(cu("one", 1))
    nodes.append(helper.make_node("Sub", ["rinclu", "one"], ["C"]))   # u8 cc_right
    nodes.append(helper.make_node("Sub", ["binclu", "one"], ["R"]))   # u8 rr_bottom
    # parity of dynamic axes
    inits.append(cu("two", 2))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Mod", ["C", "two"], ["Cm"]))
    nodes.append(helper.make_node("Mod", ["R", "two"], ["Rm"]))
    nodes.append(helper.make_node("Equal", ["Cm", "zero"], ["evC"]))   # bool
    nodes.append(helper.make_node("Equal", ["Rm", "zero"], ["evR"]))   # bool

    # corner colors (i32 reduce then u8 scalar)
    nodes.append(helper.make_node("Cast", ["g"], ["gi"], to=I32))
    inits.append(ci("axHW", [2, 3]))

    def ccolor(name, m1, m2):  # m1,m2 bool indicator components (within valid handled separately)
        nodes.append(helper.make_node("And", [m1, m2], [name + "_ib"]))
        nodes.append(helper.make_node("Cast", [name + "_ib"], [name + "_ii"], to=I32))
        nodes.append(helper.make_node("Mul", ["gi", name + "_ii"], [name + "_p"]))
        nodes.append(helper.make_node("ReduceSum", [name + "_p", "axHW"], [name + "_c32"], keepdims=1))
        nodes.append(helper.make_node("Cast", [name + "_c32"], [name + "_col"], to=U8))  # scalar u8

    # corner-presence indicators: rI==0 -> r0 etc. C==0 -> at right edge (within valid).
    nodes.append(helper.make_node("Equal", ["rI", "zero"], ["r0"]))   # bool
    nodes.append(helper.make_node("Equal", ["cI", "zero"], ["c0"]))
    nodes.append(helper.make_node("Equal", ["R", "zero"], ["R0a"]))
    nodes.append(helper.make_node("Equal", ["C", "zero"], ["C0a"]))
    nodes.append(helper.make_node("And", ["R0a", "valid"], ["R0"]))
    nodes.append(helper.make_node("And", ["C0a", "valid"], ["C0"]))
    ccolor("TL", "r0", "c0")
    ccolor("TR", "r0", "C0")
    ccolor("BL", "c0", "R0")
    ccolor("BR", "R0", "C0")

    BIG = 200
    inits.append(cu("BIG", BIG))

    # per corner: cheb, manh (u8); pattern bool; presence
    corners = [
        ("TL", "rI", "cI", "evrI", "evcI"),
        ("TR", "rI", "C", "evrI", "evC"),
        ("BL", "R", "cI", "evR", "evcI"),
        ("BR", "R", "C", "evR", "evC"),
    ]
    chs, mns, pats = [], [], []
    for nm, rr, cc, evr, evc in corners:
        # presence: col != 0
        nodes.append(helper.make_node("Greater", [nm + "_col", "zero"], [nm + "_on"]))  # bool scalar
        # cheb, manh
        nodes.append(helper.make_node("Max", [rr, cc], [nm + "_ch0"]))
        nodes.append(helper.make_node("Add", [rr, cc], [nm + "_mn0"]))
        # if off -> BIG
        nodes.append(helper.make_node("Where", [nm + "_on", nm + "_ch0", "BIG"], [nm + "_ch"]))  # u8
        nodes.append(helper.make_node("Where", [nm + "_on", nm + "_mn0", "BIG"], [nm + "_mn"]))  # u8
        chs.append(nm + "_ch"); mns.append(nm + "_mn")
        # pattern: gt = cc>rr ; pat = (E and (F or gt)) or (notE and F and notgt)
        nodes.append(helper.make_node("Greater", [cc, rr], [nm + "_gt"]))
        nodes.append(helper.make_node("Not", [nm + "_gt"], [nm + "_ngt"]))
        nodes.append(helper.make_node("Or", [evr, nm + "_gt"], [nm + "_forgt"]))
        nodes.append(helper.make_node("And", [evc, nm + "_forgt"], [nm + "_a1"]))
        nodes.append(helper.make_node("Not", [evc], [nm + "_nE"]))
        nodes.append(helper.make_node("And", [evr, nm + "_ngt"], [nm + "_a2t"]))
        nodes.append(helper.make_node("And", [nm + "_nE", nm + "_a2t"], [nm + "_a2"]))
        nodes.append(helper.make_node("Or", [nm + "_a1", nm + "_a2"], [nm + "_pat"]))  # bool
        pats.append(nm + "_pat")

    # mincheb
    nodes.append(helper.make_node("Min", chs, ["mincheb"]))  # u8
    # manh effective: where cheb==mincheb keep manh else BIG ; then min
    mnames = []
    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("Equal", [chs[i], "mincheb"], [nm + "_isch"]))  # bool
        nodes.append(helper.make_node("Where", [nm + "_isch", mns[i], "BIG"], [nm + "_mne"]))  # u8
        mnames.append(nm + "_mne")
    nodes.append(helper.make_node("Min", mnames, ["minmanh"]))  # u8

    # winners + accumulate color
    accparts = []
    nwins = []
    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("Equal", [mns[i], "minmanh"], [nm + "_ismn"]))  # bool
        nodes.append(helper.make_node("And", [nm + "_isch", nm + "_ismn"], [nm + "_win"]))  # bool
        nodes.append(helper.make_node("Cast", [nm + "_win"], [nm + "_winu"], to=U8))
        nwins.append(nm + "_winu")
    # nwin = sum winu (u8 via Add chain)
    nodes.append(helper.make_node("Add", [nwins[0], nwins[1]], ["nw01"]))
    nodes.append(helper.make_node("Add", [nwins[2], nwins[3]], ["nw23"]))
    nodes.append(helper.make_node("Add", ["nw01", "nw23"], ["nwin"]))  # u8
    nodes.append(helper.make_node("Equal", ["nwin", "one"], ["uniq"]))  # bool

    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("And", [nm + "_win", "uniq"], [nm + "_w2"]))
        nodes.append(helper.make_node("And", [nm + "_w2", pats[i]], [nm + "_fill"]))
        nodes.append(helper.make_node("Cast", [nm + "_fill"], [nm + "_fu"], to=U8))
        nodes.append(helper.make_node("Mul", [nm + "_fu", nm + "_col"], [nm + "_part"]))  # u8
        accparts.append(nm + "_part")
    nodes.append(helper.make_node("Add", [accparts[0], accparts[1]], ["ac01"]))
    nodes.append(helper.make_node("Add", [accparts[2], accparts[3]], ["ac23"]))
    nodes.append(helper.make_node("Add", ["ac01", "ac23"], ["gmin"]))  # u8 color (0 bg)

    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Where", ["valid", "gmin", "sent"], ["gm"]))  # u8

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
