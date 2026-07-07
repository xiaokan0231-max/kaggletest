"""Compact ONNX for task328 — v4. Manhattan Voronoi + OWNER-COLLAPSE.

Collapse the winning corner's (rr,cc,color) into single owner grids, then compute
the pattern ONCE. All interior tensors uint8/bool except small cumsum/color reduces.
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

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["msum"]))
    inits.append(cf("zf", 0.0))
    nodes.append(helper.make_node("Greater", ["msum", "zf"], ["valid"]))
    nodes.append(helper.make_node("Cast", ["valid"], ["validu"], to=U8))

    rI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, N, 1), (1, 1, N, N)).copy()
    cI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, 1, N), (1, 1, N, N)).copy()
    inits.append(cu("rI", rI))
    inits.append(cu("cI", cI))

    nodes.append(helper.make_node("Cast", ["validu"], ["v32"], to=I32))
    inits.append(ci("axW", 3))
    inits.append(ci("axH", 2))
    nodes.append(helper.make_node("CumSum", ["v32", "axW"], ["rincl"], reverse=1))
    nodes.append(helper.make_node("CumSum", ["v32", "axH"], ["bincl"], reverse=1))
    nodes.append(helper.make_node("Cast", ["rincl"], ["rinclu"], to=U8))
    nodes.append(helper.make_node("Cast", ["bincl"], ["binclu"], to=U8))
    inits.append(cu("one", 1))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Sub", ["rinclu", "one"], ["C"]))
    nodes.append(helper.make_node("Sub", ["binclu", "one"], ["R"]))

    # corner colors via i32 reduce
    nodes.append(helper.make_node("Cast", ["g"], ["gi"], to=I32))
    inits.append(ci("axHW", [2, 3]))
    nodes.append(helper.make_node("Equal", ["rI", "zero"], ["r0"]))
    nodes.append(helper.make_node("Equal", ["cI", "zero"], ["c0"]))
    nodes.append(helper.make_node("Equal", ["R", "zero"], ["R0a"]))
    nodes.append(helper.make_node("Equal", ["C", "zero"], ["C0a"]))
    nodes.append(helper.make_node("And", ["R0a", "valid"], ["R0"]))
    nodes.append(helper.make_node("And", ["C0a", "valid"], ["C0"]))

    def ccolor(name, m1, m2):
        nodes.append(helper.make_node("And", [m1, m2], [name + "_ib"]))
        nodes.append(helper.make_node("Cast", [name + "_ib"], [name + "_ii"], to=I32))
        nodes.append(helper.make_node("Mul", ["gi", name + "_ii"], [name + "_p"]))
        nodes.append(helper.make_node("ReduceSum", [name + "_p", "axHW"], [name + "_c32"], keepdims=1))
        nodes.append(helper.make_node("Cast", [name + "_c32"], [name + "_col"], to=U8))

    ccolor("TL", "r0", "c0")
    ccolor("TR", "r0", "C0")
    ccolor("BL", "c0", "R0")
    ccolor("BR", "R0", "C0")

    inits.append(cu("BIG", 200))
    corners = [("TL", "rI", "cI"), ("TR", "rI", "C"), ("BL", "R", "cI"), ("BR", "R", "C")]
    keys, winflags = [], []
    for nm, rr, cc in corners:
        nodes.append(helper.make_node("Greater", [nm + "_col", "zero"], [nm + "_on"]))
        nodes.append(helper.make_node("Add", [rr, cc], [nm + "_mn0"]))
        nodes.append(helper.make_node("Where", [nm + "_on", nm + "_mn0", "BIG"], [nm + "_key"]))
        keys.append(nm + "_key")
    nodes.append(helper.make_node("Min", keys, ["minkey"]))
    nwins = []
    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("Equal", [keys[i], "minkey"], [nm + "_winb"]))   # bool
        nodes.append(helper.make_node("Cast", [nm + "_winb"], [nm + "_win"], to=U8))   # u8 0/1
        winflags.append(nm + "_win")
    nodes.append(helper.make_node("Add", [winflags[0], winflags[1]], ["nw01"]))
    nodes.append(helper.make_node("Add", [winflags[2], winflags[3]], ["nw23"]))
    nodes.append(helper.make_node("Add", ["nw01", "nw23"], ["nwin"]))
    nodes.append(helper.make_node("Equal", ["nwin", "one"], ["uniq"]))  # bool

    # owner rr, cc, col = sum winflag * value  (u8)
    def owner(name, vals):
        parts = []
        for i, (nm, *_ ) in enumerate(corners):
            nodes.append(helper.make_node("Mul", [winflags[i], vals[i]], [f"{name}_p{i}"]))
            parts.append(f"{name}_p{i}")
        nodes.append(helper.make_node("Add", [parts[0], parts[1]], [f"{name}_01"]))
        nodes.append(helper.make_node("Add", [parts[2], parts[3]], [f"{name}_23"]))
        nodes.append(helper.make_node("Add", [f"{name}_01", f"{name}_23"], [name]))

    owner("orr", ["rI", "rI", "R", "R"])
    owner("occ", ["cI", "C", "cI", "C"])
    owner("ocol", ["TL_col", "TR_col", "BL_col", "BR_col"])

    # single pattern from (orr, occ)
    inits.append(cu("two", 2))
    nodes.append(helper.make_node("Mod", ["orr", "two"], ["orm"]))
    nodes.append(helper.make_node("Mod", ["occ", "two"], ["ocm"]))
    nodes.append(helper.make_node("Equal", ["orm", "zero"], ["evr"]))   # F
    nodes.append(helper.make_node("Equal", ["ocm", "zero"], ["evc"]))   # E
    nodes.append(helper.make_node("Greater", ["occ", "orr"], ["gt"]))
    nodes.append(helper.make_node("Not", ["gt"], ["ngt"]))
    nodes.append(helper.make_node("Or", ["evr", "gt"], ["forgt"]))
    nodes.append(helper.make_node("And", ["evc", "forgt"], ["a1"]))
    nodes.append(helper.make_node("Not", ["evc"], ["nE"]))
    nodes.append(helper.make_node("And", ["evr", "ngt"], ["a2t"]))
    nodes.append(helper.make_node("And", ["nE", "a2t"], ["a2"]))
    nodes.append(helper.make_node("Or", ["a1", "a2"], ["pat"]))   # bool

    # fill = uniq & pat -> color = ocol if fill else 0 ; out-of-grid -> 255
    nodes.append(helper.make_node("And", ["uniq", "pat"], ["fill"]))
    nodes.append(helper.make_node("Cast", ["fill"], ["fillu"], to=U8))
    nodes.append(helper.make_node("Mul", ["fillu", "ocol"], ["gmin"]))  # u8 color (0 bg)
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Where", ["valid", "gmin", "sent"], ["gm"]))
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

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
