"""Compact ONNX for task328 — v7. Manhattan Voronoi + owner-collapse via top/left-won.

Static shapes throughout (genuine memory measurement, no fallback).
Distances: rI,cI consts; R,C via reverse CumSum(f16). Colors via f16 masked ReduceSum.
Owner select: top_won/left_won pick rr/cc/col with 2 Where each (cheap).
Single pattern. Output Equal(gm_u8, colors).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
N = 30


def cf(n, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=n)


def cu(n, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=n)


def ci(n, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=n)


def build(path):
    nodes, inits = [], []

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))
    nodes.append(helper.make_node("Cast", ["g32"], ["gh"], to=F16))
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["msum"]))
    inits.append(cf("zf", 0.0))
    nodes.append(helper.make_node("Greater", ["msum", "zf"], ["valid"]))
    nodes.append(helper.make_node("Cast", ["valid"], ["validh"], to=F16))

    rI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, N, 1), (1, 1, N, N)).copy()
    cI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, 1, N), (1, 1, N, N)).copy()
    inits.append(cu("rI", rI))
    inits.append(cu("cI", cI))

    inits.append(ci("axW", 3))
    inits.append(ci("axH", 2))
    nodes.append(helper.make_node("CumSum", ["validh", "axW"], ["rinclh"], reverse=1))
    nodes.append(helper.make_node("CumSum", ["validh", "axH"], ["binclh"], reverse=1))
    nodes.append(helper.make_node("Cast", ["rinclh"], ["rinclu"], to=U8))
    nodes.append(helper.make_node("Cast", ["binclh"], ["binclu"], to=U8))
    inits.append(cu("one", 1))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Sub", ["rinclu", "one"], ["C"]))
    nodes.append(helper.make_node("Sub", ["binclu", "one"], ["R"]))

    # colors via f16 masked reduce
    inits.append(ci("axHW", [2, 3]))
    nodes.append(helper.make_node("Equal", ["rI", "zero"], ["r0"]))
    nodes.append(helper.make_node("Equal", ["cI", "zero"], ["c0"]))
    nodes.append(helper.make_node("Equal", ["R", "zero"], ["R0a"]))
    nodes.append(helper.make_node("Equal", ["C", "zero"], ["C0a"]))
    nodes.append(helper.make_node("And", ["R0a", "valid"], ["R0"]))
    nodes.append(helper.make_node("And", ["C0a", "valid"], ["C0"]))

    def ccolor(name, m1, m2):
        nodes.append(helper.make_node("And", [m1, m2], [name + "_ib"]))
        nodes.append(helper.make_node("Cast", [name + "_ib"], [name + "_ih"], to=F16))
        nodes.append(helper.make_node("Mul", ["gh", name + "_ih"], [name + "_p"]))
        nodes.append(helper.make_node("ReduceSum", [name + "_p", "axHW"], [name + "_ch"], keepdims=1))
        nodes.append(helper.make_node("Cast", [name + "_ch"], [name + "_col"], to=U8))

    ccolor("TL", "r0", "c0")
    ccolor("TR", "r0", "C0")
    ccolor("BL", "c0", "R0")
    ccolor("BR", "R0", "C0")

    inits.append(cu("BIG", 200))
    corners = [("TL", "rI", "cI"), ("TR", "rI", "C"), ("BL", "R", "cI"), ("BR", "R", "C")]
    keys, winb = [], []
    for nm, rr, cc in corners:
        nodes.append(helper.make_node("Greater", [nm + "_col", "zero"], [nm + "_on"]))
        nodes.append(helper.make_node("Add", [rr, cc], [nm + "_mn0"]))
        nodes.append(helper.make_node("Where", [nm + "_on", nm + "_mn0", "BIG"], [nm + "_key"]))
        keys.append(nm + "_key")
    nodes.append(helper.make_node("Min", keys, ["minkey"]))
    wu = []
    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("Equal", [keys[i], "minkey"], [nm + "_winb"]))
        winb.append(nm + "_winb")
        nodes.append(helper.make_node("Cast", [nm + "_winb"], [nm + "_winu"], to=U8))
        wu.append(nm + "_winu")
    # nwin -> uniq
    nodes.append(helper.make_node("Add", [wu[0], wu[1]], ["nw01"]))
    nodes.append(helper.make_node("Add", [wu[2], wu[3]], ["nw23"]))
    nodes.append(helper.make_node("Add", ["nw01", "nw23"], ["nwin"]))
    nodes.append(helper.make_node("Equal", ["nwin", "one"], ["uniq"]))

    # top_won = TL|TR ; left_won = TL|BL
    nodes.append(helper.make_node("Or", [winb[0], winb[1]], ["topw"]))
    nodes.append(helper.make_node("Or", [winb[0], winb[2]], ["leftw"]))
    nodes.append(helper.make_node("Where", ["topw", "rI", "R"], ["orr"]))   # u8
    nodes.append(helper.make_node("Where", ["leftw", "cI", "C"], ["occ"]))  # u8
    # owner color: where(leftw, where(topw,TL,BL), where(topw,TR,BR))
    nodes.append(helper.make_node("Where", ["topw", "TL_col", "BL_col"], ["colL"]))  # u8 scalar
    nodes.append(helper.make_node("Where", ["topw", "TR_col", "BR_col"], ["colR"]))
    nodes.append(helper.make_node("Where", ["leftw", "colL", "colR"], ["ocol"]))     # u8 scalar

    inits.append(cu("two", 2))
    nodes.append(helper.make_node("Mod", ["orr", "two"], ["orm"]))
    nodes.append(helper.make_node("Mod", ["occ", "two"], ["ocm"]))
    nodes.append(helper.make_node("Equal", ["orm", "zero"], ["evr"]))
    nodes.append(helper.make_node("Equal", ["ocm", "zero"], ["evc"]))
    nodes.append(helper.make_node("Greater", ["occ", "orr"], ["gt"]))
    nodes.append(helper.make_node("Not", ["gt"], ["ngt"]))
    nodes.append(helper.make_node("Or", ["evr", "gt"], ["forgt"]))
    nodes.append(helper.make_node("And", ["evc", "forgt"], ["a1"]))
    nodes.append(helper.make_node("Not", ["evc"], ["nE"]))
    nodes.append(helper.make_node("And", ["evr", "ngt"], ["a2t"]))
    nodes.append(helper.make_node("And", ["nE", "a2t"], ["a2"]))
    nodes.append(helper.make_node("Or", ["a1", "a2"], ["pat"]))

    nodes.append(helper.make_node("And", ["uniq", "pat"], ["fill"]))
    nodes.append(helper.make_node("Cast", ["fill"], ["fillu"], to=U8))
    nodes.append(helper.make_node("Mul", ["fillu", "ocol"], ["gmin"]))
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
