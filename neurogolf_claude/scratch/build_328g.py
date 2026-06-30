"""Compact ONNX for task328 — v9. Manhattan Voronoi + owner-collapse.

Corner colors via STATIC reduces on the sparse color grid g (g is nonzero only at
the <=4 corners): TL=g[0,0]; (TL+TR)=rowsum[0]; (TL+BL)=colsum[0]; total=sum;
BR=total-TL-TR-BL. No [1,1,30,30] color grids. CumSum(f16) for R,C. Static shapes.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
BOOL = TensorProto.BOOL
N = 30


def cf(n, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=n)


def cu(n, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=n)


def ci(n, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=n)


def ci32(n, v):
    return numpy_helper.from_array(np.asarray(v, np.int32), name=n)


def build(path):
    nodes, inits = [], []

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g16"], to=F16))   # f16 color grid (sparse) for reduces
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["msum"]))
    inits.append(cf("zf", 0.0))
    nodes.append(helper.make_node("Greater", ["msum", "zf"], ["valid"]))  # bool valid
    nodes.append(helper.make_node("Cast", ["valid"], ["validh"], to=F16))

    # vector index tensors (broadcast)
    inits.append(cu("rI", np.arange(N, dtype=np.uint8).reshape(1, 1, N, 1)))
    inits.append(cu("cI", np.arange(N, dtype=np.uint8).reshape(1, 1, 1, N)))

    # R, C via reverse cumsum (f16)
    inits.append(ci("axW", 3))
    inits.append(ci("axH", 2))
    nodes.append(helper.make_node("CumSum", ["validh", "axW"], ["rinclh"], reverse=1))
    nodes.append(helper.make_node("CumSum", ["validh", "axH"], ["binclh"], reverse=1))
    nodes.append(helper.make_node("Cast", ["rinclh"], ["rinclu"], to=U8))
    nodes.append(helper.make_node("Cast", ["binclh"], ["binclu"], to=U8))
    inits.append(cu("one", 1))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Sub", ["rinclu", "one"], ["C"]))   # u8 cc_right
    nodes.append(helper.make_node("Sub", ["binclu", "one"], ["R"]))   # u8 rr_bottom

    # --- corner colors via static reduces on sparse g (f16) ---
    inits.append(ci("ax2", [2]))
    inits.append(ci("ax3", [3]))
    inits.append(ci("ax23", [2, 3]))
    nodes.append(helper.make_node("ReduceSum", ["g16", "ax23"], ["total"], keepdims=1))   # [1,1,1,1] TL+TR+BL+BR
    nodes.append(helper.make_node("ReduceSum", ["g16", "ax2"], ["colsum"], keepdims=1))   # [1,1,1,30]
    nodes.append(helper.make_node("ReduceSum", ["g16", "ax3"], ["rowsum"], keepdims=1))   # [1,1,30,1]
    inits.append(ci("s0", [0]))
    inits.append(ci("e1", [1]))
    # TL = g[0,0]: slice g16 rows0:1 cols0:1
    inits.append(ci("s00", [0, 0]))
    inits.append(ci("e11", [1, 1]))
    nodes.append(helper.make_node("Slice", ["g16", "s00", "e11", "ax23"], ["TLi"]))       # [1,1,1,1]
    nodes.append(helper.make_node("Slice", ["rowsum", "s0", "e1", "ax2"], ["TLpTR"]))    # row0 of rowsum
    nodes.append(helper.make_node("Slice", ["colsum", "s0", "e1", "ax3"], ["TLpBL"]))    # col0 of colsum
    nodes.append(helper.make_node("Sub", ["TLpTR", "TLi"], ["TRi"]))
    nodes.append(helper.make_node("Sub", ["TLpBL", "TLi"], ["BLi"]))
    nodes.append(helper.make_node("Sub", ["total", "TLi"], ["tmTL"]))
    nodes.append(helper.make_node("Sub", ["tmTL", "TRi"], ["tmTLTR"]))
    nodes.append(helper.make_node("Sub", ["tmTLTR", "BLi"], ["BRi"]))
    for nm in ["TL", "TR", "BL", "BR"]:
        nodes.append(helper.make_node("Cast", [nm + "i"], [nm + "_col"], to=U8))   # u8 scalar

    inits.append(cu("BIG", 200))
    corners = [("TL", "rI", "cI"), ("TR", "rI", "C"), ("BL", "R", "cI"), ("BR", "R", "C")]
    keys, winb, wu = [], [], []
    for nm, rr, cc in corners:
        nodes.append(helper.make_node("Greater", [nm + "_col", "zero"], [nm + "_on"]))
        nodes.append(helper.make_node("Add", [rr, cc], [nm + "_mn0"]))
        nodes.append(helper.make_node("Where", [nm + "_on", nm + "_mn0", "BIG"], [nm + "_key"]))
        keys.append(nm + "_key")
    nodes.append(helper.make_node("Min", keys, ["minkey"]))
    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("Equal", [keys[i], "minkey"], [nm + "_winb"]))
        winb.append(nm + "_winb")
        nodes.append(helper.make_node("Cast", [nm + "_winb"], [nm + "_winu"], to=U8))
        wu.append(nm + "_winu")
    nodes.append(helper.make_node("Add", [wu[0], wu[1]], ["nw01"]))
    nodes.append(helper.make_node("Add", [wu[2], wu[3]], ["nw23"]))
    nodes.append(helper.make_node("Add", ["nw01", "nw23"], ["nwin"]))
    nodes.append(helper.make_node("Equal", ["nwin", "one"], ["uniq"]))

    nodes.append(helper.make_node("Or", [winb[0], winb[1]], ["topw"]))
    nodes.append(helper.make_node("Or", [winb[0], winb[2]], ["leftw"]))
    nodes.append(helper.make_node("Where", ["topw", "rI", "R"], ["orr"]))
    nodes.append(helper.make_node("Where", ["leftw", "cI", "C"], ["occ"]))
    nodes.append(helper.make_node("Where", ["topw", "TL_col", "BL_col"], ["colL"]))
    nodes.append(helper.make_node("Where", ["topw", "TR_col", "BR_col"], ["colR"]))
    nodes.append(helper.make_node("Where", ["leftw", "colL", "colR"], ["ocol"]))

    inits.append(cu("two", 2))
    nodes.append(helper.make_node("Mod", ["orr", "two"], ["orm"]))
    nodes.append(helper.make_node("Mod", ["occ", "two"], ["ocm"]))
    nodes.append(helper.make_node("Equal", ["orm", "zero"], ["evr"]))
    nodes.append(helper.make_node("Equal", ["ocm", "zero"], ["evc"]))
    # pat = (E and gt) or (F and not gt)   [E=evc, F=evr, gt=occ>orr]
    nodes.append(helper.make_node("Greater", ["occ", "orr"], ["gt"]))
    nodes.append(helper.make_node("Not", ["gt"], ["ngt"]))
    nodes.append(helper.make_node("And", ["evc", "gt"], ["a1"]))
    nodes.append(helper.make_node("And", ["evr", "ngt"], ["a2"]))
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
