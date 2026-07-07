"""Compact ONNX for task328 — v5. Manhattan Voronoi + owner-collapse.
Corner colors via dynamic Slice (no [1,1,30,30] product grids).
CumSum in float16. All big interior tensors uint8/bool/f16.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL
N = 30


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=U8))         # u8 color grid
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["msum"]))
    inits.append(cf("zf", 0.0))
    nodes.append(helper.make_node("Greater", ["msum", "zf"], ["valid"]))  # bool
    nodes.append(helper.make_node("Cast", ["valid"], ["validh"], to=F16))

    rI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, N, 1), (1, 1, N, N)).copy()
    cI = np.broadcast_to(np.arange(N, dtype=np.uint8).reshape(1, 1, 1, N), (1, 1, N, N)).copy()
    inits.append(cu("rI", rI))
    inits.append(cu("cI", cI))

    # R, C via reverse cumsum (f16)
    inits.append(ci("axW", 3))
    inits.append(ci("axH", 2))
    nodes.append(helper.make_node("CumSum", ["validh", "axW"], ["rinclh"], reverse=1))
    nodes.append(helper.make_node("CumSum", ["validh", "axH"], ["binclh"], reverse=1))
    nodes.append(helper.make_node("Cast", ["rinclh"], ["rinclu"], to=U8))
    nodes.append(helper.make_node("Cast", ["binclh"], ["binclu"], to=U8))
    inits.append(cu("one", 1))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Sub", ["rinclu", "one"], ["C"]))    # u8 cc_right
    nodes.append(helper.make_node("Sub", ["binclu", "one"], ["R"]))    # u8 rr_bottom

    # N-1 scalar from R[0,0]  (= N-1)
    inits.append(ci("s00", [0, 0]))
    inits.append(ci("e11", [1, 1]))
    inits.append(ci("ax23", [2, 3]))
    nodes.append(helper.make_node("Slice", ["R", "s00", "e11", "ax23"], ["nm1g"]))  # u8 [1,1,1,1]
    nodes.append(helper.make_node("Cast", ["nm1g"], ["nm1i"], to=I64))               # i64 [1,1,1,1]
    inits.append(ci("shp1", [1]))
    nodes.append(helper.make_node("Reshape", ["nm1i", "shp1"], ["nm1"]))             # i64 [1]
    inits.append(ci("z1", [0]))
    inits.append(ci("o1", [1]))
    inits.append(ci("ax2", [2]))
    inits.append(ci("ax3", [3]))
    nodes.append(helper.make_node("Add", ["nm1", "o1"], ["ne"]))                     # N (end)

    # corner colors via Slice on g (no big grids)
    # TL = g[...,0:1,0:1]
    nodes.append(helper.make_node("Slice", ["g", "s00", "e11", "ax23"], ["TL_col"]))
    # row0 = g[...,0:1,:] ; rowN = g[...,N-1:N,:]
    nodes.append(helper.make_node("Slice", ["g", "z1", "o1", "ax2"], ["row0"]))      # [1,1,1,30]
    nodes.append(helper.make_node("Slice", ["g", "nm1", "ne", "ax2"], ["rowN"]))     # [1,1,1,30]
    # TR = row0[..., N-1:N] ; BL = rowN[...,0:1] ; BR = rowN[...,N-1:N]
    nodes.append(helper.make_node("Slice", ["row0", "nm1", "ne", "ax3"], ["TR_col"]))
    nodes.append(helper.make_node("Slice", ["rowN", "z1", "o1", "ax3"], ["BL_col"]))
    nodes.append(helper.make_node("Slice", ["rowN", "nm1", "ne", "ax3"], ["BR_col"]))

    inits.append(cu("BIG", 200))
    corners = [("TL", "rI", "cI"), ("TR", "rI", "C"), ("BL", "R", "cI"), ("BR", "R", "C")]
    keys, winflags = [], []
    for nm, rr, cc in corners:
        nodes.append(helper.make_node("Greater", [nm + "_col", "zero"], [nm + "_on"]))
        nodes.append(helper.make_node("Add", [rr, cc], [nm + "_mn0"]))
        nodes.append(helper.make_node("Where", [nm + "_on", nm + "_mn0", "BIG"], [nm + "_key"]))
        keys.append(nm + "_key")
    nodes.append(helper.make_node("Min", keys, ["minkey"]))
    for i, (nm, *_ ) in enumerate(corners):
        nodes.append(helper.make_node("Equal", [keys[i], "minkey"], [nm + "_winb"]))
        nodes.append(helper.make_node("Cast", [nm + "_winb"], [nm + "_win"], to=U8))
        winflags.append(nm + "_win")
    nodes.append(helper.make_node("Add", [winflags[0], winflags[1]], ["nw01"]))
    nodes.append(helper.make_node("Add", [winflags[2], winflags[3]], ["nw23"]))
    nodes.append(helper.make_node("Add", ["nw01", "nw23"], ["nwin"]))
    nodes.append(helper.make_node("Equal", ["nwin", "one"], ["uniq"]))

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
