"""Build compact ONNX for task208 (separable + run-length selection).

Rule (verified 266/266): input has a hollow box (border = least-frequent nonzero
color `bc`, interior all-0).  Output adds a 2nd copy of that box, same interior
size, around the unique OTHER all-zero rectangle (dest hole); interior stays 0.

Static pipeline (no dynamic shape / Loop).  Interior-size detection uses a
run-length sum so a SINGLE runtime comparison against ih_t / iw_t replaces a
4-way size branch:
  vrun_ge[k] = (Conv ones(k,1)(Z) >= k)            k=2..5  vertical k-zero runs
  Sv = sum_k vrun_ge[k]                            Sv = clamp(run-1,0,4)
  Vsel = (Sv >= ih_t-1)                            vertical run >= ih_t  (1 compare)
  Sh = sum_k hrun_ge[k](Vsel) ; Csel = (Sh >= iw_t-1)        all-zero-window corners
  cover = hdil_iw( vdil_ih( Csel ) )               4-branch selected dilation
  dest_int = cover AND Z AND ~templateInterior
  ring = dilate3(dest_int) AND ~dest_int
  gm = where(ring, bc, g) ; pad 30x30 sentinel 255 ; output = Equal(gm, 0..9)
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 21
KS = range(2, 6)


def cf(n, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), n)


def c16(n, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), n)


def ci(n, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), n)


def cu(n, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), n)


def build(path):
    nodes, inits = [], []

    def N(op, i, o, **kw):
        nodes.append(helper.make_node(op, i, o, **kw))

    # ---- color grid g uint8 ----
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    N("Conv", ["input", "Wcol"], ["g30"])                       # f32 [1,1,30,30]
    N("Cast", ["g30"], ["g30u"], to=U8)
    inits += [ci("s0", [0, 0]), ci("s21", [S, S]), ci("ax23", [2, 3])]
    N("Slice", ["g30u", "s0", "s21", "ax23"], ["g"])            # u8 [1,1,21,21]

    # ---- bc = least-frequent nonzero color ----
    inits.append(ci("ax23i", [2, 3]))
    N("ReduceSum", ["input", "ax23i"], ["chcnt"], keepdims=0)   # [1,10]
    inits.append(cf("ch0big", np.array([[1e9] + [0] * 9], np.float32)))
    N("Add", ["chcnt", "ch0big"], ["cnt1"])
    inits.append(cf("zero", 0.0))
    N("Equal", ["cnt1", "zero"], ["isabs"])
    inits.append(cf("big", 1e9))
    N("Where", ["isabs", "big", "cnt1"], ["cntf"])
    N("ArgMin", ["cntf"], ["bcidx"], axis=1, keepdims=1)
    N("Cast", ["bcidx"], ["bcu"], to=U8)
    inits.append(ci("sh1111", [1, 1, 1, 1]))
    N("Reshape", ["bcu", "sh1111"], ["bc"])                     # [1,1,1,1] u8

    # ---- B, bbox -> ih_t, iw_t, insideB ----
    N("Equal", ["g", "bc"], ["Bb"])
    N("Cast", ["Bb"], ["Bf"], to=F32)
    N("ReduceMax", ["Bf"], ["rowhas"], axes=[3], keepdims=0)
    N("ReduceMax", ["Bf"], ["colhas"], axes=[2], keepdims=0)
    inits.append(ci("ax2i", [2]))
    N("ReduceSum", ["rowhas", "ax2i"], ["Hb"], keepdims=1)
    N("ReduceSum", ["colhas", "ax2i"], ["Wb"], keepdims=1)
    inits.append(cf("two", 2.0))
    inits.append(ci("sh111", [1, 1, 1, 1]))
    N("Sub", ["Hb", "two"], ["ih_t0"])
    N("Sub", ["Wb", "two"], ["iw_t0"])
    N("Reshape", ["ih_t0", "sh111"], ["ih_t"])                 # [1,1,1,1] f32
    N("Reshape", ["iw_t0", "sh111"], ["iw_t"])
    # thresholds ih_t-1, iw_t-1 for run-length compare
    inits.append(cf("one", 1.0))
    N("Sub", ["ih_t", "one"], ["ihm1"])                        # [1,1,1,1]
    N("Sub", ["iw_t", "one"], ["iwm1"])

    inits.append(ci("axc", 2))
    inits.append(cf("half", 0.5))
    N("CumSum", ["rowhas", "axc"], ["rct"])
    N("CumSum", ["rowhas", "axc"], ["rcb"], reverse=1)
    N("Greater", ["rct", "half"], ["rt"])
    N("Greater", ["rcb", "half"], ["rbm"])
    N("And", ["rt", "rbm"], ["rspan"])
    N("Less", ["rowhas", "half"], ["rnob"])
    N("And", ["rspan", "rnob"], ["rin"])
    N("CumSum", ["colhas", "axc"], ["cct"])
    N("CumSum", ["colhas", "axc"], ["ccb"], reverse=1)
    N("Greater", ["cct", "half"], ["cct2"])
    N("Greater", ["ccb", "half"], ["ccb2"])
    N("And", ["cct2", "ccb2"], ["cspan"])
    N("Less", ["colhas", "half"], ["cnob"])
    N("And", ["cspan", "cnob"], ["cin"])
    inits += [ci("shR", [1, 1, S, 1]), ci("shC", [1, 1, 1, S])]
    N("Reshape", ["rin", "shR"], ["rinR"])
    N("Reshape", ["cin", "shC"], ["cinC"])
    N("And", ["rinR", "cinC"], ["insideB"])

    # ---- Z ----
    inits.append(cu("zerou", 0))
    N("Equal", ["g", "zerou"], ["Zb"])
    N("Cast", ["Zb"], ["Z16"], to=F16)
    N("And", ["Zb", "insideB"], ["tmplIntb"])
    N("Not", ["tmplIntb"], ["nottmpl"])

    inits.append(c16("half16", 0.5))

    # ---- vertical run-length sum Sv -> Vsel = (Sv >= ih_t-1) ----
    accSv = None
    for k in KS:
        inits.append(c16(f"Kv{k}", np.ones((1, 1, k, 1), np.float16)))
        N("Conv", ["Z16", f"Kv{k}"], [f"vc{k}"], pads=[0, 0, k - 1, 0])  # f16 [1,1,21,21]
        inits.append(c16(f"vn{k}", float(k) - 0.5))
        N("Greater", [f"vc{k}", f"vn{k}"], [f"vge{k}b"])       # bool run>=k
        N("Cast", [f"vge{k}b"], [f"vge{k}"], to=F16)
        if accSv is None:
            accSv = f"vge{k}"
        else:
            N("Add", [accSv, f"vge{k}"], [f"Sv{k}"])
            accSv = f"Sv{k}"
    # Vsel = Sv >= ih_t-1  (ih_t-1 in f16)
    N("Cast", ["ihm1"], ["ihm1_16"], to=F16)
    N("GreaterOrEqual", [accSv, "ihm1_16"], ["Vselb"])         # bool [1,1,21,21]
    N("Cast", ["Vselb"], ["Vsel16"], to=F16)

    # ---- horizontal run-length sum Sh -> Csel = (Sh >= iw_t-1) ----
    accSh = None
    for k in KS:
        inits.append(c16(f"Kh{k}", np.ones((1, 1, 1, k), np.float16)))
        N("Conv", ["Vsel16", f"Kh{k}"], [f"hc{k}"], pads=[0, 0, 0, k - 1])
        inits.append(c16(f"hn{k}", float(k) - 0.5))
        N("Greater", [f"hc{k}", f"hn{k}"], [f"hge{k}b"])
        N("Cast", [f"hge{k}b"], [f"hge{k}"], to=F16)
        if accSh is None:
            accSh = f"hge{k}"
        else:
            N("Add", [accSh, f"hge{k}"], [f"Sh{k}"])
            accSh = f"Sh{k}"
    N("Cast", ["iwm1"], ["iwm1_16"], to=F16)
    N("GreaterOrEqual", [accSh, "iwm1_16"], ["Cselb"])         # corners bool
    N("Cast", ["Cselb"], ["Csel16"], to=F16)

    # size-selectors for dilation (one-hot bool)
    for k in KS:
        inits.append(cf(f"kc{k}", float(k)))
        N("Equal", ["ih_t", f"kc{k}"], [f"selh{k}"])
        N("Equal", ["iw_t", f"kc{k}"], [f"selw{k}"])

    # ---- dilate corners: vertical (ih_t) then horizontal (iw_t) ----
    accVd = None
    for ih in KS:
        N("MaxPool", ["Csel16"], [f"vd{ih}"], kernel_shape=[ih, 1],
          strides=[1, 1], pads=[ih - 1, 0, 0, 0])
        N("Greater", [f"vd{ih}", "half16"], [f"vdb{ih}"])
        N("And", [f"vdb{ih}", f"selh{ih}"], [f"vdg{ih}"])
        if accVd is None:
            accVd = f"vdg{ih}"
        else:
            N("Or", [accVd, f"vdg{ih}"], [f"Vda{ih}"])
            accVd = f"Vda{ih}"
    N("Cast", [accVd], ["Vd16"], to=F16)

    accCov = None
    for iw in KS:
        N("MaxPool", ["Vd16"], [f"hd{iw}"], kernel_shape=[1, iw],
          strides=[1, 1], pads=[0, iw - 1, 0, 0])
        N("Greater", [f"hd{iw}", "half16"], [f"hdb{iw}"])
        N("And", [f"hdb{iw}", f"selw{iw}"], [f"hdg{iw}"])
        if accCov is None:
            accCov = f"hdg{iw}"
        else:
            N("Or", [accCov, f"hdg{iw}"], [f"Cova{iw}"])
            accCov = f"Cova{iw}"

    # dest_int = cover & Z & ~tmpl
    N("And", [accCov, "Zb"], ["di1"])
    N("And", ["di1", "nottmpl"], ["dest_intb"])
    N("Cast", ["dest_intb"], ["dest16"], to=F16)

    # ring = dilate3(dest) & ~dest
    N("MaxPool", ["dest16"], ["dil3"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    N("Greater", ["dil3", "half16"], ["dil3b"])
    N("Not", ["dest_intb"], ["ndest"])
    N("And", ["dil3b", "ndest"], ["ringb"])

    # gm = where(ring, bc, g)  (uint8)
    N("Where", ["ringb", "bc", "g"], ["gmu"])
    inits.append(ci("padg", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(cu("sent", 255))
    N("Pad", ["gmu", "padg", "sent"], ["gm30"])
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    N("Equal", ["gm30", "colors"], ["output"])

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t208", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes), "inits", len(inits))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1a_task208.onnx")
