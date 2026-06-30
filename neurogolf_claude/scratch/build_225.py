"""Build compact ONNX for task225 (leaner v2).

Rule: single 2x2 colored block on a 6x6 grid with corner colors TL,TR,BL,BR.
Place 4 solid 2x2 halo blocks diagonally outward, each colored by the
diagonally-OPPOSITE corner (point reflection through block center, exploded by
2 cells), clipped to grid. Verified on all 262 cases.

Implementation: crop input to 6x6, collapse one-hot->color grid (Conv), detect
the 4 corners via 3-term neighbor masks (block is an isolated 2x2 so NOT-terms
are unnecessary), pull each opposite color to its halo's near corner, dilate to
2x2 with a single MaxPool each (directional via asymmetric pads), max-combine,
then Equal->one-hot 6x6 and Pad to 30x30 as the (excluded) graph output.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

N = 6
inits, nodes = [], []


def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name))
    return name


def node(op, ins, outs, **kw):
    nodes.append(helper.make_node(op, ins, outs, **kw))
    return outs[0] if isinstance(outs, list) else outs


# ---- crop one-hot input to top-left 6x6, collapse channels -> color grid uint8 ----
C("st0", np.array([0, 0], dtype=np.int64))
C("en6", np.array([N, N], dtype=np.int64))
C("ax23", np.array([2, 3], dtype=np.int64))
node("Slice", ["input", "st0", "en6", "ax23"], ["oh6"])
C("Wc", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
node("Conv", ["oh6", "Wc"], ["g6f"], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
node("Cast", ["g6f"], ["g"], to=TensorProto.UINT8)


# single-op shift via Pad with negative pads on [1,1,N,N]; positive dr moves content DOWN.
_shift_tag = [0]
C("z0u8", np.array(0, dtype=np.uint8))


def shift(src, dr, dc):
    t = _shift_tag[0]; _shift_tag[0] += 1
    # pad order [b,c,h_begin,w_begin, b,c,h_end,w_end]; net size change 0.
    # positive dr -> pad top +dr, crop bottom -dr -> content moves DOWN.
    C(f"spd{t}", np.array([0, 0, dr, dc, 0, 0, -dr, -dc], dtype=np.int64))
    node("Pad", [src, f"spd{t}", "z0u8"], [f"shf{t}"], mode="constant")
    return f"shf{t}"


# ---- nonzero mask (uint8 0/1) via Min(g,1), one op ----
C("one", np.array(1, dtype=np.uint8))
node("Min", ["g", "one"], ["nz"])
up = shift("nz", +1, 0)    # out[r,c]=nz[r-1,c]
dn = shift("nz", -1, 0)    # out[r,c]=nz[r+1,c]
lf = shift("nz", 0, +1)    # out[r,c]=nz[r,c-1]
rt = shift("nz", 0, -1)    # out[r,c]=nz[r,c+1]


def mul(a, b, out):
    return node("Mul", [a, b], [out])


# isolated corner colors directly: g*neighbor*neighbor (g*nz == g, so nz factor is implicit).
# share g*right (-> TL,BL) and g*left (-> TR,BR).
mul("g", rt, "gr")   # color where a right neighbor exists
mul("g", lf, "gl")   # color where a left  neighbor exists
mul("gr", dn, "TLc")  # TL: right & down
mul("gr", up, "BLc")  # BL: right & up
mul("gl", dn, "TRc")  # TR: left  & down
mul("gl", up, "BRc")  # BR: left  & up


# place halo: DILATE corner color 2x2 in place (one MaxPool), THEN shift to its halo region.
# This order keeps MaxPool's input off a Pad node, so it is robust under graph optimization.
# MaxPool kernel 2x2 stride1, pads=[pt,pl,pb,pr]. pad bottom+right -> dilate DOWN-RIGHT;
# pad top+left -> dilate UP-LEFT.
_h = [0]


def halo(colorgrid, pads, sdr, sdc):
    h = _h[0]; _h[0] += 1
    node("MaxPool", [colorgrid], [f"dil{h}"], kernel_shape=[2, 2], strides=[1, 1], pads=pads)
    return shift(f"dil{h}", sdr, sdc)


h_tl = halo("BRc", [0, 0, 1, 1], -2, -2)  # dilate up-left,    shift to TL region
h_br = halo("TLc", [1, 1, 0, 0], +2, +2)  # dilate down-right, shift to BR region
h_bl = halo("TRc", [1, 0, 0, 1], +2, -2)  # dilate down-left,  shift to BL region
h_tr = halo("BLc", [0, 1, 1, 0], -2, +2)  # dilate up-right,   shift to TR region

node("Max", ["g", h_tl, h_br, h_bl, h_tr], ["gm6"])

# Equal -> one-hot bool [1,10,6,6] (interior), Pad to 30x30 (false) as graph output.
C("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
node("Equal", ["gm6", "colors"], ["oh6out"])
C("pad30", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], dtype=np.int64))
C("falseb", np.array(False, dtype=bool))
node("Pad", ["oh6out", "pad30", "falseb"], ["output"], mode="constant")

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "task225", [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b15_task225.onnx")
print("saved nodes=", len(nodes), "inits=", len(inits))
