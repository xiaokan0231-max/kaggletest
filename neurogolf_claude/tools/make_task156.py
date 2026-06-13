"""task156: grid has exactly two solid axis-aligned rectangles of color 4 (never
sharing a row, never equal area, never touching). Output keeps each rectangle's
border as 4 but recolors its strict interior (cells whose four orthogonal
neighbors are also =4): the LARGER-area rectangle's interior -> 2, smaller -> 1.

Label-free, genuine algorithm (per-ROW discriminator; the two rectangles never
share a row):
  M        = channel-4 mask [1,1,30,30]
  interior = M AND shift(M) in all 4 orthogonal directions
  rw[r]    = sum_W M                          # width of rect on row r (0 if empty)
  rmask    = rw > 0
  h[r]     = vertical run of rmask through r   # block height (1D over 30 rows)
  area[r]  = rw[r]*h[r]                         # area of the rect occupying row r
  gmax     = max area ; big rows = (area==gmax) AND rmask
  interior cell -> 2 if its row is "big", else 1.

Memory budget: the big [1,1,30,30] tensors (M, 4 shifts, interior, border,
big/small cell, value grid) are INT8. All run-length work is done on length-30
row vectors [1,1,30,1] in FLOAT (tiny). Output is a single [1,10,30,30] bool.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/wf2_task156.onnx"
H = W = 30
I8 = TensorProto.INT8
F32 = TensorProto.FLOAT
nodes, inits = [], []


def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name))
    return name


def n(op, i, o, **kw):
    if isinstance(o, str):
        o = [o]
    if isinstance(i, str):
        i = [i]
    nodes.append(helper.make_node(op, i, o, **kw))
    return o[0]


# ---- constants ----
C("ax3", np.array([3], np.int64))
C("ax2", np.array([2], np.int64))
w4 = np.zeros((1, 10, 1, 1), np.float32)
w4[0, 4, 0, 0] = 1.0
C("w4", w4)
C("zero_i8", np.array(0, np.int8))
C("one_i8", np.array(1, np.int8))
C("c4_i8", np.array(4, np.int8))
C("c2_i8", np.array(2, np.int8))
C("colors", np.arange(10, dtype=np.int8).reshape(1, 10, 1, 1))
C("zero_f", np.array(0.0, np.float32))
C("one_f", np.array(1.0, np.float32))

# ---- Mf = channel-4 mask [1,1,30,30] (float, for the row reduction) ----
n("Conv", ["input", "w4"], "Mf")            # float 0/1
n("Cast", ["Mf"], "M", to=I8)               # int8 0/1 for spatial AND ops


def shift_i8(src, dy, dx, name):
    pt, pb = max(dy, 0), max(-dy, 0)
    pl, pr = max(dx, 0), max(-dx, 0)
    C(name + "_pads", np.array([0, 0, pt, pl, 0, 0, pb, pr], np.int64))
    n("Pad", [src, name + "_pads", "zero_i8"], name + "_pad")
    C(name + "_st", np.array([pb, pr], np.int64))
    C(name + "_en", np.array([pb + H, pr + W], np.int64))
    C(name + "_ax", np.array([2, 3], np.int64))
    n("Slice", [name + "_pad", name + "_st", name + "_en", name + "_ax"], name)
    return name


# ---- interior = min(M, up, down, left, right) == AND for 0/1 int8 ----
shift_i8("M", 1, 0, "Mu")
shift_i8("M", -1, 0, "Md")
shift_i8("M", 0, 1, "Ml")
shift_i8("M", 0, -1, "Mr")
n("Min", ["M", "Mu", "Md", "Ml", "Mr"], "interior")   # [1,1,30,30] int8 0/1

# =====================  per-row work (FLOAT, length-30 vectors)  =============
# rw[r] = width of rect on row r
n("ReduceSum", ["Mf", "ax3"], "rw", keepdims=1)        # [1,1,30,1] float
n("Greater", ["rw", "zero_f"], "rmask_b")
n("Cast", ["rmask_b"], "rmask", to=F32)                # [1,1,30,1] float 0/1


def cummax_rows(src, name):
    cur = src
    for k in range(5):                      # 2^5=32 covers 30
        s = 1 << k
        C(f"{name}_p{k}_pads", np.array([0, 0, s, 0, 0, 0, 0, 0], np.int64))
        C(f"{name}_p{k}_pv", np.array(-1e9, np.float32))
        n("Pad", [cur, f"{name}_p{k}_pads", f"{name}_p{k}_pv"], f"{name}_p{k}_pad")
        C(f"{name}_p{k}_st", np.array([0], np.int64))
        C(f"{name}_p{k}_en", np.array([H], np.int64))
        C(f"{name}_p{k}_ax", np.array([2], np.int64))
        n("Slice", [f"{name}_p{k}_pad", f"{name}_p{k}_st", f"{name}_p{k}_en",
                    f"{name}_p{k}_ax"], f"{name}_p{k}_sh")
        n("Max", [cur, f"{name}_p{k}_sh"], f"{name}_p{k}_m")
        cur = f"{name}_p{k}_m"
    return cur


def reverse_rows(src, name):
    C(name + "_st", np.array([H - 1], np.int64))
    C(name + "_en", np.array([-(H + 1)], np.int64))
    C(name + "_ax", np.array([2], np.int64))
    C(name + "_sp", np.array([-1], np.int64))
    n("Slice", [src, name + "_st", name + "_en", name + "_ax", name + "_sp"], name)
    return name


# up = (cumsum(rmask) - cummax_above(cumsum*(1-rmask))) * rmask
n("CumSum", ["rmask", "ax2"], "cs")
n("Sub", ["one_f", "rmask"], "notrm")
n("Mul", ["cs", "notrm"], "csz")
baseU = cummax_rows("csz", "bU")
n("Sub", ["cs", baseU], "upraw")
n("Mul", ["upraw", "rmask"], "up")
# down = reverse of up-logic on reversed rmask
reverse_rows("rmask", "rmrev")
n("CumSum", ["rmrev", "ax2"], "csr")
n("Sub", ["one_f", "rmrev"], "notrmr")
n("Mul", ["csr", "notrmr"], "csrz")
baseD = cummax_rows("csrz", "bD")
n("Sub", ["csr", baseD], "downraw_r")
n("Mul", ["downraw_r", "rmrev"], "down_r")
reverse_rows("down_r", "down")
n("Add", ["up", "down"], "ud")
n("Sub", ["ud", "rmask"], "h")                         # [1,1,30,1] block height

# area[r] = rw*h ; gmax ; big rows = (area==gmax)&rmask
n("Mul", ["rw", "h"], "area")
n("ReduceMax", ["area", "ax2"], "gmax", keepdims=1)    # [1,1,1,1]
n("Equal", ["area", "gmax"], "eqg_b")
n("Cast", ["eqg_b"], "eqg", to=F32)
n("Mul", ["eqg", "rmask"], "bigrow_f")                 # [1,1,30,1] float 0/1
n("Cast", ["bigrow_f"], "bigrow", to=I8)               # [1,1,30,1] int8 0/1

# =====================  assemble output (INT8 spatial)  =====================
# border cells keep 4; interior recolored (2 if big row else 1).
n("Sub", ["M", "interior"], "border")                  # int8 0/1
n("Mul", ["interior", "bigrow"], "bigcell")            # int8, bigrow broadcast over W
n("Sub", ["interior", "bigcell"], "smallcell")
# value grid g = 4*border + 2*bigcell + 1*smallcell
n("Mul", ["border", "c4_i8"], "v4")
n("Mul", ["bigcell", "c2_i8"], "v2")
n("Add", ["v4", "v2"], "v42")
n("Add", ["v42", "smallcell"], "g")                    # [1,1,30,30] int8 color grid

# output one-hot = (g == colors) cast to float -> single big bool intermediate
n("Equal", ["g", "colors"], "oh_b")                    # [1,10,30,30] bool
n("Cast", ["oh_b"], "output", to=F32)

graph = helper.make_graph(
    nodes, "task156",
    [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 10
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
nparams = sum(int(np.prod(t.dims)) if t.dims else 1 for t in inits)
print(f"saved {OUT} nodes={len(nodes)} params={nparams} bytes={len(model.SerializeToString())}")
