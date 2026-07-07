"""Compact ONNX for task288 (verified rule 267/267).

Rule: figure is a symmetric 2-row pyramid at the bottom of the grid. C = bottom-centre
colour. Draw two diagonal rays of colour C from the apex-band endpoints up-left and
up-right, clipped to the figure's columns. Input figure is always exactly 2 rows tall.

All grids <= 9x9 (arc-gen max). Crop to 9x9; all interior masks are bool (81B each);
the only large interior tensor is gf [1,1,30,30] f32 (the collapse output).
Logical ops on bool (And/Or/Not); colour composition via Where; output via Equal.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
K = 9

nodes, inits = [], []


def cf(name, v): return numpy_helper.from_array(np.asarray(v, np.float32), name)
def cu(name, v): return numpy_helper.from_array(np.asarray(v, np.uint8), name)
def ci(name, v): return numpy_helper.from_array(np.asarray(v, np.int64), name)
def cb(name, v): return numpy_helper.from_array(np.asarray(v, np.bool_), name)


def add(op, ins, outs, **kw):
    nodes.append(helper.make_node(op, ins, outs, **kw))


_sc = [0]


def shift(src, dr, dc, dtype="bool"):
    """result[r,c] = src[r-dr, c-dc] within KxK, zero/false fill, in ONE Pad op.
    Pad top=dr, bottom=-dr, left=dc, right=-dc (negative pads crop the vacated side)."""
    nm = f"s{_sc[0]}"; _sc[0] += 1
    inits.append(ci(nm + "_pp", [0, 0, dr, dc, 0, 0, -dr, -dc]))
    inits.append(cb(nm + "_pv", False) if dtype == "bool" else cu(nm + "_pv", 0))
    add("Pad", [src, nm + "_pp", nm + "_pv"], [nm + "_o"])
    return nm + "_o"


# crop the one-hot input to KxK and channels 1..9 (channel 0 = background contributes 0
# to the colour sum). x9 [1,9,9,9] is the only large interior tensor.
inits += [ci("cs", [1, 0, 0]), ci("ce", [10, K, K]), ci("ca", [1, 2, 3])]
add("Slice", ["input", "cs", "ce", "ca"], ["x9"])       # f32 [1,9,9,9]

# collapse -> colour grid g [1,1,9,9] u8 ; weight = colours 1..9
inits.append(cf("Wcol", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)))
add("Conv", ["x9", "Wcol"], ["g9f"])                    # f32 [1,1,9,9]
add("Cast", ["g9f"], ["g"], to=U8)                      # u8 colour grid

inits.append(cu("u0", 0))
add("Greater", ["g", "u0"], ["M"])                      # bool figure mask

# colHas [1,1,1,K] bool : reduce colours over rows (base full width -> exact column extent)
add("ReduceMax", ["g"], ["colHasU"], axes=[2], keepdims=1)  # u8 [1,1,1,K] (max colour/col)
add("Greater", ["colHasU", "u0"], ["colHas"])           # bool [1,1,1,K]

# grids are square and the figure spans the full width with its base on the bottom row,
# so the in-grid row extent equals the column extent: rowValid = transpose(colHas).
add("Transpose", ["colHas"], ["rowValid"], perm=[0, 1, 3, 2])  # bool [1,1,K,1]

# in-grid validity v = rowValid AND colHas (broadcasts to [1,1,K,K])
add("And", ["rowValid", "colHas"], ["v"])

# figure is exactly 2 rows. apex (upper) band = M & shift_up(M) (base sits below it).
mUp = shift("M", -1, 0)                                  # M(r+1,c) (the row below)
add("And", ["M", mUp], ["apex"])                        # cells with figure directly below

# diagonal seeds directly: the up-left neighbour of the leftmost apex cell is
#   Lseed = shift(apex,up-left) & ~shift(apex,up); symmetric for the right.
# (shared shift(apex,up) feeds both Nots.)
apUL = shift("apex", -1, -1)
apUR = shift("apex", -1, 1)
apU = shift("apex", -1, 0)
add("Not", [apU], ["napU"])
add("And", [apUL, "napU"], ["Lseed"])
add("And", [apUR, "napU"], ["Rseed"])
Lseed, Rseed = "Lseed", "Rseed"


def propagate(seed, dc_sign, tag):
    # max ray length per side is 3 -> shifts 1,2 cover offsets 0..3
    cur = seed
    for k in [1, 2]:
        s = shift(cur, -k, dc_sign * k)
        out = f"{tag}{k}"
        add("Or", [cur, s], [out])
        cur = out
    return cur


Lray = propagate(Lseed, -1, "L")
Rray = propagate(Rseed, 1, "R")
add("Or", [Lray, Rray], ["ray0"])
# rays never overlap the figure (seeds start above it), so only the column clip is needed.
add("And", ["ray0", "colHas"], ["ray"])                 # clip to figure columns

# inner colour C from C-band (base under apex)
cband = shift("apex", 1, 0)                              # bool
inits.append(cu("uz", 0))
add("Where", [cband, "g", "uz"], ["gcb"])               # u8: colour at C-band else 0
add("ReduceMax", ["gcb"], ["Cscalar"], axes=[2, 3], keepdims=1)  # u8 [1,1,1,1]

# gm = where(ray, C, g); then out-of-grid cells -> sentinel 255 (Equal yields all-false)
add("Where", ["ray", "Cscalar", "g"], ["gm9a"])         # u8
inits.append(cu("sent", 255))
add("Where", ["v", "gm9a", "sent"], ["gm9"])            # u8: out-of-grid = 255

# one-hot at 9x9 (interior [1,10,9,9] bool), then Pad to 30x30 as the graph output.
inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
add("Equal", ["gm9", "colors"], ["oh9"])                # bool [1,10,9,9]
inits.append(ci("p30", [0, 0, 0, 0, 0, 0, 30 - K, 30 - K]))
inits.append(cb("pf", False))
add("Pad", ["oh9", "p30", "pf"], ["output"])            # bool [1,10,30,30] (graph output)

x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "t288", [x_in], [y], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b18_task288.onnx"
onnx.save(model, path)
print("saved", path, "nodes", len(nodes), "inits", len(inits))
