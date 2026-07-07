"""Build compact ONNX for task288 (verified 267/267).

Rule: bottom grid row is [B..B C..C B..B] (B border, C inner). From the C-block's
two top corners, draw two diagonal rays of color C going up-and-outward (left ray
up-left along '\\', right ray up-right along '/') across the empty grid above.

Tensor pipeline on a single color grid g [1,1,30,30]:
  v       = validity mask (in-grid)               bool
  bottomr = v & ~shift_up(v)                       (last grid row)
  apex    = (g>0) & shift_up(bottomr) & v          (B-block at row rb-1)
  Lcorner = apex & ~shift_right(apex)              (left corner col cL)
  Rcorner = apex & ~shift_left(apex)
  Lseed   = shift(Lcorner, up1,left1);  Rseed = shift(Rcorner, up1,right1)
  L ray   = log-doubling OR of Lseed along up-left (shifts 1,2,4,8,16)
  R ray   = log-doubling OR of Rseed along up-right
  ray     = (L|R) & v
  C        = ReduceMax(g where cband), cband=shift_down(apex)  (inner color scalar)
  gm       = where(ray, C, g)                       uint8 color grid
  gm       = where(v, gm, 255)                      sentinel outside grid
  output   = Equal(gm, colors[0..9])  -> bool [1,10,30,30]  (graph output, excluded)

All interior tensors are bool or uint8 on 30x30 (<=900B each).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8  = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

def cf(name, v): return numpy_helper.from_array(np.asarray(v, np.float32), name)
def cu(name, v): return numpy_helper.from_array(np.asarray(v, np.uint8), name)
def ci(name, v): return numpy_helper.from_array(np.asarray(v, np.int64), name)
def cb(name, v): return numpy_helper.from_array(np.asarray(v, np.bool_), name)

nodes, inits = [], []

def add(op, ins, outs, **kw):
    nodes.append(helper.make_node(op, ins, outs, **kw))

# ---- color grid g [1,1,30,30] u8 from one-hot input (input excluded) ----
inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1,10,1,1)))
add("Conv", ["input","Wcol"], ["gf"])             # f32 [1,1,30,30], 0..9 (out-of-grid=0)
add("Cast", ["gf"], ["g"], to=U8)                 # u8 color grid

# validity mask v: sum over channels of input >0  -> use ReduceSum on input then >0.
inits.append(ci("ax1",[1]))
add("ReduceSum", ["input","ax1"], ["vsum"], keepdims=1)   # f32 [1,1,30,30]
inits.append(cf("zero",0.0))
add("Greater", ["vsum","zero"], ["v"])            # bool validity

# ---- shifts implemented as Pad-then-Slice (zero fill) ----
# helper to register a shift of a bool/u8 tensor by (dr,dc): content moves down dr, right dc.
shift_counter = [0]
def shift(src, dr, dc, dtype_is_bool=True):
    # pad amounts: to move content DOWN by dr (dr>0) pad top by dr; to move UP pad bottom.
    # ONNX Pad pads = [b0,b1,b2,b3, e0,e1,e2,e3] for 4D (begin then end per axis).
    pt = dr if dr>0 else 0      # pad top (begin axis2)
    pb = -dr if dr<0 else 0     # pad bottom (end axis2)
    pl = dc if dc>0 else 0      # pad left (begin axis3)
    pr = -dc if dc<0 else 0     # pad right (end axis3)
    name = f"sh{shift_counter[0]}"; shift_counter[0]+=1
    padname = name+"_p"
    inits.append(ci(padname, [0,0,pt,pl, 0,0,pb,pr]))
    if dtype_is_bool:
        inits.append(cb(name+"_cv", False))
    else:
        inits.append(cu(name+"_cv", 0))
    add("Pad", [src, padname, name+"_cv"], [name+"_pad"])
    # now crop back to 30x30: after padding top/left we slice off the extra at the
    # opposite ends. New shape = (30+pt+pb, 30+pl+pr). We want the 30x30 window that
    # corresponds to shifted content: starts at (pb, pr)?? Let's compute.
    # Content originally rows 0..29. After pad top pt: content at rows pt..pt+29.
    # We want output[r] = src[r-dr]; output rows 0..29 map to src rows -dr..29-dr.
    # In padded coords (offset by pt at top), src row k is at padded row k+pt.
    # output row r needs src row r-dr at padded row (r-dr)+pt. For dr>0 pt=dr ->padded row r. good.
    # We slice padded rows [start: start+30]. start = pt - dr? pt-dr: if dr>0 pt=dr ->0; if dr<0 pt=0,dr<0 -> -dr=pb start = 0-dr = pb. 
    rstart = pt - dr
    cstart = pl - dc
    inits.append(ci(name+"_ss", [rstart, cstart]))
    inits.append(ci(name+"_se", [rstart+30, cstart+30]))
    inits.append(ci(name+"_sa", [2,3]))
    add("Slice", [name+"_pad", name+"_ss", name+"_se", name+"_sa"], [name+"_out"])
    return name+"_out"

# bottomrow = v & ~shift(v, up1)  (shift up = dr=-1)
v_up = shift("v", -1, 0)
add("Not", [v_up], ["nv_up"])
add("And", ["v","nv_up"], ["bottomr"])

# g>0
inits.append(cu("u0",0))
add("Greater", ["g","u0"], ["gpos"])   # bool

# apex = gpos & shift(bottomr, up1) & v
br_up = shift("bottomr", -1, 0)
add("And", ["gpos", br_up], ["t1"])
add("And", ["t1","v"], ["apex"])

# corners
ap_r = shift("apex", 0, 1)   # shift right -> apex[r,c-1]
ap_l = shift("apex", 0, -1)
add("Not",[ap_r],["nap_r"]); add("And",["apex","nap_r"],["Lcorner"])
add("Not",[ap_l],["nap_l"]); add("And",["apex","nap_l"],["Rcorner"])

# seeds: left up-left (dr=-1,dc=-1); right up-right (dr=-1,dc=+1)
Lseed = shift("Lcorner", -1, -1)
Rseed = shift("Rcorner", -1,  1)

# log-doubling propagation
def propagate(seed, dc_sign, tag):
    cur = seed
    for k in [1,2,4,8,16]:
        s = shift(cur, -k, dc_sign*k)
        out = f"{tag}_{k}"
        add("Or", [cur, s], [out])
        cur = out
    return cur
Lray = propagate(Lseed, -1, "L")
Rray = propagate(Rseed,  1, "R")
add("Or", [Lray, Rray], ["ray0"])
add("And", ["ray0","v"], ["ray"])

# inner color C scalar: ReduceMax over gf masked to the C-block bottom cells.
# cband = shift(apex, down1) -> the C-block in the bottom grid row.
cband = shift("apex", 1, 0)
inits.append(cf("f0",0.0))
add("Where", [cband, "gf", "f0"], ["gc"])   # f32, =gf at C-block bottom cells else 0
add("ReduceMax", ["gc"], ["Cscalarf"], axes=[2,3], keepdims=1)  # f32 [1,1,1,1]
add("Cast", ["Cscalarf"], ["Cscalar"], to=U8)             # u8 scalar color

# gm = where(ray, Cscalar(broadcast), g)
add("Where", ["ray","Cscalar","g"], ["gm0"])   # u8 [1,1,30,30]
# gm = where(v, gm0, 255)
inits.append(cu("sent",255))
add("Where", ["v","gm0","sent"], ["gm"])

# output = Equal(gm, colors[0..9])
inits.append(cu("colors", np.arange(10,dtype=np.uint8).reshape(1,10,1,1)))
add("Equal", ["gm","colors"], ["output"])   # bool [1,10,30,30]

x_in = helper.make_tensor_value_info("input", F32, [1,10,30,30])
y    = helper.make_tensor_value_info("output", BOOL, [1,10,30,30])
graph = helper.make_graph(nodes, "t288", [x_in], [y], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])
model.ir_version = 9
import sys
path = sys.argv[1] if len(sys.argv)>1 else "/tmp/b10_task288.onnx"
onnx.save(model, path)
print("saved", path, "nodes", len(nodes), "inits", len(inits))
