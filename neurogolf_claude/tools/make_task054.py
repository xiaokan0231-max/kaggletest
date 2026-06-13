"""Build compact ONNX for task054.
Algorithm (verified on all 266 examples in /tmp/onnxsim2.py):
 - input one-hot [1,10,30,30]; derive color grid g [1,1,30,30]
 - bg = most common border color; F = largest non-bg color
 - legend center (cy,cx) = bbox center of candidate cells (non-bg,non-F, no F within cheby dist2)
 - sample M, arm colors, template, extension flags via shifted center one-hot + reductions
 - seeds = M-colored non-legend cells; run = F or seed
 - segment-fill (cumsum + WxW broadcast equal) clips lines to the rectangle
 - stamp 5x5 legend template at each seed inside the rect
Output one-hot via equality of final color grid to each channel index, masked by active region.
All tensors kept in [1,1,30,30] or smaller except the two [1,30,30,30] segfill tensors.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/wf_task054.onnx"
H = W = 30
nodes = []
inits = []
def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name
def n(op, i, o, **kw):
    if isinstance(o, str): o=[o]
    if isinstance(i, str): i=[i]
    nodes.append(helper.make_node(op, i, o, **kw)); return o[0]

# ---- constants ----
colors = np.arange(10, dtype=np.float32).reshape(1,10,1,1)
C("colors", colors)                      # [1,10,1,1]
ix = (np.arange(W, dtype=np.float32).reshape(1,1,1,W) * np.ones((1,1,H,1),np.float32))
iy = (np.arange(H, dtype=np.float32).reshape(1,1,H,1) * np.ones((1,1,1,W),np.float32))
C("ix", ix); C("iy", iy)                 # [1,1,30,30]
bmask = np.zeros((1,1,H,W), np.float32); bmask[0,0,0,:]=1; bmask[0,0,-1,:]=1; bmask[0,0,:,0]=1; bmask[0,0,:,-1]=1
C("bmask", bmask)
C("one", np.array(1.0, np.float32))
C("half", np.array(0.5, np.float32))
C("two", np.array(2.0, np.float32))
C("big", np.array(1e4, np.float32))
C("negbig", np.array(-1e4, np.float32))
C("eps", np.array(0.25, np.float32))
C("zero", np.array(0.0, np.float32))
C("ax1", np.array([1], np.int64))
C("axHW", np.array([2,3], np.int64))
C("scal1", np.array([1], np.int64))         # reshape scalar->[1]
C("scal4", np.array([1,1,1,1], np.int64))   # reshape ->[1,1,1,1]
C("twoi", np.array([2], np.int64))
C("threei", np.array([3], np.int64))
C("pax23", np.array([2,3], np.int64))       # slice axes for spatial
C("c22s", np.array([2,2], np.int64)); C("c22e", np.array([3,3], np.int64))

# ---- color grid g = sum_c c*onehot ----
n("Mul", ["input","colors"], "gc")             # [1,10,30,30]
n("ReduceSum", ["gc","ax1"], "g", keepdims=1)  # [1,1,30,30]

# ---- bg: border counts per color, argmax ----
n("ReduceSum", ["input","axHW"], "tot", keepdims=1)        # [1,10,1,1] total per color
n("Mul", ["input","bmask"], "binp")                        # [1,10,30,30]
n("ReduceSum", ["binp","axHW"], "bcount", keepdims=1)      # [1,10,1,1]
n("ArgMax", ["bcount"], "bgidx", axis=1, keepdims=1)       # [1,1,1,1] int64
n("Cast", ["bgidx"], "bgf", to=TensorProto.FLOAT)          # bg as float [1,1,1,1]
# F: largest non-bg color. mask bg's total to -1.
# bg one-hot over channel: colors == bg
n("Equal", ["colors","bgf"], "bgoh_b")                     # [1,10,1,1] bool (broadcast)
n("Cast", ["bgoh_b"], "bgoh", to=TensorProto.FLOAT)        # 1 at bg channel
n("Mul", ["bgoh","big"], "bgpenalty")
n("Sub", ["tot","bgpenalty"], "tot2")                      # bg channel pushed down
n("ArgMax", ["tot2"], "Fidx", axis=1, keepdims=1)
n("Cast", ["Fidx"], "Ff", to=TensorProto.FLOAT)            # F float [1,1,1,1]

# ---- masks over grid ----
n("Equal", ["g","bgf"], "isbg_b"); n("Cast", ["isbg_b"], "isbg", to=TensorProto.FLOAT)
n("Equal", ["g","Ff"], "isF_b"); n("Cast", ["isF_b"], "isF", to=TensorProto.FLOAT)
# nonbgF = 1 - isbg - isF (cells are exactly one color)
n("Add", ["isbg","isF"], "bgF")
n("Sub", ["one","bgF"], "nonbgF")                          # [1,1,30,30]

# ---- F dilation by cheby 2 (maxpool 5x5) ----
n("MaxPool", ["isF"], "Fd", kernel_shape=[5,5], pads=[2,2,2,2], strides=[1,1])
# candidate cells cc = nonbgF * (Fd<=0)
n("LessOrEqual", ["Fd","zero"], "noF_b"); n("Cast", ["noF_b"], "noF", to=TensorProto.FLOAT)
n("Mul", ["nonbgF","noF"], "cc")                           # [1,1,30,30] 0/1

# ---- bbox center of cc ----
# xmin = min over grid of where(cc, ix, big); etc.
def masked_extreme(src, name, big_const, reduce_op):
    # out = reduce_op over HW of  cc>0 ? src : big_const
    n("Sub", ["one","cc"], name+"_inv")          # 1 where not cc
    n("Mul", [name+"_inv", big_const], name+"_pen")
    n("Mul", ["cc", src], name+"_v")
    n("Add", [name+"_v", name+"_pen"], name+"_f")  # cc? src : big_const  (since cc?src:0 + notcc?big:0)
    n(reduce_op, [name+"_f","axHW"], name+"_r", keepdims=1)  # [1,1,1,1]
    return name+"_r"
xmin = masked_extreme("ix","xmin","big","ReduceMin")
ymin = masked_extreme("iy","ymin","big","ReduceMin")
# for max use negbig penalty
n("Sub", ["one","cc"], "ccinv2"); n("Mul", ["ccinv2","negbig"], "negpen")
n("Mul", ["cc","ix"], "ccx"); n("Add", ["ccx","negpen"], "xmaxf"); n("ReduceMax", ["xmaxf","axHW"], "xmax", keepdims=1)
n("Mul", ["cc","iy"], "ccy"); n("Add", ["ccy","negpen"], "ymaxf"); n("ReduceMax", ["ymaxf","axHW"], "ymax", keepdims=1)
# cx = floor((xmin+xmax)/2)
n("Add", [xmin,"xmax"], "xsum"); n("Mul", ["xsum","half"], "cxh"); n("Floor", ["cxh"], "cx")
n("Add", [ymin,"ymax"], "ysum"); n("Mul", ["ysum","half"], "cyh"); n("Floor", ["cyh"], "cy")

# ---- center one-hot cmask = (ix==cx)&(iy==cy) ----
n("Equal", ["ix","cx"], "ex_b"); n("Cast", ["ex_b"], "ex", to=TensorProto.FLOAT)
n("Equal", ["iy","cy"], "ey_b"); n("Cast", ["ey_b"], "ey", to=TensorProto.FLOAT)
n("Mul", ["ex","ey"], "cmask")                              # [1,1,30,30] one-hot center

# helper: shift a [1,1,30,30] by (dy,dx) via Pad+Slice. We'll precompute shifted cmask for needed offsets.
# Implement shift using Pad with zeros then Slice.
def shift(src, dy, dx, name):
    # produce src shifted so that value originally at (r,c) moves to (r+dy, c+dx)
    # pad amounts then slice 30x30
    pt = max(dy,0); pb = max(-dy,0); pl = max(dx,0); pr = max(-dx,0)
    pads = np.array([0,0,pt,pl, 0,0,pb,pr], np.int64)
    C(name+"_pads", pads)
    n("Pad", [src, name+"_pads"], name+"_pad")
    starts = np.array([pb, pr], np.int64); ends = np.array([pb+H, pr+W], np.int64)
    C(name+"_st", starts); C(name+"_en", ends); C(name+"_axs", np.array([2,3],np.int64))
    n("Slice", [name+"_pad", name+"_st", name+"_en", name+"_axs"], name+"_out")
    return name+"_out"

# ---- extract a tiny [1,1,5,5] patch of g centered at (cy,cx) via dynamic Slice ----
# cy,cx in [3,26] so cy-2>=1, cy+2<=28: always safe. Cast to int64.
n("Cast",["cy"],"cyi",to=TensorProto.INT64); n("Cast",["cx"],"cxi",to=TensorProto.INT64)
n("Reshape",["cyi","scal1"],"cyi1"); n("Reshape",["cxi","scal1"],"cxi1")  # [1] each
n("Sub",["cyi1","twoi"],"y0i"); n("Sub",["cxi1","twoi"],"x0i")
n("Add",["cyi1","threei"],"y1i"); n("Add",["cxi1","threei"],"x1i")
n("Concat",["y0i","x0i"],"pstart",axis=0); n("Concat",["y1i","x1i"],"pend",axis=0)
n("Slice",["g","pstart","pend","pax23"],"patch")   # [1,1,5,5] tiny
# present = (patch != bg) -> [1,1,5,5]; Vk = patch where present else 0
n("Equal",["patch","bgf"],"pisbg_b"); n("Cast",["pisbg_b"],"pisbg",to=TensorProto.FLOAT)
n("Sub",["one","pisbg"],"ppres")            # [1,1,5,5] present (1 where non-bg)
n("Mul",["patch","ppres"],"pval")           # value where present else 0
# M = patch[2,2]
n("Slice",["patch","c22s","c22e","pax23"],"M0"); n("Reshape",["M0","scal4"],"M")  # [1,1,1,1]
# arm colors / ext flags from specific patch cells (static indices)
def cell(r,cc,nm):
    C(nm+"_s",np.array([r,cc],np.int64)); C(nm+"_e",np.array([r+1,cc+1],np.int64))
    n("Slice",["patch",nm+"_s",nm+"_e","pax23"],nm+"_v")          # value [1,1,1,1]
    n("Slice",["ppres",nm+"_s",nm+"_e","pax23"],nm+"_p")          # present [1,1,1,1]
    return nm+"_v", nm+"_p"
vU,pU = cell(1,2,"cU"); vD,pD = cell(3,2,"cD")
vL,pL = cell(2,1,"cL"); vR,pR = cell(2,3,"cR")
_,pU2 = cell(0,2,"cU2"); _,pD2 = cell(4,2,"cD2")
_,pL2 = cell(2,0,"cL2"); _,pR2 = cell(2,4,"cR2")
# Lv = pU? vU : vD ; Lh = pL? vL : vR
n("Mul",[pU,vU],"lv1"); n("Sub",["one",pU],"npU"); n("Mul",["npU",vD],"lv2"); n("Add",["lv1","lv2"],"Lv")
n("Mul",[pL,vL],"lh1"); n("Sub",["one",pL],"npL"); n("Mul",["npL",vR],"lh2"); n("Add",["lh1","lh2"],"Lh")
n("Add",[pU2,pD2],"vsum"); n("Clip",["vsum","zero","one"],"vert_ext")
n("Add",[pL2,pR2],"hsum"); n("Clip",["hsum","zero","one"],"horiz_ext")

# ---- legend mask & seed & run ----
# win = |ix-cx|<=2 & |iy-cy|<=2
n("Sub",["ix","cx"],"dxc"); n("Abs",["dxc"],"adxc"); n("LessOrEqual",["adxc","two"],"winx_b"); n("Cast",["winx_b"],"winx",to=TensorProto.FLOAT)
n("Sub",["iy","cy"],"dyc"); n("Abs",["dyc"],"adyc"); n("LessOrEqual",["adyc","two"],"winy_b"); n("Cast",["winy_b"],"winy",to=TensorProto.FLOAT)
n("Mul",["winx","winy"],"win")
# legend = win * nonbgF
n("Mul",["win","nonbgF"],"legend")              # [1,1,30,30] 0/1
# seed = (g==M) & not legend
n("Equal",["g","M"],"isM_b"); n("Cast",["isM_b"],"isM",to=TensorProto.FLOAT)
n("Sub",["one","legend"],"notleg"); n("Mul",["isM","notleg"],"seed")   # [1,1,30,30]
# run = max(isF, seed)
n("Max",["isF","seed"],"run")

# ---- run_has_seed via cumulative ops only (all [1,1,30,30]) ----
C("ax3",np.array([3],np.int64)); C("ax2v",np.array([2],np.int64))
C("neg9",np.array(-1e9,np.float32))
n("Sub",["one","run"],"notrun")
# reverse along axis via Slice with step -1
def reverse(src, axis, name):
    st=np.array([H-1 if axis==2 else W-1],np.int64); en=np.array([-(H+1) if axis==2 else -(W+1)],np.int64)
    stp=np.array([-1],np.int64); ax=np.array([axis],np.int64)
    C(name+"_st",st); C(name+"_en",en); C(name+"_ax",ax); C(name+"_sp",stp)
    n("Slice",[src,name+"_st",name+"_en",name+"_ax",name+"_sp"], name); return name
def cummax(src, axis, name):
    # log-doubling cummax along axis (left->right). 5 steps cover 30.
    cur=src
    for k in range(5):
        s=1<<k
        if (axis==3 and s>=W) or (axis==2 and s>=H): break
        # shift cur right by s along axis (fill -1e9), then max
        if axis==3:
            pads=np.array([0,0,0,s, 0,0,0,0],np.int64)
            stt=np.array([0],np.int64); enn=np.array([W],np.int64); aaa=np.array([3],np.int64)
        else:
            pads=np.array([0,0,s,0, 0,0,0,0],np.int64)
            stt=np.array([0],np.int64); enn=np.array([H],np.int64); aaa=np.array([2],np.int64)
        pn=f"{name}_p{k}"; C(pn+"_pads",pads); C(pn+"_pv",np.array(-1e9,np.float32))
        n("Pad",[cur,pn+"_pads",pn+"_pv"], pn+"_pad")
        C(pn+"_st",stt); C(pn+"_en",enn); C(pn+"_ax",aaa)
        n("Slice",[pn+"_pad",pn+"_st",pn+"_en",pn+"_ax"], pn+"_sh")
        n("Max",[cur,pn+"_sh"], pn+"_m")
        cur=pn+"_m"
    return cur
def run_has_seed(seedn, runn, axis, name):
    notrunn = "notrun"
    cs = name+"_cs"; n("CumSum",[seedn, "ax3" if axis==3 else "ax2v"], cs)
    n("Mul",[cs, notrunn], name+"_csm")
    base = cummax(name+"_csm", axis, name+"_bm")
    n("Sub",[cs, base], name+"_sl")     # seeds-left-in-run
    # reverse seed & run, recompute, reverse back
    rs = reverse(seedn, axis, name+"_rs"); rr = reverse(runn, axis, name+"_rr")
    csr = name+"_csr"; n("CumSum",[rs, "ax3" if axis==3 else "ax2v"], csr)
    n("Sub",["one",rr], name+"_nrr"); n("Mul",[csr, name+"_nrr"], name+"_csrm")
    baser = cummax(name+"_csrm", axis, name+"_bmr")
    n("Sub",[csr, baser], name+"_srr")
    sr = reverse(name+"_srr", axis, name+"_srrev")
    n("Add",[name+"_sl", sr], name+"_tot0"); n("Sub",[name+"_tot0", seedn], name+"_total")
    n("Greater",[name+"_total","half"], name+"_hit_b"); n("Cast",[name+"_hit_b"], name+"_hitf", to=TensorProto.FLOAT)
    n("Mul",[name+"_hitf", runn], name+"_hit")
    return name+"_hit"
Hh = run_has_seed("seed","run",3,"RH")
Vv = run_has_seed("seed","run",2,"RV")
n("Identity",[Hh],"Hh"); n("Identity",[Vv],"Vv")

# ---- compose output color grid ----
# start: out = g, then legend cells -> bg, then lines, then stamp.
# erase legend: out = g*(1-legend) + bg*legend
n("Sub",["one","legend"],"nleg2"); n("Mul",["g","nleg2"],"o0a"); n("Mul",["bgf","legend"],"o0b"); n("Add",["o0a","o0b"],"o0")
# horizontal line: apply Lh where Hh>0 AND horiz_ext
n("Mul",["Hh","horiz_ext"],"Happly")   # [1,1,30,30]*scalar
# where Happly>0 set Lh: out = o0*(1-Happly)+Lh*Happly  (Happly is 0/1)
n("Sub",["one","Happly"],"nHa"); n("Mul",["o0","nHa"],"o1a"); n("Mul",["Lh","Happly"],"o1b"); n("Add",["o1a","o1b"],"o1")
n("Mul",["Vv","vert_ext"],"Vapply")
n("Sub",["one","Vapply"],"nVa"); n("Mul",["o1","nVa"],"o2a"); n("Mul",["Lv","Vapply"],"o2b"); n("Add",["o2a","o2b"],"o2")

# inside mask = max(isF, seed, Hh, Vv)
n("Max",["isF","seed"],"in0"); n("Max",["in0","Hh"],"in1"); n("Max",["in1","Vv"],"inside")

# ---- stamp template via two Conv ops (kernel built from 25 sampled scalars) ----
# The patch IS the 5x5 template. Vk = pval (value where present), Pk = ppres. Flip for Conv scatter.
C("flip_st",np.array([4,4],np.int64)); C("flip_en",np.array([-6,-6],np.int64))
C("flip_ax",np.array([2,3],np.int64)); C("flip_sp",np.array([-1,-1],np.int64))
n("Slice",["pval","flip_st","flip_en","flip_ax","flip_sp"],"Vk")     # [1,1,5,5]
n("Slice",["ppres","flip_st","flip_en","flip_ax","flip_sp"],"Pk")
# Conv: pad 2 so output stays [1,1,30,30]
n("Conv",["seed","Vk"],"stampval",pads=[2,2,2,2],kernel_shape=[5,5])
n("Conv",["seed","Pk"],"stamppres",pads=[2,2,2,2],kernel_shape=[5,5])
# apply = (stamppres>0.5) * inside  ; finalg = o2*(1-apply)+stampval*apply
n("Greater",["stamppres","half"],"sp_b"); n("Cast",["sp_b"],"spf",to=TensorProto.FLOAT)
n("Mul",["spf","inside"],"apply")
n("Sub",["one","apply"],"napply"); n("Mul",["o2","napply"],"fa"); n("Mul",["stampval","apply"],"fb")
n("Add",["fa","fb"],"finalg")
finalg = "finalg"

# ---- active mask: cells where input had any channel set ----
n("ReduceSum",["input","ax1"],"act0",keepdims=1)   # [1,1,30,30]
n("Greater",["act0","half"],"act_b"); n("Cast",["act_b"],"active",to=TensorProto.FLOAT)
# Mask finalg outside active region to a sentinel that never equals any color 0..9.
# fgm = finalg*active + (active-1)*BIG  -> outside: -BIG ; inside: finalg
n("Mul",[finalg,"active"],"fg_in")
n("Sub",["active","one"],"actm1"); n("Mul",["actm1","big"],"fg_out")
n("Add",["fg_in","fg_out"],"fgm")
# Single [1,10,30,30] bool intermediate, then cast straight into the free 'output'.
n("Equal",["fgm","colors"],"oh_b")     # [1,10,30,30] bool (only big intermediate)
n("Cast",["oh_b"],"output",to=TensorProto.FLOAT)

graph = helper.make_graph(nodes, "task054",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,10,30,30])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",18)])
model.ir_version = 10
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
nparams = sum(int(np.prod(t.dims)) if t.dims else 1 for t in inits)
print(f"saved {OUT} nodes={len(nodes)} params={nparams}")
