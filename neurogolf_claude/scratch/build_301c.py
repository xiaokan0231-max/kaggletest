"""task301 v3: scatter outcolor; big [30,30] grids only bool/uint8/f16."""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
nodes=[]; inits=[]
def C(n,a): inits.append(numpy_helper.from_array(np.asarray(a),n))
def N(op,i,o,**k): nodes.append(helper.make_node(op,i,o,**k))
I32=TensorProto.INT32; I64=TensorProto.INT64; U8=TensorProto.UINT8
F=TensorProto.FLOAT; F16=TensorProto.FLOAT16; B=TensorProto.BOOL

# ---- color grid ----
C("w_color", np.arange(10,dtype=np.float32).reshape(1,10,1,1))
N("Conv",["input","w_color"],["g"])                       # [1,1,30,30] f32 (3600)
N("Cast",["g"],["gu"],to=U8)                              # u8 (900)
C("zf",np.array(0.5,np.float32))
N("Greater",["g","zf"],["mask_b"])                        # bool (900)
N("Cast",["mask_b"],["mask16"],to=F16)                    # f16 (1800)

# ---- per-row length & color ----
C("ax3",np.array([3],np.int64))
N("ReduceSum",["mask16","ax3"],["rowlen16"],keepdims=1)   # f16 [1,1,30,1]
N("Cast",["rowlen16"],["rowlen"],to=I32)                  # i32 [1,1,30,1]
N("ReduceMax",["gu","ax3"],["rowcolor_u"],keepdims=1)     # u8 [1,1,30,1]
N("Cast",["rowcolor_u"],["rowcolor"],to=I32)              # i32

# ---- anyrow, N ----
C("z32",np.array(0,np.int32))
N("Greater",["rowlen","z32"],["anyrow_b"])                # bool [1,1,30,1]
N("Cast",["anyrow_b"],["anyrow32"],to=I32)
C("ax2",np.array([2],np.int64))
N("ReduceSum",["anyrow32","ax2"],["Nn"],keepdims=1)       # i32 [1,1,1,1] = N

# ---- anycol, H, W ----
N("ReduceMax",["gu","ax2"],["colmax_u"],keepdims=1)       # u8 [1,1,1,30] max color per col
N("Cast",["colmax_u"],["colmax32"],to=I32)
N("Greater",["colmax32","z32"],["anycol_b"])              # bool [1,1,1,30]
N("Cast",["anycol_b"],["anycol32"],to=I32)
# H = max(r*anyrow)+1
C("ridx_col",np.arange(30,dtype=np.int32).reshape(1,1,30,1))
N("Mul",["ridx_col","anyrow32"],["r_anyrow"])             # i32 [1,1,30,1]
N("ReduceMax",["r_anyrow","ax2"],["maxr"],keepdims=1)     # i32 [1,1,1,1]
C("one32",np.array(1,np.int32))
N("Add",["maxr","one32"],["H"])                           # i32
# W = max(c*anycol)+1
C("cidx_row",np.arange(30,dtype=np.int32).reshape(1,1,1,30))
N("Mul",["cidx_row","anycol32"],["c_anycol"])             # i32 [1,1,1,30]
N("ReduceMax",["c_anycol","ax3"],["maxc"],keepdims=1)     # i32
N("Add",["maxc","one32"],["W"])                           # i32

# ---- targetrow = (H-N)+(rowlen-1); empty rows -> dump idx 30 ----
N("Sub",["H","Nn"],["HmN"])                               # i32 [1,1,1,1]
N("Add",["HmN","rowlen"],["tr_a"])                        # i32 [1,1,30,1]
N("Sub",["tr_a","one32"],["targetrow"])                   # i32 [1,1,30,1]
C("dump",np.array(30,np.int32))
N("Where",["anyrow_b","targetrow","dump"],["tr_idx"])     # i32 [1,1,30,1]  empty->30

# ---- scatter rowcolor into outbuf[1,1,31,1] at tr_idx (axis2) ----
C("outbuf0", np.zeros((1,1,31,1),np.int32))
N("ScatterElements",["outbuf0","tr_idx","rowcolor"],["outbuf"],axis=2)  # i32 [1,1,31,1]
C("s0",np.array([0],np.int64)); C("s30",np.array([30],np.int64)); C("ax2s",np.array([2],np.int64))
N("Slice",["outbuf","s0","s30","ax2s"],["outcolor"])      # i32 [1,1,30,1] color per out-row
N("Cast",["outcolor"],["outcolor_u"],to=U8)               # u8 [1,1,30,1]

# ---- staircase fill ----
# T = W+H-1-N
N("Add",["W","H"],["WpH"]); N("Sub",["WpH","one32"],["WpHm1"]); N("Sub",["WpHm1","Nn"],["T"])  # i32 [1,1,1,1]
# rpc const (param)
rpc = (np.arange(30).reshape(30,1)+np.arange(30).reshape(1,30)).astype(np.int32).reshape(1,1,30,30)
C("rpc", rpc)
N("GreaterOrEqual",["rpc","T"],["ge_b"])                  # bool [1,1,30,30] (900)
N("Less",["ridx_col","H"],["rltH_b"])                     # bool [1,1,30,1]
N("Less",["cidx_row","W"],["cltW_b"])                     # bool [1,1,1,30]
N("And",["ge_b","rltH_b"],["f1_b"])                       # bool [1,1,30,30]
N("And",["f1_b","cltW_b"],["filled_b"])                   # bool [1,1,30,30] (900)
N("And",["rltH_b","cltW_b"],["ingrid_b"])                 # bool [1,1,30,30] (900)

# gm = filled? outcolor : 0 ; then ingrid? gm : 255
C("z_u",np.array(0,np.uint8))
N("Where",["filled_b","outcolor_u","z_u"],["gm_in"])      # u8 [1,1,30,30] (900) (outcolor_u broadcasts axis2)
C("s255",np.array(255,np.uint8))
N("Where",["ingrid_b","gm_in","s255"],["gm_u"])           # u8 [1,1,30,30] (900)

C("colors",np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
N("Equal",["gm_u","colors"],["output"])                   # bool [1,10,30,30] OUTPUT (excluded)

g=helper.make_graph(nodes,"task301",
    [helper.make_tensor_value_info("input",F,[1,10,30,30])],
    [helper.make_tensor_value_info("output",B,[1,10,30,30])], inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=9)
onnx.checker.check_model(m)
onnx.save(m,"/tmp/b13_task301.onnx")
print("saved")
