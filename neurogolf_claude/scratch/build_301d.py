"""task301 v4: minimal [30,30] tensors. color grids g(f32)+g16(f16); rowlen via sum/color; per-row staircase threshold."""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
nodes=[]; inits=[]
def C(n,a): inits.append(numpy_helper.from_array(np.asarray(a),n))
def N(op,i,o,**k): nodes.append(helper.make_node(op,i,o,**k))
I32=TensorProto.INT32; I64=TensorProto.INT64; U8=TensorProto.UINT8
F=TensorProto.FLOAT; F16=TensorProto.FLOAT16; B=TensorProto.BOOL

# ===== color grid (only two [30,30] tensors live: g f32, g16 f16) =====
C("w_color", np.arange(10,dtype=np.float32).reshape(1,10,1,1))
N("Conv",["input","w_color"],["g"])                       # [1,1,30,30] f32  (3600)
N("Cast",["g"],["g16"],to=F16)                            # [1,1,30,30] f16  (1800)

C("ax3",np.array([3],np.int64))
C("ax2",np.array([2],np.int64))
# per-row color (max) & sum
N("ReduceMax",["g16","ax3"],["rowcolor16"],keepdims=1)    # f16 [1,1,30,1]
N("ReduceSum",["g16","ax3"],["rowsum16"],keepdims=1)      # f16 [1,1,30,1]
# rowlen = round(rowsum/(rowcolor+eps))
C("eps16",np.array(1e-3,np.float16))
N("Add",["rowcolor16","eps16"],["rc_safe"])
N("Div",["rowsum16","rc_safe"],["rl_f16"])
N("Round",["rl_f16"],["rl_rnd"])
N("Cast",["rl_rnd"],["rowlen"],to=I32)                    # i32 [1,1,30,1]
N("Cast",["rowcolor16"],["rowcolor"],to=I32)             # i32 [1,1,30,1]

# anyrow, N
C("z32",np.array(0,np.int32))
N("Greater",["rowlen","z32"],["anyrow_b"])               # bool [1,1,30,1]
N("Cast",["anyrow_b"],["anyrow32"],to=I32)
N("ReduceSum",["anyrow32","ax2"],["Nn"],keepdims=1)      # i32 [1,1,1,1]

# anycol via column max color
N("ReduceMax",["g16","ax2"],["colmax16"],keepdims=1)     # f16 [1,1,1,30]
N("Cast",["colmax16"],["colmax32"],to=I32)
N("Greater",["colmax32","z32"],["anycol_b"])             # bool [1,1,1,30]
N("Cast",["anycol_b"],["anycol32"],to=I32)

# H = max(r*anyrow)+1 ; W = max(c*anycol)+1
C("ridx_col",np.arange(30,dtype=np.int32).reshape(1,1,30,1))
C("cidx_row",np.arange(30,dtype=np.int32).reshape(1,1,1,30))
C("one32",np.array(1,np.int32))
N("Mul",["ridx_col","anyrow32"],["r_anyrow"])
N("ReduceMax",["r_anyrow","ax2"],["maxr"],keepdims=1)
N("Add",["maxr","one32"],["H"])
N("Mul",["cidx_row","anycol32"],["c_anycol"])
N("ReduceMax",["c_anycol","ax3"],["maxc"],keepdims=1)
N("Add",["maxc","one32"],["W"])

# targetrow = (H-N)+(rowlen-1); empty rows -> dump 30
N("Sub",["H","Nn"],["HmN"])
N("Add",["HmN","rowlen"],["tr_a"])
N("Sub",["tr_a","one32"],["targetrow"])                  # i32 [1,1,30,1]
C("dump",np.array(30,np.int32))
N("Where",["anyrow_b","targetrow","dump"],["tr_idx"])    # i32 [1,1,30,1]

# scatter rowcolor into outbuf[1,1,31,1] at tr_idx
C("outbuf0", np.zeros((1,1,31,1),np.int32))
N("ScatterElements",["outbuf0","tr_idx","rowcolor"],["outbuf"],axis=2)  # i32 [1,1,31,1]
C("s0",np.array([0],np.int64)); C("s30",np.array([30],np.int64))
N("Slice",["outbuf","s0","s30","ax2"],["outcolor"])      # i32 [1,1,30,1]
N("Cast",["outcolor"],["outcolor_u"],to=U8)              # u8 [1,1,30,1]

# ===== staircase (per-row threshold, no rpc const) =====
# T = W+H-1-N ; filled[r,c] = c >= (T - r) AND r<H AND c<W
N("Add",["W","H"],["WpH"]); N("Sub",["WpH","one32"],["WpHm1"]); N("Sub",["WpHm1","Nn"],["T"])  # i32 [1,1,1,1]
N("Sub",["T","ridx_col"],["thr"])                        # i32 [1,1,30,1] = T-r per row
N("GreaterOrEqual",["cidx_row","thr"],["ge_b"])          # bool [1,1,30,30] (900) broadcast
N("Less",["ridx_col","H"],["rltH_b"])                    # bool [1,1,30,1]
N("Less",["cidx_row","W"],["cltW_b"])                    # bool [1,1,1,30]
N("And",["rltH_b","cltW_b"],["ingrid_b"])                # bool [1,1,30,30] (900)
N("And",["ge_b","ingrid_b"],["filled_b"])               # bool [1,1,30,30] (900)

# gm = filled? color : (ingrid? 0 : 255)
C("z_u",np.array(0,np.uint8)); C("s255",np.array(255,np.uint8))
N("Where",["ingrid_b","z_u","s255"],["bg_u"])            # u8 [1,1,30,30] (900) 0 in-grid else 255
N("Where",["filled_b","outcolor_u","bg_u"],["gm_u"])     # u8 [1,1,30,30] (900)

C("colors",np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
N("Equal",["gm_u","colors"],["output"])                  # bool [1,10,30,30] OUTPUT (excluded)

g=helper.make_graph(nodes,"task301",
    [helper.make_tensor_value_info("input",F,[1,10,30,30])],
    [helper.make_tensor_value_info("output",B,[1,10,30,30])], inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=9)
onnx.checker.check_model(m)
onnx.save(m,"/tmp/b13_task301.onnx")
print("saved")
