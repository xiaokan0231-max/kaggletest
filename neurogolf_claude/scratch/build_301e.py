"""task301 v5: NO color grid. Conv gives row counts/sums & col counts as tiny vecs."""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
nodes=[]; inits=[]
def C(n,a): inits.append(numpy_helper.from_array(np.asarray(a),n))
def N(op,i,o,**k): nodes.append(helper.make_node(op,i,o,**k))
I32=TensorProto.INT32; U8=TensorProto.UINT8; F=TensorProto.FLOAT; F16=TensorProto.FLOAT16; B=TensorProto.BOOL

# ===== Conv reductions (tiny outputs, no [30,30] grid) =====
wcnt=np.zeros((1,10,1,30),np.float32); wcnt[0,1:,0,:]=1.0   # rowcnt [1,1,30,1]
wsum=np.zeros((1,10,1,30),np.float32)
for ch in range(10): wsum[0,ch,0,:]=ch                      # rowsum (=c*L) [1,1,30,1]
C("wcnt",wcnt); C("wsum",wsum)
N("Conv",["input","wcnt"],["rowcnt_f"])      # f32 [1,1,30,1]
N("Conv",["input","wsum"],["rowsum_f"])      # f32 [1,1,30,1]

# rowlen (int) and rowcolor (=rowsum/rowcnt)
N("Round",["rowcnt_f"],["rowcnt_r"])
N("Cast",["rowcnt_r"],["rowlen"],to=I32)                    # i32 [1,1,30,1]
C("epsf",np.array(1e-3,np.float32))
N("Add",["rowcnt_f","epsf"],["cnt_safe"])
N("Div",["rowsum_f","cnt_safe"],["rowcolor_f"])
N("Round",["rowcolor_f"],["rowcolor_r"])
N("Cast",["rowcolor_r"],["rowcolor"],to=I32)               # i32 [1,1,30,1]

C("z32",np.array(0,np.int32))
C("ax2",np.array([2],np.int64)); C("ax3",np.array([3],np.int64))
# anyrow & N
N("Greater",["rowlen","z32"],["anyrow_b"])                 # bool [1,1,30,1]
N("Cast",["anyrow_b"],["anyrow32"],to=I32)
N("ReduceSum",["anyrow32","ax2"],["Nn"],keepdims=1)        # i32 [1,1,1,1]
# anycol
N("Round",["colcnt_f"],["colcnt_r"]); N("Cast",["colcnt_r"],["colcnt"],to=I32)  # i32 [1,1,1,30]
N("Greater",["colcnt","z32"],["anycol_b"])
N("Cast",["anycol_b"],["anycol32"],to=I32)

# H,W
C("ridx_col",np.arange(30,dtype=np.int32).reshape(1,1,30,1))
C("cidx_row",np.arange(30,dtype=np.int32).reshape(1,1,1,30))
C("one32",np.array(1,np.int32))
N("Mul",["ridx_col","anyrow32"],["r_anyrow"]); N("ReduceMax",["r_anyrow","ax2"],["maxr"],keepdims=1); N("Add",["maxr","one32"],["H"])
N("Mul",["cidx_row","anycol32"],["c_anycol"]); N("ReduceMax",["c_anycol","ax3"],["maxc"],keepdims=1); N("Add",["maxc","one32"],["W"])

# targetrow & scatter outcolor
N("Sub",["H","Nn"],["HmN"]); N("Add",["HmN","rowlen"],["tr_a"]); N("Sub",["tr_a","one32"],["targetrow"])
C("dump",np.array(30,np.int32))
N("Where",["anyrow_b","targetrow","dump"],["tr_idx"])      # i32 [1,1,30,1]
C("outbuf0",np.zeros((1,1,31,1),np.int32))
N("ScatterElements",["outbuf0","tr_idx","rowcolor"],["outbuf"],axis=2)
C("s0",np.array([0],np.int64)); C("s30",np.array([30],np.int64))
N("Slice",["outbuf","s0","s30","ax2"],["outcolor"]); N("Cast",["outcolor"],["outcolor_u"],to=U8)  # u8 [1,1,30,1]

# staircase: T=W+H-1-N ; thr=T-r ; filled = (c>=thr)&(r<H)&(c<W)
N("Add",["W","H"],["WpH"]); N("Sub",["WpH","one32"],["WpHm1"]); N("Sub",["WpHm1","Nn"],["T"])
N("Sub",["T","ridx_col"],["thr"])                          # i32 [1,1,30,1]
N("GreaterOrEqual",["cidx_row","thr"],["ge_b"])            # bool [1,1,30,30] (900)
N("Less",["ridx_col","H"],["rltH_b"]); N("Less",["cidx_row","W"],["cltW_b"])
N("And",["rltH_b","cltW_b"],["ingrid_b"])                  # bool [1,1,30,30] (900)
N("And",["ge_b","ingrid_b"],["filled_b"])                 # bool [1,1,30,30] (900)

C("z_u",np.array(0,np.uint8)); C("s255",np.array(255,np.uint8))
N("Where",["ingrid_b","z_u","s255"],["bg_u"])              # u8 [1,1,30,30] (900)
N("Where",["filled_b","outcolor_u","bg_u"],["gm_u"])       # u8 [1,1,30,30] (900)

C("colors",np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
N("Equal",["gm_u","colors"],["output"])                    # bool [1,10,30,30] OUTPUT

g=helper.make_graph(nodes,"task301",
    [helper.make_tensor_value_info("input",F,[1,10,30,30])],
    [helper.make_tensor_value_info("output",B,[1,10,30,30])], inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=9)
onnx.checker.check_model(m)
onnx.save(m,"/tmp/b13_task301.onnx")
print("saved")
