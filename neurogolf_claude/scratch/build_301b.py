"""task301 compact v2: uint8/bool everywhere on [1,1,30,30] tensors."""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

nodes=[]; inits=[]
def C(name,arr): inits.append(numpy_helper.from_array(np.asarray(arr),name))
def N(op,i,o,**k): nodes.append(helper.make_node(op,i,o,**k))

# color grid g (f32) via Conv
C("w_color", np.arange(10,dtype=np.float32).reshape(1,10,1,1))
N("Conv",["input","w_color"],["g"])                  # [1,1,30,30] f32 (interior, 3600B)
# uint8 color grid
N("Cast",["g"],["gu"],to=TensorProto.UINT8)          # [1,1,30,30] u8 900B
# mask bool
C("zero_u",np.array(0,np.uint8))
N("Greater",["gu","zero_u"],["mask_b"])              # bool 900B
N("Cast",["mask_b"],["mask_u"],to=TensorProto.UINT8) # u8 900B (for ReduceSum)

C("ax3",np.array([3],np.int64))
N("ReduceSum",["mask_u","ax3"],["rowlen_u"],keepdims=1)   # [1,1,30,1] u8 (int32? cast) -> actually ReduceSum on u8 -> u8
N("ReduceMax",["gu","ax3"],["rowcolor_u"],keepdims=1)     # [1,1,30,1] u8

# anyrow = rowlen>0
N("Greater",["rowlen_u","zero_u"],["anyrow_b"])           # [1,1,30,1] bool
N("Cast",["anyrow_b"],["anyrow_u"],to=TensorProto.UINT8)

C("ax2",np.array([2],np.int64))
N("ReduceSum",["anyrow_u","ax2"],["Nn_u"],keepdims=1)     # [1,1,1,1] u8 = N
N("ReduceMax",["mask_u","ax2"],["anycol_u"],keepdims=1)   # [1,1,1,30] u8

# H = max(r*anyrow)+1
C("ridx_col",np.arange(30,dtype=np.uint8).reshape(1,1,30,1))
N("Mul",["ridx_col","anyrow_u"],["r_anyrow"])             # u8 [1,1,30,1]
N("ReduceMax",["r_anyrow","ax2"],["maxr"],keepdims=1)     # u8 [1,1,1,1]
C("one_u",np.array(1,np.uint8))
N("Add",["maxr","one_u"],["H_u"])                         # u8
# W = max(c*anycol)+1
C("cidx_row",np.arange(30,dtype=np.uint8).reshape(1,1,1,30))
N("Mul",["cidx_row","anycol_u"],["c_anycol"])
N("ReduceMax",["c_anycol","ax3"],["maxc"],keepdims=1)
N("Add",["maxc","one_u"],["W_u"])

# targetrow = (H-N)+(rowlen-1) = H - N + rowlen - 1.
# Do in int16-safe via uint8: H>=N, rowlen>=1 so all >=0. Compute (H-N) then +rowlen then -1.
N("Sub",["H_u","Nn_u"],["HmN"])                           # u8
N("Add",["HmN","rowlen_u"],["tr_a"])                      # u8 [1,1,30,1]
N("Sub",["tr_a","one_u"],["targetrow_u"])                 # u8 [1,1,30,1]

C("rout_idx",np.arange(30,dtype=np.uint8).reshape(1,1,1,30))
N("Equal",["targetrow_u","rout_idx"],["eq_b"])            # bool [1,1,30,30] 900B
N("And",["eq_b","anyrow_b"],["eqm_b"])                    # bool [1,1,30,30] 900B (anyrow_b broadcasts axis2)
N("Cast",["eqm_b"],["eqm_u"],to=TensorProto.UINT8)        # u8 900B
N("Mul",["eqm_u","rowcolor_u"],["wc_u"])                  # u8 [1,1,30,30] 900B
N("ReduceSum",["wc_u","ax2"],["outcolor_u"],keepdims=1)   # u8 [1,1,1,30]

# T = W+H-1-N
N("Add",["W_u","H_u"],["WpH"])
N("Sub",["WpH","one_u"],["WpHm1"])
N("Sub",["WpHm1","Nn_u"],["T_u"])                         # u8
# rpc = r+c (max 58 fits u8)
C("ridx2",np.arange(30,dtype=np.uint8).reshape(1,1,30,1))
C("cidx2",np.arange(30,dtype=np.uint8).reshape(1,1,1,30))
N("Add",["ridx2","cidx2"],["rpc"])                        # u8 [1,1,30,30] 900B
N("GreaterOrEqual",["rpc","T_u"],["ge_b"])                # bool [1,1,30,30]
N("Less",["ridx2","H_u"],["rltH_b"])                      # bool [1,1,30,1]
N("Less",["cidx2","W_u"],["cltW_b"])                      # bool [1,1,1,30]
N("And",["ge_b","rltH_b"],["f1_b"])
N("And",["f1_b","cltW_b"],["filled_b"])                   # bool [1,1,30,30]
N("And",["rltH_b","cltW_b"],["ingrid_b"])                 # bool [1,1,30,30]

# outcolor (indexed by out-row on axis3) -> transpose to axis2
N("Transpose",["outcolor_u"],["outcolor_t"],perm=[0,1,3,2])  # u8 [1,1,30,1]
N("Cast",["filled_b"],["filled_u"],to=TensorProto.UINT8)
N("Mul",["filled_u","outcolor_t"],["gm_in"])              # u8 [1,1,30,30] 900B
C("sent255",np.array(255,np.uint8))
N("Where",["ingrid_b","gm_in","sent255"],["gm_u"])        # u8 [1,1,30,30] 900B

C("colors",np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
N("Equal",["gm_u","colors"],["output"])                   # bool [1,10,30,30] OUTPUT (excluded)

g=helper.make_graph(nodes,"task301",
    [helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info("output",TensorProto.BOOL,[1,10,30,30])], inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",18)],ir_version=9)
onnx.checker.check_model(m)
onnx.save(m,"/tmp/b13_task301.onnx")
print("saved")
