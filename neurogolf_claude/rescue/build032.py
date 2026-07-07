"""Non-iterative gravity (CumSum+ScatterElements), f16 working tensors to beat 16.867."""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
N = 6

def f16(x): return np.asarray(x, dtype=np.float16)
def C(name, arr): return helper.make_node("Constant", [], [name], value=nh.from_array(np.asarray(arr), name+"_v"))

def build(path="/tmp/p1f_task032.onnx"):
    nodes=[]; inits=[]
    # crop input to [1,10,6,6]
    inits.append(nh.from_array(np.array([0,0],dtype=np.int64),"loc_st"))
    inits.append(nh.from_array(np.array([N,N],dtype=np.int64),"loc_en"))
    inits.append(nh.from_array(np.array([2,3],dtype=np.int64),"axes_hw"))
    nodes.append(helper.make_node("Slice",["input","loc_st","loc_en","axes_hw"],["inc"]))  # [1,10,6,6] f32

    # encoded = color+1 (out-grid -> 0). conv weight [1..10] f32 -> output f32 [1,1,6,6]
    inits.append(nh.from_array(np.arange(1,11,dtype=np.float32).reshape(1,10,1,1),"conv_w"))
    nodes.append(helper.make_node("Conv",["inc","conv_w"],["encoded_f"],kernel_shape=[1,1],pads=[0,0,0,0]))
    # cast encoded to f16 for cheap working tensors
    nodes.append(helper.make_node("Cast",["encoded_f"],["encoded"],to=F16))  # [1,1,6,6] f16

    # in_grid = Sign(encoded)  (1 in-grid, 0 out-grid) f16
    nodes.append(helper.make_node("Sign",["encoded"],["in_grid"]))           # f16
    # H = sqrt(sum in_grid) over h,w
    nodes.append(helper.make_node("ReduceSum",["in_grid","axes_hw"],["n_sq"],keepdims=1))  # [1,1,1,1] f16
    nodes.append(helper.make_node("Sqrt",["n_sq"],["H_f"]))
    nodes.append(C("one_h", f16([[[[1.0]]]])))
    nodes.append(helper.make_node("Sub",["H_f","one_h"],["Hm1"]))            # H-1
    # nonzero (object) mask = encoded > 1
    nodes.append(helper.make_node("Greater",["encoded","one_h"],["cond_nz_b"]))   # bool [1,1,6,6]
    nodes.append(helper.make_node("Cast",["cond_nz_b"],["nz"],to=F16))       # f16 0/1
    # K per column = sum over rows (axis 2)
    inits.append(nh.from_array(np.array([2],dtype=np.int64),"axes_2"))
    nodes.append(helper.make_node("ReduceSum",["nz","axes_2"],["K_col"],keepdims=1))  # [1,1,1,6] f16
    nodes.append(helper.make_node("Sub",["Hm1","K_col"],["HmK"]))            # [1,1,1,6]
    # cum_incl = lower-tri (inclusive) MatMul over rows (replaces CumSum, f16-friendly)
    Li = np.tril(np.ones((N,N),dtype=np.float16), k=0)
    inits.append(nh.from_array(Li,"Li"))
    nodes.append(helper.make_node("MatMul",["Li","nz"],["cum_incl"]))        # [1,1,6,6] objects at-or-above
    # object target row = (H-1-K) + cum_incl
    nodes.append(helper.make_node("Add",["HmK","cum_incl"],["tgt_nz"]))
    # background target row = row_idx - cum_masked ; cum_masked = cum_incl * in_grid
    inits.append(nh.from_array(np.arange(N,dtype=np.float16).reshape(1,1,N,1),"row_idx"))
    nodes.append(helper.make_node("Mul",["cum_incl","in_grid"],["cum_masked"]))
    nodes.append(helper.make_node("Sub",["row_idx","cum_masked"],["tgt_bg"]))
    # target = where(object, tgt_nz, tgt_bg)
    nodes.append(helper.make_node("Where",["cond_nz_b","tgt_nz","tgt_bg"],["tgt_f"]))
    nodes.append(helper.make_node("Cast",["tgt_f"],["tgt_i"],to=TensorProto.INT32))  # indices (int32)
    # scatter encoded into zero grid along rows
    inits.append(nh.from_array(np.zeros((1,1,N,N),dtype=np.float16),"zero_grid"))
    nodes.append(helper.make_node("ScatterElements",["zero_grid","tgt_i","encoded"],["out_enc"],axis=2))
    # decode one-hot: Equal(out_enc, conv_w_h) ; need same dtype -> compare f16 vs f16 colors+1
    inits.append(nh.from_array(np.arange(1,11,dtype=np.float16).reshape(1,10,1,1),"colorsp1"))
    # Equal on f16? opset: Equal supports float in opset 11+. We'll use opset 13.
    nodes.append(helper.make_node("Equal",["out_enc","colorsp1"],["decoded_b"]))  # [1,10,6,6] bool
    # pad bool directly to 30x30 -> output (bool)
    inits.append(nh.from_array(np.array([0,0,0,0,0,0,30-N,30-N],dtype=np.int64),"pad30"))
    inits.append(nh.from_array(np.array(False),"padv"))
    nodes.append(helper.make_node("Pad",["decoded_b","pad30","padv"],["output"],mode="constant"))

    inp = helper.make_tensor_value_info("input", F32, [1,10,30,30])
    out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1,10,30,30])
    graph = helper.make_graph(nodes,"grav032",[inp],[out],inits)
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("",13)])
    onnx.save(model, path)
    return path

if __name__=="__main__":
    print("saved", build())
