"""Compact task005 (v3): full float16 pipeline on 21x21 extent, int8 output, single end Pad."""
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
S = 21
N = S*S
F16 = TensorProto.FLOAT16

KMAX = 4  # max number of period-4 copies that can fit on a 21 grid
R1 = 4    # single-shift kernel half-size
RA = 4*KMAX  # all-copies kernel half-size

def shift_kernel():
    # single shift by 4 in each direction
    k=2*R1+1
    w=np.zeros((8,1,k,k),dtype=np.float16)
    for d,(dr,dc) in enumerate(DIRS):
        w[d,0,R1-4*dr,R1-4*dc]=1.0
    return w

def all_kernel():
    # sum of shifts by 4*j for j=1..KMAX in each direction
    k=2*RA+1
    w=np.zeros((8,1,k,k),dtype=np.float16)
    for d,(dr,dc) in enumerate(DIRS):
        for j in range(1,KMAX+1):
            w[d,0,RA-4*j*dr,RA-4*j*dc]=1.0
    return w

def build():
    padsA=[RA,RA,RA,RA]
    nodes=[
        # cast input to int8 (9000, half of fp16), crop color channels to SxS in int8, cast to f16
        helper.make_node("Cast",["input"],["in8"],to=TensorProto.INT8),        # [1,10,30,30] int8 (9000)
        helper.make_node("Slice",["in8","f30_st","f30_en","f30_ax"],["x19i"]), # [1,9,S,S] int8 (3969)
        helper.make_node("Cast",["x19i"],["x19"],to=F16),                      # [1,9,S,S] f16
        # extent (1 on every real grid cell): bg channel + any color (one-hot, disjoint)
        helper.make_node("Slice",["in8","bg_st","bg_en","f30_ax"],["bg0i"]),   # [1,1,S,S] bg channel int8
        helper.make_node("Cast",["bg0i"],["bg0"],to=F16),                      # [1,1,S,S] f16
        helper.make_node("ReduceSum",["x19","axes1"],["occ"],keepdims=1),      # occupancy [1,1,S,S]
        helper.make_node("Add",["bg0","occ"],["gridS"]),                       # extent [1,1,S,S] f16
        # template selection
        helper.make_node("AveragePool",["x19"],["box"],kernel_shape=[3,3]),
        helper.make_node("ReduceMax",["box"],["wscore"],axes=[2,3],keepdims=1),
        helper.make_node("ReduceMax",["wscore"],["wmax"],axes=[1],keepdims=1),
        helper.make_node("Sub",["wscore","wmax"],["wdiff"]),
        helper.make_node("Sign",["wdiff"],["wsign"]),
        helper.make_node("Add",["wsign","one"],["tsel"]),                  # [1,9,1,1] master selector
        # flatten colors once; reuse for tmpl and the direction Gemm
        helper.make_node("Reshape",["x19","shp_k"],["kflat"]),             # [9,N]
        helper.make_node("Reshape",["tsel","shp_t"],["tselr"]),           # [1,9]
        helper.make_node("MatMul",["tselr","kflat"],["tmplf"]),           # [1,N] master plane flat
        helper.make_node("Reshape",["tmplf","shp_m"],["tmpl"]),           # [1,1,S,S]
        # all period-4 copies (k=1..KMAX) per direction; first copy (k=1) overlaps the marker
        helper.make_node("Conv",["tmpl","wall"],["tsum"],pads=padsA),        # [1,8,S,S]
        helper.make_node("Reshape",["tsum","shp_s"],["tsumflat"]),          # [8,N]
        # direction activation per color: ov[9,8] = kflat[9,N] @ tsumflat[8,N]^T
        helper.make_node("Gemm",["kflat","tsumflat"],["ov"],transB=1),      # [9,8]
        helper.make_node("Sign",["ov"],["act"]),
        helper.make_node("MatMul",["act","tsumflat"],["stamp"]),            # [9,N] fp16
        # cast flat stamps to int8 first (cheaper), then reshape to spatial int8
        helper.make_node("Cast",["stamp"],["stamp8"],to=TensorProto.INT8),  # [9,N] int8
        helper.make_node("Reshape",["stamp8","shp_y"],["sp8"]),            # [1,9,S,S] int8
        helper.make_node("Cast",["tsel"],["tsel8"],to=TensorProto.INT8),   # [1,9,1,1] int8 selector
        helper.make_node("Mul",["x19i","tsel8"],["xt8"]),                  # master glyph in its channel, int8
        helper.make_node("Max",["sp8","xt8"],["c19"]),                     # union (int8)
        # background = in-grid AND uncolored (gridSi - coloredSi)
        helper.make_node("Cast",["gridS"],["gridSi"],to=TensorProto.INT8), # extent int8
        helper.make_node("ReduceMax",["c19"],["coloredSi"],axes=[1],keepdims=1),  # int8
        helper.make_node("Sub",["gridSi","coloredSi"],["bgSi"]),           # 1 only where in-grid uncolored
        helper.make_node("Concat",["bgSi","c19"],["y10i"],axis=1),         # [1,10,S,S] int8
        helper.make_node("Pad",["y10i","pad10"],["output"]),
    ]
    inits=[
        numpy_helper.from_array(np.array([1,0,0],dtype=np.int64),"f30_st"),
        numpy_helper.from_array(np.array([10,S,S],dtype=np.int64),"f30_en"),
        numpy_helper.from_array(np.array([1,2,3],dtype=np.int64),"f30_ax"),
        numpy_helper.from_array(np.array([0,0,0],dtype=np.int64),"bg_st"),
        numpy_helper.from_array(np.array([1,S,S],dtype=np.int64),"bg_en"),
        numpy_helper.from_array(np.array([1.0],dtype=np.float16),"one"),
        numpy_helper.from_array(np.array([1],dtype=np.int64),"axes1"),
        numpy_helper.from_array(all_kernel(),"wall"),
        numpy_helper.from_array(np.array([9,N],dtype=np.int64),"shp_k"),
        numpy_helper.from_array(np.array([1,9],dtype=np.int64),"shp_t"),
        numpy_helper.from_array(np.array([1,1,S,S],dtype=np.int64),"shp_m"),
        numpy_helper.from_array(np.array([8,N],dtype=np.int64),"shp_s"),
        numpy_helper.from_array(np.array([1,9,S,S],dtype=np.int64),"shp_y"),
        numpy_helper.from_array(np.array([0,0,0,0, 0,0,30-S,30-S],dtype=np.int64),"pad10"),
    ]
    shp_in=[1,10,30,30]; shp_out=[1,10,30,30]
    g=helper.make_graph(nodes,"t005",
        [helper.make_tensor_value_info("input",TensorProto.FLOAT,shp_in)],
        [helper.make_tensor_value_info("output",TensorProto.INT8,shp_out)],inits)
    m=helper.make_model(g,ir_version=10,opset_imports=[helper.make_opsetid("",14)])
    onnx.checker.check_model(m)
    return m

if __name__=="__main__":
    onnx.save(build(),"/tmp/wf_task005.onnx"); print("saved")
