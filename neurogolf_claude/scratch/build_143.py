"""task143 - pad-once shift trick to cut value-grid tensors; channel isolation."""
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
F32=TensorProto.FLOAT; F16=TensorProto.FLOAT16; BOOL=TensorProto.BOOL; U8=TensorProto.UINT8
S=10
def cf(n,v): return numpy_helper.from_array(np.asarray(v,np.float32),n)
def ch(n,v): return numpy_helper.from_array(np.asarray(v,np.float16),n)
def ci(n,v): return numpy_helper.from_array(np.asarray(v,np.int64),n)
def cb(n,v): return numpy_helper.from_array(np.asarray(v,np.bool_),n)
def build(path):
    nodes,inits=[],[]
    n=lambda *a,**k: nodes.append(helper.make_node(*a,**k))
    inits.append(cf("Wcol", np.arange(10,dtype=np.float32).reshape(1,10,1,1)))
    n("Conv",["input","Wcol"],["g30"])
    inits += [ci("cs",[0,0]), ci("ce",[S,S]), ci("ca",[2,3])]
    n("Slice",["g30","cs","ce","ca"],["g10f"]); n("Cast",["g10f"],["g"],to=F16)
    inits.append(ch("z0",0.0))
    boxm=np.ones((1,1,10,10),np.float16); boxm[0,0,0:4,0:4]=0; inits.append(ch("boxm",boxm))
    n("Mul",["g","boxm"],["gw"])   # box-clear also removes color5 (only in box)
    n("Greater",["gw","z0"],["nzwb"]); n("Cast",["nzwb"],["nzw"],to=F16)
    inits.append(ch("W3",np.ones((1,1,3,3),np.float16)))
    n("Conv",["nzw","W3"],["bsum3"],pads=[0,0,2,2])
    inits += [ci("ks",[0,0]), ci("ke",[3,3]), ci("ka",[2,3])]
    n("Slice",["g","ks","ke","ka"],["kreg"]); n("Greater",["kreg","z0"],["kb"]); n("Cast",["kb"],["K"],to=F16)
    n("ReduceSum",["K"],["Ksum"],keepdims=1)
    # shift-Conv: svstack[k][y,x] = gw[y+di,x+dj] for the 9 offsets (static delta kernels)
    Wsh=np.zeros((9,1,3,3),np.float16)
    k=0
    for di in range(3):
        for dj in range(3):
            Wsh[k,0,di,dj]=1.0; k+=1
    inits.append(ch("Wsh",Wsh))
    n("Conv",["gw","Wsh"],["svstack"],pads=[0,0,2,2])   # [1,9,10,10]
    inits.append(ci("kshape",[1,9,1,1])); n("Reshape",["K","kshape"],["Kstack"])
    n("Mul",["svstack","Kstack"],["cvstack"])
    inits.append(ci("ax1c",[1]))
    n("ReduceSum",["cvstack","ax1c"],["csum"],keepdims=1)
    n("ReduceMax",["cvstack"],["cmax"],axes=[1],keepdims=1)
    csum="csum"; cmax="cmax"
    n("Mul",["Ksum",cmax],["kcmax"]); n("Equal",[csum,"kcmax"],["mono1"]); n("Greater",[cmax,"z0"],["mono2"])
    n("Equal",["bsum3","Ksum"],["okB"])
    n("And",["mono1","mono2"],["ok2"]); n("And",["okB","ok2"],["shape_ok"])
    v=cmax
    colors8=np.array([1,2,3,4,6,7,8,9],np.float16).reshape(1,8,1,1); inits.append(ch("colors",colors8))
    n("Equal",["gw","colors"],["Bb"]); n("Cast",["Bb"],["Bf"],to=F16)
    W5=np.ones((8,1,5,5),np.float16); inits.append(ch("W5",W5))
    n("Conv",["Bf","W5"],["bsum5"],group=8,pads=[2,2,2,2])
    n("Equal",["colors",v],["cveq"])
    inits.append(ch("BIG",9999.0))
    n("Where",["cveq","bsum5","BIG"],["picked"])        # [1,8,*]: v-channel keeps bsum5, else BIG
    n("ReduceMin",["picked"],["bsum5v"],axes=[1],keepdims=1)
    n("Equal",["bsum5v","Ksum"],["isob"])
    n("And",["shape_ok","isob"],["Mb"]); n("Cast",["Mb"],["M"],to=F16)
    n("MaxPool",["M"],["dil"],kernel_shape=[3,3],strides=[1,1],pads=[2,2,0,0])
    n("Mul",["dil","nzw"],["recf"]); n("Greater",["recf","z0"],["recb"])
    inits.append(ci("mpads",[0,0,0,0,0,0,30-S,30-S])); inits.append(cb("fp",False))
    n("Pad",["recb","mpads","fp"],["mask30"])
    e5=np.zeros((1,10,1,1),np.float32); e5[0,5]=1.0; inits.append(cf("e5",e5))
    n("Where",["mask30","e5","input"],["output"])
    x_in=helper.make_tensor_value_info("input",F32,[1,10,30,30])
    y=helper.make_tensor_value_info("output",F32,[1,10,30,30])
    graph=helper.make_graph(nodes,"t143",[x_in],[y],inits)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",13)]); model.ir_version=9
    onnx.save(model,path); print("saved",path,"nodes",len(nodes))
if __name__=="__main__":
    import sys; build(sys.argv[1] if len(sys.argv)>1 else "/tmp/p1b_task143.onnx")
