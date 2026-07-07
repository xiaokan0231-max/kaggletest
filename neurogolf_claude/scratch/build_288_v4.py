"""task288 compact v4: maximally fused (verified 267/267 numpy).
Drop ray&v (final v-blank handles out-of-grid). apex via 2 shifts.
"""
import numpy as np, onnx
from onnx import TensorProto as T, helper, numpy_helper
def cf(n,v): return numpy_helper.from_array(np.asarray(v,np.float32),n)
def c16(n,v): return numpy_helper.from_array(np.asarray(v,np.float16),n)
def cu(n,v): return numpy_helper.from_array(np.asarray(v,np.uint8),n)
def ci(n,v): return numpy_helper.from_array(np.asarray(v,np.int64),n)
def cb(n,v): return numpy_helper.from_array(np.asarray(v,np.bool_),n)
nodes,inits=[],[]
def add(op,i,o,**k): nodes.append(helper.make_node(op,i,o,**k))
sc=[0]
def shift(src,dr,dc,boolt=True):
    pt=dr if dr>0 else 0; pb=-dr if dr<0 else 0
    pl=dc if dc>0 else 0; pr=-dc if dc<0 else 0
    nm=f"s{sc[0]}"; sc[0]+=1
    inits.append(ci(nm+"p",[0,0,pt,pl,0,0,pb,pr]))
    inits.append(cb(nm+"cv",False) if boolt else cu(nm+"cv",0))
    add("Pad",[src,nm+"p",nm+"cv"],[nm+"pad"])
    inits.append(ci(nm+"ss",[pt-dr,pl-dc])); inits.append(ci(nm+"se",[pt-dr+30,pl-dc+30])); inits.append(ci(nm+"sa",[2,3]))
    add("Slice",[nm+"pad",nm+"ss",nm+"se",nm+"sa"],[nm+"o"])
    return nm+"o"

inits.append(cf("Wcol",np.arange(10,dtype=np.float32).reshape(1,10,1,1)))
add("Conv",["input","Wcol"],["gf"])
add("Cast",["gf"],["g"],to=T.UINT8)
inits.append(ci("ax1",[1])); add("ReduceSum",["input","ax1"],["vsum"],keepdims=1)
inits.append(cf("zero",0.0)); add("Greater",["vsum","zero"],["v"])
inits.append(cu("u0",0)); add("Greater",["g","u0"],["gpos"])
# apex = gpos & up(gpos) & ~up2(gpos)
up1=shift("gpos",-1,0); up2=shift("gpos",-2,0)
add("Not",[up2],["nup2"]); add("And",["gpos",up1],["a1"]); add("And",["a1","nup2"],["apex"])
# corners
apr=shift("apex",0,1); add("Not",[apr],["napr"]); add("And",["apex","napr"],["Lcorner"])
apl=shift("apex",0,-1); add("Not",[apl],["napl"]); add("And",["apex","napl"],["Rcorner"])
# diagonal Conv rays
Kl=np.zeros((1,1,30,30),np.float16); Kr=np.zeros((1,1,30,30),np.float16)
for k in range(30): Kl[0,0,k,k]=1.0; Kr[0,0,k,29-k]=1.0
inits.append(c16("Kl",Kl)); inits.append(c16("Kr",Kr))
add("Cast",["Lcorner"],["Lf"],to=T.FLOAT16); add("Cast",["Rcorner"],["Rf"],to=T.FLOAT16)
add("Conv",["Lf","Kl"],["Lray"],pads=[0,0,29,29])
add("Conv",["Rf","Kr"],["Rray"],pads=[0,29,29,0])
add("Add",["Lray","Rray"],["raysum"])
inits.append(c16("h",0.5)); add("Greater",["raysum","h"],["rayb"])
# ray = rayb & ~gpos  (drop &v: final v-blank handles out-of-grid)
add("Not",["gpos"],["gbg"]); add("And",["rayb","gbg"],["ray"])
# C
cband=shift("apex",1,0); inits.append(cf("f0",0.0)); add("Where",[cband,"gf","f0"],["gc"])
add("ReduceMax",["gc"],["Cf"],axes=[2,3],keepdims=1); add("Cast",["Cf"],["C"],to=T.UINT8)
# gm + output
add("Where",["ray","C","g"],["gm0"]); inits.append(cu("sent",255)); add("Where",["v","gm0","sent"],["gm"])
inits.append(cu("colors",np.arange(10,dtype=np.uint8).reshape(1,10,1,1)))
add("Equal",["gm","colors"],["output"])
xin=helper.make_tensor_value_info("input",T.FLOAT,[1,10,30,30])
yo =helper.make_tensor_value_info("output",T.BOOL,[1,10,30,30])
gr=helper.make_graph(nodes,"t288",[xin],[yo],inits)
m=helper.make_model(gr,opset_imports=[helper.make_opsetid("",13)]); m.ir_version=9
import sys; p=sys.argv[1] if len(sys.argv)>1 else "/tmp/b10_task288.onnx"
onnx.save(m,p); print("saved",p,"nodes",len(nodes))
