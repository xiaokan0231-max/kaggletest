"""task288 compact v2: Conv diagonal-kernel rays, f16/bool/u8 interior.

Pipeline (verified 267/267 in numpy):
  g(u8) color grid, v(bool) validity from ReduceSum(input)>0
  bottomr = v & ~up(v)
  apex    = (g>0) & up(bottomr) & v           # B-block at row rb-1
  Lcorner = apex & ~right(apex);  Rcorner = apex & ~left(apex)
  L = Conv(Lcorner_f16, Kdiag_ul);  R = Conv(Rcorner_f16, Kdiag_ur)   # f16 30x30
  ray = (L+R>0) & v & (g==0)                  # paint only background
  C   = ReduceMax(gf where cband) ; cband = down(apex)
  gm  = where(ray, C, g); gm = where(v, gm, 255)
  output = Equal(gm, colors)                  # excluded
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

# color grid
inits.append(cf("Wcol",np.arange(10,dtype=np.float32).reshape(1,10,1,1)))
add("Conv",["input","Wcol"],["gf"])             # f32 30x30
add("Cast",["gf"],["g"],to=T.UINT8)             # u8 color grid
# validity
inits.append(ci("ax1",[1]))
add("ReduceSum",["input","ax1"],["vsum"],keepdims=1)  # f32 30x30
inits.append(cf("zero",0.0))
add("Greater",["vsum","zero"],["v"])            # bool

sc=[0]
def shift(src,dr,dc,boolt=True):
    pt=dr if dr>0 else 0; pb=-dr if dr<0 else 0
    pl=dc if dc>0 else 0; pr=-dc if dc<0 else 0
    nm=f"s{sc[0]}"; sc[0]+=1
    inits.append(ci(nm+"p",[0,0,pt,pl,0,0,pb,pr]))
    inits.append(cb(nm+"cv",False) if boolt else cu(nm+"cv",0))
    add("Pad",[src,nm+"p",nm+"cv"],[nm+"pad"])
    rstart=pt-dr; cstart=pl-dc
    inits.append(ci(nm+"ss",[rstart,cstart])); inits.append(ci(nm+"se",[rstart+30,cstart+30])); inits.append(ci(nm+"sa",[2,3]))
    add("Slice",[nm+"pad",nm+"ss",nm+"se",nm+"sa"],[nm+"o"])
    return nm+"o"

vup=shift("v",-1,0); add("Not",[vup],["nvup"]); add("And",["v","nvup"],["bottomr"])
inits.append(cu("u0",0)); add("Greater",["g","u0"],["gpos"])
brup=shift("bottomr",-1,0); add("And",["gpos",brup],["t1"]); add("And",["t1","v"],["apex"])
apr=shift("apex",0,1); add("Not",[apr],["napr"]); add("And",["apex","napr"],["Lcorner"])
apl=shift("apex",0,-1); add("Not",[apl],["napl"]); add("And",["apex","napl"],["Rcorner"])

# Conv diagonal kernels (f16). kernel [1,1,30,30]; cross-correlation.
# left ray reads corner at (r+k,c+k): need K[k,k]=1, with pads so output is 30x30
# onnx Conv with kernel KxK and pads p produces out = in_size - K +1 + 2p. For same (30): p=(K-1)/2? K=30 even -> use pads to get 30.
# Use K=30, pads [29,29,0,0]? Let's compute: in=30,K=30,no stride. out = 30-30+1 + padtop+padbot = 1 + pt+pb. Need 30 -> pt+pb=29.
# For diagonal reading (r+k,c+k): conv out[r,c]=sum_{i,j} in[r+i-pt, c+j-pl]*W[i,j]. We want term in[r+k,c+k] -> i-pt=k,j-pl=k.
# Set pt=pl=0 -> i=k,j=k, but then out rows only valid for r where r+i in range -> need pad bottom/right.
# Use pt=0,pb=29,pl=0,pr=29 -> out 30x30. out[r,c]=sum W[i,j]*in_padded[r+i,c+j], in_padded has orig at [0:30,0:30].
# in[r+k,c+k] corresponds i=k,j=k. So W[k,k]=1 for left ray. 
Kl=np.zeros((1,1,30,30),np.float16); Kr=np.zeros((1,1,30,30),np.float16)
for k in range(30):
    Kl[0,0,k,k]=1.0
# right ray reads corner at (r+k,c-k): i=k, j such that c+j-pl=c-k -> with pl=29: j-29=-k -> j=29-k
for k in range(30):
    Kr[0,0,k,29-k]=1.0
inits.append(c16("Kl",Kl)); inits.append(c16("Kr",Kr))
add("Cast",["Lcorner"],["Lf"],to=T.FLOAT16); add("Cast",["Rcorner"],["Rf"],to=T.FLOAT16)
add("Conv",["Lf","Kl"],["Lray"],pads=[0,0,29,29])      # pt0 pl0 pb29 pr29: reads (r+k,c+k)
add("Conv",["Rf","Kr"],["Rray"],pads=[0,29,29,0])      # pt0 pl29 pb29 pr0: reads (r+k,c-k)
add("Add",["Lray","Rray"],["raysum"])
inits.append(c16("h",0.5)); add("Greater",["raysum","h"],["rayb"])   # bool
# ray & v & (g==0)
add("And",["rayb","v"],["r2"])
add("Equal",["g","u0"],["gbg"])     # g==0 -> background bool
add("And",["r2","gbg"],["ray"])

# inner color C
cband=shift("apex",1,0)
inits.append(cf("f0",0.0)); add("Where",[cband,"gf","f0"],["gc"])
add("ReduceMax",["gc"],["Cf"],axes=[2,3],keepdims=1); add("Cast",["Cf"],["C"],to=T.UINT8)

add("Where",["ray","C","g"],["gm0"])
inits.append(cu("sent",255)); add("Where",["v","gm0","sent"],["gm"])
inits.append(cu("colors",np.arange(10,dtype=np.uint8).reshape(1,10,1,1)))
add("Equal",["gm","colors"],["output"])

xin=helper.make_tensor_value_info("input",T.FLOAT,[1,10,30,30])
yo =helper.make_tensor_value_info("output",T.BOOL,[1,10,30,30])
g=helper.make_graph(nodes,"t288",[xin],[yo],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",13)]); m.ir_version=9
import sys; p=sys.argv[1] if len(sys.argv)>1 else "/tmp/b10_task288.onnx"
onnx.save(m,p); print("saved",p,"nodes",len(nodes))
