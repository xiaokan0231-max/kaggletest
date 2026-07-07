import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

nodes=[]; inits=[]; _names=set()
def C(name, arr):
    if name in _names: return name
    inits.append(nh.from_array(np.asarray(arr), name)); _names.add(name); return name
def N(op, ins, outs, **kw):
    olist=[outs] if isinstance(outs,str) else list(outs)
    nodes.append(helper.make_node(op, ins, olist, name=olist[0], **kw)); return outs

GS=3  # all grids are 3x3 in the top-left of the 30x30 canvas

# ---- crop the one-hot input to the 3x3 region FIRST (tiny: [1,10,3,3] = 360B) ----
C('cst',np.array([0,0],dtype=np.int64))
C('cen',np.array([GS,GS],dtype=np.int64))
C('ax23',np.array([2,3],dtype=np.int64))
N('Slice',['input','cst','cen','ax23'],'in3')            # [1,10,3,3] f32

# collapse to single color grid [1,1,3,3]
wcol=np.zeros((1,10,1,1),np.float32); wcol[0,:,0,0]=np.arange(10,dtype=np.float32)
C('Wcol',wcol)
N('Conv',['in3','Wcol'],'g3f')                           # [1,1,3,3] f32
N('Cast',['g3f'],'g3', to=TensorProto.UINT8)             # [1,1,3,3] uint8

# ---- per-row uniformity: row is uniform iff max(row)==min(row) ----
C('ax3',np.array([3],dtype=np.int64))
N('ReduceMax',['g3','ax3'],'rmax',keepdims=1)            # [1,1,3,1] uint8
N('ReduceMin',['g3','ax3'],'rmin',keepdims=1)            # [1,1,3,1] uint8
N('Equal',['rmax','rmin'],'uni')                         # [1,1,3,1] bool per-row uniform

# ---- color grid: uniform -> 5 else -> 0, broadcast across the 3 columns ----
C('five',np.array([[[[5]]]],dtype=np.uint8))
C('zero',np.array([[[[0]]]],dtype=np.uint8))
N('Where',['uni','five','zero'],'gm3a')                  # [1,1,3,1] uint8 (numeric Where is supported)
C('z33',np.zeros((1,1,3,3),dtype=np.uint8))
N('Add',['gm3a','z33'],'gm3')                            # [1,1,3,3] uint8

# ---- one-hot in the 3x3 region via Equal, then Pad with False -> full-canvas OUTPUT ----
colors=np.arange(10,dtype=np.uint8).reshape(1,10,1,1)
C('colors',colors)
N('Equal',['gm3','colors'],'oh3')                        # [1,10,3,3] bool
C('padfull',np.array([0,0,0,0, 0,0,30-GS,30-GS],dtype=np.int64))
C('falsec',np.array(False,dtype=bool))
N('Pad',['oh3','padfull','falsec'],'output',mode='constant')  # [1,10,30,30] bool = graph output

graph=helper.make_graph(nodes,'task052',
    [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])], inits)
model=helper.make_model(graph, opset_imports=[helper.make_operatorsetid("",18)])
model.ir_version=9
onnx.checker.check_model(model, full_check=True)
inf=onnx.shape_inference.infer_shapes(model, strict_mode=True)
bad=sum(1 for v in list(inf.graph.value_info)+list(inf.graph.output)
        for dd in v.type.tensor_type.shape.dim
        if dd.HasField('dim_param') or not dd.HasField('dim_value') or dd.dim_value<=0)
print('non-concrete shapes:', bad)
io={'input','output'}
tot=0
tm={t.name:t for t in list(inf.graph.value_info)+list(inf.graph.output)}
for n,t in tm.items():
    if n in io: continue
    tt=t.type.tensor_type
    ne=1
    for dd in tt.shape.dim: ne*=dd.dim_value
    isz=np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
    tot+=ne*isz
print('approx interior bytes:', tot)
onnx.save(model,'/tmp/p1f_task052.onnx')
print('saved. nodes',len(nodes),'inits',len(inits))
