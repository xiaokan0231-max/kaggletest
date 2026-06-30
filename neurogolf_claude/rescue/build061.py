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

GS=18; PAD=9

# ---- color grid uint8 18x18 ----
wcol=np.zeros((1,10,1,1),np.float32); wcol[0,:,0,0]=np.arange(10,dtype=np.float32)
C('Wcol',wcol)
N('Conv',['input','Wcol'],'g_f32')
N('Cast',['g_f32'],'g_u30', to=TensorProto.UINT8)    # [1,1,30,30] uint8
C('cst',np.array([0,0],dtype=np.int64)); C('cen',np.array([GS,GS],dtype=np.int64)); C('ax23',np.array([2,3],dtype=np.int64))
N('Slice',['g_u30','cst','cen','ax23'],'g0')         # [1,1,18,18] uint8

# ---- P = max value (int64 scalar) ----
C('all_axes',np.array([0,1,2,3],dtype=np.int64))
N('ReduceMax',['g0','all_axes'], 'Pmaxu', keepdims=0)   # scalar uint8
N('Cast',['Pmaxu'],'P', to=TensorProto.INT64)           # scalar int64

# index tensors: base[18] + (PAD +/- P)
C('base', np.arange(GS,dtype=np.int64))                 # [18]
C('PADc', np.array(PAD,dtype=np.int64))
N('Add',['PADc','P'],'pp')      # PAD+P scalar
N('Sub',['PADc','P'],'pm')      # PAD-P scalar
N('Add',['base','pp'],'idx_pos')   # [18] base+PAD+P
N('Add',['base','pm'],'idx_neg')   # [18] base+PAD-P

# ---- Gather-based shift on padded buffers (uint8) ----
C('padr',np.array([0,0,PAD,0,0,0,PAD,0],dtype=np.int64))   # rows -> [1,1,36,18]
C('padc',np.array([0,0,0,PAD,0,0,0,PAD],dtype=np.int64))   # cols -> [1,1,18,36]

def shift_dirs(cur, it, dirs):
    outs=[cur]
    if 'v' in dirs:
        N('Pad',[cur,'padr'], f'pr_{it}', mode='constant')
        N('Gather',[f'pr_{it}','idx_pos'], f'sd_{it}', axis=2); outs.append(f'sd_{it}')   # down +P
        N('Gather',[f'pr_{it}','idx_neg'], f'su_{it}', axis=2); outs.append(f'su_{it}')   # up -P
    if 'h' in dirs:
        N('Pad',[cur,'padc'], f'pc_{it}', mode='constant')
        N('Gather',[f'pc_{it}','idx_pos'], f'sr_{it}', axis=3); outs.append(f'sr_{it}')   # right +P
        N('Gather',[f'pc_{it}','idx_neg'], f'sl_{it}', axis=3); outs.append(f'sl_{it}')   # left -P
    N('Max',outs, f'mx{it}')
    return f'mx{it}'

cur=shift_dirs('g0', 0, 'hv')   # iter0 4-dir
cur=shift_dirs(cur, 1, 'h')     # iter1 horizontal
filled=cur

C('padto30',np.array([0,0,0,0,0,0,30-GS,30-GS],dtype=np.int64))
N('Pad',[filled,'padto30'],'gm30',mode='constant')
colors=np.arange(10,dtype=np.uint8).reshape(1,10,1,1); colors[0,0,0,0]=255
C('colors',colors)
N('Equal',['gm30','colors'],'output')

graph=helper.make_graph(nodes,'task061',
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
onnx.save(model,'/tmp/p1b_task061.onnx')
print('saved v11. nodes',len(nodes),'inits',len(inits))
