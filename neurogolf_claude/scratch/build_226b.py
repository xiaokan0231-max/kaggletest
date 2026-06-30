import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

nodes=[]; inits=[]
def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name)); return name

# 1) collapse 10ch -> color grid g30f [1,1,30,30] f32 (3600B, irreducible)
C('Wcol', np.arange(10,dtype=np.float32).reshape(1,10,1,1))
nodes.append(helper.make_node('Conv',['input','Wcol'],['g30f'],kernel_shape=[1,1]))

# crop to 10x10 then cast uint8 immediately (interior g f32 10x10 we avoid by casting -> but slice still f32)
C('s0', np.array([0,0],dtype=np.int64)); C('e0', np.array([10,10],dtype=np.int64)); C('ax', np.array([2,3],dtype=np.int64))
nodes.append(helper.make_node('Slice',['g30f','s0','e0','ax'],['g10f']))  # f32 [1,1,10,10] 400B
nodes.append(helper.make_node('Cast',['g10f'],['g'],to=TensorProto.UINT8))  # u8 100B

# gray mask uint8
C('five', np.array(5,dtype=np.uint8))
nodes.append(helper.make_node('Equal',['g','five'],['grayb']))            # bool 100B
nodes.append(helper.make_node('Cast',['grayb'],['grayf'],to=TensorProto.FLOAT))  # f32 400B (needed for Conv)

# per-row/col gray counts via Conv
C('Wrow', np.ones((1,1,1,10),dtype=np.float32))
C('Wcolk', np.ones((1,1,10,1),dtype=np.float32))
nodes.append(helper.make_node('Conv',['grayf','Wrow'],['rowcnt']))   # [1,1,10,1] f32
nodes.append(helper.make_node('Conv',['grayf','Wcolk'],['colcnt']))  # [1,1,1,10] f32
C('ten', np.array(10,dtype=np.float32))
nodes.append(helper.make_node('Equal',['rowcnt','ten'],['growb']))   # [1,1,10,1] bool
nodes.append(helper.make_node('Equal',['colcnt','ten'],['gcolb']))
nodes.append(helper.make_node('Cast',['growb'],['growf'],to=TensorProto.FLOAT))
nodes.append(helper.make_node('Cast',['gcolb'],['gcolf'],to=TensorProto.FLOAT))

# Rg=#grayrows, Cg=#graycols ; Rhalf=Rg/2
nodes.append(helper.make_node('ReduceSum',['growf'],['Rg'],keepdims=0))
nodes.append(helper.make_node('ReduceSum',['gcolf'],['Cg'],keepdims=0))
C('two', np.array(2.0,dtype=np.float32))
nodes.append(helper.make_node('Div',['Rg','two'],['Rhalf']))
nodes.append(helper.make_node('Div',['Cg','two'],['Chalf']))

# exclusive prefix sum band index
C('axisH', np.array(2,dtype=np.int64)); C('axisW', np.array(3,dtype=np.int64))
nodes.append(helper.make_node('CumSum',['growf','axisH'],['rincl']))
nodes.append(helper.make_node('Sub',['rincl','growf'],['ridx']))
nodes.append(helper.make_node('CumSum',['gcolf','axisW'],['cincl']))
nodes.append(helper.make_node('Sub',['cincl','gcolf'],['cidx']))

# background-0 mask
C('zero', np.array(0,dtype=np.uint8))
nodes.append(helper.make_node('Equal',['g','zero'],['bg0b']))

C('f0', np.array(0.0,dtype=np.float32))
def band_mask(rowtgt, coltgt, p):
    nodes.append(helper.make_node('Equal',['ridx',rowtgt],[p+'rm']))   # [1,1,10,1] bool
    nodes.append(helper.make_node('Equal',['cidx',coltgt],[p+'cm']))   # [1,1,1,10] bool
    nodes.append(helper.make_node('And',[p+'rm',p+'cm'],[p+'rc']))     # [1,1,10,10]
    nodes.append(helper.make_node('And',[p+'rc','bg0b'],[p+'msk']))
    return p+'msk'
m1=band_mask('f0','f0','c1_')
m2=band_mask('Rhalf','Chalf','c2_')
m3=band_mask('Rg','Cg','c3_')

C('col1', np.array(1,dtype=np.uint8)); C('col2', np.array(2,dtype=np.uint8)); C('col3', np.array(3,dtype=np.uint8))
nodes.append(helper.make_node('Where',[m1,'col1','g'],['gm1']))
nodes.append(helper.make_node('Where',[m2,'col2','gm1'],['gm2']))
nodes.append(helper.make_node('Where',[m3,'col3','gm2'],['gm']))   # [1,1,10,10] u8

# pad to 30x30 with sentinel 255
C('pads', np.array([0,0,0,0, 0,0,20,20],dtype=np.int64)); C('sent', np.array(255,dtype=np.uint8))
nodes.append(helper.make_node('Pad',['gm','pads','sent'],['gm30'],mode='constant'))  # [1,1,30,30] u8 900B
C('colors', np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
nodes.append(helper.make_node('Equal',['gm30','colors'],['output']))

graph=helper.make_graph(nodes,'task226',
    [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],inits)
mdl=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]); mdl.ir_version=8
onnx.checker.check_model(mdl); onnx.save(mdl,'/tmp/p1e_task226.onnx')
print('saved nodes',len(nodes))
