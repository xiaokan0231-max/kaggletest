import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

nodes=[]; inits=[]
def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name)); return name

# 1) Slice gray channel (5) at 10x10 directly from input -> gray f32 [1,1,10,10] (400B). Inputs only 0/5.
C('s5', np.array([5,0,0],dtype=np.int64))       # start: ch5, r0, c0
C('e5', np.array([6,10,10],dtype=np.int64))     # end:  ch6, r10, c10
C('a5', np.array([1,2,3],dtype=np.int64))       # axes: channel,H,W
nodes.append(helper.make_node('Slice',['input','s5','e5','a5'],['grayf']))  # f32 [1,1,10,10]
# grayb bool
C('half', np.array(0.5,dtype=np.float32))
nodes.append(helper.make_node('Greater',['grayf','half'],['grayb']))   # bool 100B

# 2) per-row/col full-gray counts via Conv on grayf
C('Wrow', np.ones((1,1,1,10),dtype=np.float32))
C('Wcolk', np.ones((1,1,10,1),dtype=np.float32))
nodes.append(helper.make_node('Conv',['grayf','Wrow'],['rowcnt']))
nodes.append(helper.make_node('Conv',['grayf','Wcolk'],['colcnt']))
C('ten', np.array(10,dtype=np.float32))
nodes.append(helper.make_node('Equal',['rowcnt','ten'],['growb']))   # [1,1,10,1]
nodes.append(helper.make_node('Equal',['colcnt','ten'],['gcolb']))
nodes.append(helper.make_node('Cast',['growb'],['growf'],to=TensorProto.FLOAT))
nodes.append(helper.make_node('Cast',['gcolb'],['gcolf'],to=TensorProto.FLOAT))

# 3) Rg,Cg,Rhalf,Chalf
nodes.append(helper.make_node('ReduceSum',['growf'],['Rg'],keepdims=0))
nodes.append(helper.make_node('ReduceSum',['gcolf'],['Cg'],keepdims=0))
C('two', np.array(2.0,dtype=np.float32))
nodes.append(helper.make_node('Div',['Rg','two'],['Rhalf']))
nodes.append(helper.make_node('Div',['Cg','two'],['Chalf']))

# 4) exclusive prefix band index
C('axisH', np.array(2,dtype=np.int64)); C('axisW', np.array(3,dtype=np.int64))
nodes.append(helper.make_node('CumSum',['growf','axisH'],['rincl']))
nodes.append(helper.make_node('Sub',['rincl','growf'],['ridx']))
nodes.append(helper.make_node('CumSum',['gcolf','axisW'],['cincl']))
nodes.append(helper.make_node('Sub',['cincl','gcolf'],['cidx']))

# 5) background mask = NOT gray (since only 0/5 exist). bg0b = not grayb
# We need bg0 to restrict painting to background cells. Since input is only 0/5, bg = ~gray.
nodes.append(helper.make_node('Not',['grayb'],['bg0b']))

# 6) band masks
C('f0', np.array(0.0,dtype=np.float32))
def band_mask(rowtgt, coltgt, p):
    nodes.append(helper.make_node('Equal',['ridx',rowtgt],[p+'rm']))
    nodes.append(helper.make_node('Equal',['cidx',coltgt],[p+'cm']))
    nodes.append(helper.make_node('And',[p+'rm',p+'cm'],[p+'rc']))
    nodes.append(helper.make_node('And',[p+'rc','bg0b'],[p+'msk']))
    return p+'msk'
m1=band_mask('f0','f0','c1_')
m2=band_mask('Rhalf','Chalf','c2_')
m3=band_mask('Rg','Cg','c3_')

# 7) base color grid g (uint8): gray->5 else 0
C('five_u8', np.array(5,dtype=np.uint8)); C('zero_u8', np.array(0,dtype=np.uint8))
nodes.append(helper.make_node('Where',['grayb','five_u8','zero_u8'],['g']))  # [1,1,10,10] u8
# paint
C('col1', np.array(1,dtype=np.uint8)); C('col2', np.array(2,dtype=np.uint8)); C('col3', np.array(3,dtype=np.uint8))
nodes.append(helper.make_node('Where',[m1,'col1','g'],['gm1']))
nodes.append(helper.make_node('Where',[m2,'col2','gm1'],['gm2']))
nodes.append(helper.make_node('Where',[m3,'col3','gm2'],['gm']))   # [1,1,10,10] u8

# 8) pad to 30x30 sentinel 255 -> Equal -> output
C('pads', np.array([0,0,0,0, 0,0,20,20],dtype=np.int64)); C('sent', np.array(255,dtype=np.uint8))
nodes.append(helper.make_node('Pad',['gm','pads','sent'],['gm30'],mode='constant'))  # u8 900B
C('colors', np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
nodes.append(helper.make_node('Equal',['gm30','colors'],['output']))

graph=helper.make_graph(nodes,'task226',
    [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],inits)
mdl=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]); mdl.ir_version=8
onnx.checker.check_model(mdl); onnx.save(mdl,'/tmp/p1e_task226.onnx')
print('saved nodes',len(nodes))
