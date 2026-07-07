import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
nodes=[]; inits=[]
def C(n,a): inits.append(numpy_helper.from_array(np.asarray(a),name=n)); return n

C('s5',np.array([5,0,0],dtype=np.int64));C('e5',np.array([6,10,10],dtype=np.int64));C('a5',np.array([1,2,3],dtype=np.int64))
nodes.append(helper.make_node('Slice',['input','s5','e5','a5'],['grayf']))
C('half',np.array(0.5,dtype=np.float32))
nodes.append(helper.make_node('Greater',['grayf','half'],['grayb']))

C('axW',np.array([3],dtype=np.int64));C('axH',np.array([2],dtype=np.int64))
nodes.append(helper.make_node('ReduceSum',['grayf','axW'],['rowcnt'],keepdims=1))
nodes.append(helper.make_node('ReduceSum',['grayf','axH'],['colcnt'],keepdims=1))
C('ten',np.array(10,dtype=np.float32))
nodes.append(helper.make_node('Equal',['rowcnt','ten'],['growb']))
nodes.append(helper.make_node('Equal',['colcnt','ten'],['gcolb']))
nodes.append(helper.make_node('Cast',['growb'],['growf'],to=TensorProto.FLOAT))
nodes.append(helper.make_node('Cast',['gcolb'],['gcolf'],to=TensorProto.FLOAT))

nodes.append(helper.make_node('ReduceSum',['growf'],['Rg'],keepdims=0))
nodes.append(helper.make_node('ReduceSum',['gcolf'],['Cg'],keepdims=0))
C('two',np.array(2.0,dtype=np.float32))
nodes.append(helper.make_node('Div',['Rg','two'],['Rhalf']))
nodes.append(helper.make_node('Div',['Cg','two'],['Chalf']))

# INCLUSIVE cumsum used directly (gray overlaid last)
C('cH',np.array(2,dtype=np.int64));C('cW',np.array(3,dtype=np.int64))
nodes.append(helper.make_node('CumSum',['growf','cH'],['ridx']))   # inclusive [1,1,10,1]
nodes.append(helper.make_node('CumSum',['gcolf','cW'],['cidx']))   # inclusive [1,1,1,10]

C('f0',np.array(0.0,dtype=np.float32))
def bm(rt,ct,p):
    nodes.append(helper.make_node('Equal',['ridx',rt],[p+'rm']))
    nodes.append(helper.make_node('Equal',['cidx',ct],[p+'cm']))
    nodes.append(helper.make_node('And',[p+'rm',p+'cm'],[p+'m']))
    return p+'m'
m1=bm('f0','f0','c1_');m2=bm('Rhalf','Chalf','c2_');m3=bm('Rg','Cg','c3_')

C('z',np.array(0,dtype=np.uint8));C('c1',np.array(1,dtype=np.uint8));C('c2',np.array(2,dtype=np.uint8));C('c3',np.array(3,dtype=np.uint8))
nodes.append(helper.make_node('Where',[m1,'c1','z'],['p1']))
nodes.append(helper.make_node('Where',[m2,'c2','p1'],['p2']))
nodes.append(helper.make_node('Where',[m3,'c3','p2'],['painted']))
C('f5',np.array(5,dtype=np.uint8))
nodes.append(helper.make_node('Where',['grayb','f5','painted'],['gm']))

C('pads',np.array([0,0,0,0,0,0,20,20],dtype=np.int64));C('sent',np.array(255,dtype=np.uint8))
nodes.append(helper.make_node('Pad',['gm','pads','sent'],['gm30'],mode='constant'))
C('colors',np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
nodes.append(helper.make_node('Equal',['gm30','colors'],['output']))

graph=helper.make_graph(nodes,'task226',
    [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],inits)
mdl=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);mdl.ir_version=8
onnx.checker.check_model(mdl);onnx.save(mdl,'/tmp/p1e_task226.onnx')
print('saved nodes',len(nodes))
