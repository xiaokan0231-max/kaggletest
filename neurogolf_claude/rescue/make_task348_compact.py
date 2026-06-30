import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
N=30
U8=TensorProto.UINT8; B=TensorProto.BOOL; F=TensorProto.FLOAT
def u8(x): return np.array(x,dtype=np.uint8)
def const(n,a): return helper.make_node('Constant',[],[n],value=numpy_helper.from_array(a,n+'_v'))
nodes=[];inits=[]
def init(a,n): inits.append(numpy_helper.from_array(a,n))

init(np.array([3],dtype=np.int64),'ax3')
init(np.array([1,2],dtype=np.int64),'ax12')
nodes.append(const('zero_f',np.array(0,dtype=np.float32)))

# Conv -> channel-7 column profile prof[1,1,1,30]
Wc=np.zeros((1,10,30,1),dtype=np.float32); Wc[0,7,:,0]=1.0
init(Wc,'Wc')
nodes.append(helper.make_node('Conv',['input','Wc'],['prof'],kernel_shape=[30,1]))  # [1,1,1,30] fp32
# tip+1 = max(prof) ; col = argmax(prof)
nodes.append(helper.make_node('ReduceMax',['prof','ax3'],['tipp1_f'],keepdims=1))
nodes.append(helper.make_node('Cast',['tipp1_f'],['tip1'],to=U8))            # tip+1 scalar
nodes.append(helper.make_node('ArgMax',['prof'],['col_i'],axis=3,keepdims=1)) # int64 scalar
nodes.append(helper.make_node('Cast',['col_i'],['col'],to=U8))               # u8 scalar

# pcol presence [1,1,1,30]
nodes.append(helper.make_node('ReduceMax',['input','ax12'],['pcol_f'],keepdims=1))
nodes.append(helper.make_node('Greater',['pcol_f','zero_f'],['pcol_b']))     # bool [1,1,1,30]

# vector arithmetic [1,1,1,30]
Cvec=np.arange(N,dtype=np.uint8).reshape(1,1,1,N); init(Cvec,'Cvec')
nodes.append(helper.make_node('Max',['Cvec','col'],['hiC']))
nodes.append(helper.make_node('Min',['Cvec','col'],['loC']))
nodes.append(helper.make_node('Sub',['hiC','loC'],['absCvec']))   # [1,1,1,30]
nodes.append(const('two',u8(2)))
nodes.append(helper.make_node('Mod',['absCvec','two'],['parVec']))
nodes.append(const('seven',u8(7)))
nodes.append(helper.make_node('Add',['parVec','seven'],['colorvalVec']))  # [1,1,1,30]
colors10=np.arange(10,dtype=np.uint8).reshape(1,10,1,1); init(colors10,'colors10')
nodes.append(helper.make_node('Equal',['colorvalVec','colors10'],['cv_oh_b']))
nodes.append(helper.make_node('Cast',['cv_oh_b'],['cv_oh'],to=F))  # fp32 [1,10,1,30]

# absPad -> thr -> cone
nodes.append(const('c100',u8(100)))
nodes.append(helper.make_node('Where',['pcol_b','absCvec','c100'],['absPad']))
nodes.append(helper.make_node('Max',['tip1','absPad'],['mx']))
nodes.append(helper.make_node('Sub',['mx','absPad'],['thr']))       # [1,1,1,30]
Rvec=np.arange(N,dtype=np.uint8).reshape(1,1,N,1); init(Rvec,'Rvec')
nodes.append(helper.make_node('Less',['Rvec','thr'],['cone']))      # bool [1,1,30,30]

nodes.append(helper.make_node('Where',['cone','cv_oh','input'],['output']))

graph=helper.make_graph(nodes,'task348',
    inputs=[helper.make_tensor_value_info('input',F,[1,10,30,30])],
    outputs=[helper.make_tensor_value_info('output',F,[1,10,30,30])],
    initializer=inits)
model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]); model.ir_version=9
onnx.checker.check_model(model)
onnx.save(model,'/tmp/p1e_task348.onnx'); print('saved')
