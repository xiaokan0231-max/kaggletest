import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Per-cell value v: out-grid=0, in-grid-nonzero=1, in-grid-color0=M(=100).
# ConvRow kernel [1,30] single out-channel -> rowsum [1,1,30,1].
#   line row <=> 0.5 < rowsum < 99.5  (nonempty, no color0).
# center = rowline (x) colline. box = dilate each. neigh = box xor center.
# output = Where(neigh, e4, input).

M = 100.0
def make(path='/tmp/b16_task151.onnx'):
    nodes=[]; inits=[]
    def C(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name

    wrow = np.ones((1,10,1,30),dtype=np.float32); wrow[0,0,0,:]=M   # channel0 -> M, others 1
    C('Wr', wrow)
    nodes.append(helper.make_node('Conv',['input','Wr'],['rsum'],kernel_shape=[1,30]))  # [1,1,30,1]
    wcol = np.ones((1,10,30,1),dtype=np.float32); wcol[0,0,:,0]=M
    C('Wc', wcol)
    nodes.append(helper.make_node('Conv',['input','Wc'],['csum'],kernel_shape=[30,1]))  # [1,1,1,30]

    C('half', np.array([0.5],dtype=np.float32))
    C('hi',   np.array([99.5],dtype=np.float32))
    nodes.append(helper.make_node('Greater',['rsum','half'],['rg']))
    nodes.append(helper.make_node('Less',   ['rsum','hi'],['rl']))
    nodes.append(helper.make_node('And',['rg','rl'],['rowline']))   # [1,1,30,1] bool
    nodes.append(helper.make_node('Greater',['csum','half'],['cg']))
    nodes.append(helper.make_node('Less',   ['csum','hi'],['cl']))
    nodes.append(helper.make_node('And',['cg','cl'],['colline']))   # [1,1,1,30] bool

    C('halfh', np.array([0.5],dtype=np.float16))
    nodes.append(helper.make_node('Cast',['rowline'],['rl16'],to=TensorProto.FLOAT16))
    nodes.append(helper.make_node('MaxPool',['rl16'],['brl16'],kernel_shape=[3,1],pads=[1,0,1,0]))
    nodes.append(helper.make_node('Greater',['brl16','halfh'],['boxrow']))
    nodes.append(helper.make_node('Cast',['colline'],['cl16'],to=TensorProto.FLOAT16))
    nodes.append(helper.make_node('MaxPool',['cl16'],['bcl16'],kernel_shape=[1,3],pads=[0,1,0,1]))
    nodes.append(helper.make_node('Greater',['bcl16','halfh'],['boxcol']))

    nodes.append(helper.make_node('And',['rowline','colline'],['center']))
    nodes.append(helper.make_node('And',['boxrow','boxcol'],['box']))
    nodes.append(helper.make_node('Xor',['box','center'],['neigh']))

    e4 = np.zeros((1,10,1,1),dtype=np.float32); e4[0,4,0,0]=1.0; C('e4', e4)
    nodes.append(helper.make_node('Where',['neigh','e4','input'],['output']))

    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,10,30,30])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,10,30,30])
    graph = helper.make_graph(nodes, 'task151', [inp], [out], inits)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("",10)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    print('saved', path)

if __name__=='__main__':
    make()
