import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
def C(name, arr): return numpy_helper.from_array(np.asarray(arr), name=name)
nodes=[]; inits=[]
F16=TensorProto.FLOAT16; U8=TensorProto.UINT8

# 1. collapse -> g30f (3600 floor)
inits.append(C('Wc', np.arange(10, dtype=np.float32).reshape(1,10,1,1)))
nodes.append(helper.make_node('Conv',['input','Wc'],['g30f'], kernel_shape=[1,1], pads=[0,0,0,0]))
# 2. slice 11x11 f32, cast uint8
inits.append(C('s0', np.array([0,0], dtype=np.int64)))
inits.append(C('e11', np.array([11,11], dtype=np.int64)))
inits.append(C('ax23', np.array([2,3], dtype=np.int64)))
nodes.append(helper.make_node('Slice',['g30f','s0','e11','ax23'],['g11f']))   # f32 484
nodes.append(helper.make_node('Cast',['g11f'],['g11u'], to=U8))               # u8 121
# 3. gather 9x9 content uint8
idx = np.array([0,1,2,4,5,6,8,9,10], dtype=np.int64)
inits.append(C('idx', idx))
nodes.append(helper.make_node('Gather',['g11u','idx'],['gH'], axis=2))   # u8 99
nodes.append(helper.make_node('Gather',['gH','idx'],['ch'], axis=3))     # u8 81
# 4. reshape to blocks [3,3,3,3] uint8
inits.append(C('s3333', np.array([3,3,3,3], dtype=np.int64)))
nodes.append(helper.make_node('Reshape',['ch','s3333'],['blk_u']))      # u8 81
# 5. keyInd: blockMax==8 means block has 8 (not key); key block max is 6
nodes.append(helper.make_node('ReduceMax',['blk_u'],['blkMax'], axes=[1,3], keepdims=1))  # u8 [3,1,3,1] 9
inits.append(C('c8', np.array(8, dtype=np.uint8)))
nodes.append(helper.make_node('Less',['blkMax','c8'],['kib']))         # bool [3,1,3,1] 9  (max<8 -> key)
# 6. keyblock = sum_{bi,bj} where(keyInd, blk_u, 0)
inits.append(C('zu2', np.array(0, dtype=np.uint8)))
nodes.append(helper.make_node('Where',['kib','blk_u','zu2'],['masked_u']))  # u8 [3,3,3,3] 81
nodes.append(helper.make_node('Cast',['masked_u'],['masked'], to=F16))      # fp16 162
inits.append(C('ax02', np.array([0,2], dtype=np.int64)))
nodes.append(helper.make_node('ReduceSum',['masked','ax02'],['keyblock'], keepdims=0))  # fp16 [3,3] 18
# 7. upsample + separators
P=np.zeros((11,3),dtype=np.float16)
for r in range(11):
    if r in (3,7): continue
    P[r, 0 if r<3 else (1 if r<7 else 2)]=1
inits.append(C('P', P)); inits.append(C('PT', P.T.copy()))
nodes.append(helper.make_node('MatMul',['P','keyblock'],['tmp']))      # fp16 66
sep5=np.zeros((11,11),dtype=np.float16); sep5[3,:]=5; sep5[7,:]=5; sep5[:,3]=5; sep5[:,7]=5
inits.append(C('sep5', sep5))
nodes.append(helper.make_node('Gemm',['tmp','PT','sep5'],['ans'], alpha=1.0, beta=1.0))  # fp16 242
nodes.append(helper.make_node('Cast',['ans'],['ansu'], to=U8))        # u8 121
inits.append(C('pads', np.array([0,0, 19,19], dtype=np.int64)))
inits.append(C('p255', np.array(255, dtype=np.uint8)))
nodes.append(helper.make_node('Pad',['ansu','pads','p255'],['gm'], mode='constant'))  # u8 900
inits.append(C('colors', np.arange(10, dtype=np.uint8).reshape(1,10,1,1)))
nodes.append(helper.make_node('Equal',['gm','colors'],['output']))

inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,10,30,30])
out = helper.make_tensor_value_info('output', TensorProto.BOOL, [1,10,30,30])
graph = helper.make_graph(nodes, 'task011', [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',13)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, '/tmp/b9_task011.onnx')
print('saved i. nodes', len(nodes))
