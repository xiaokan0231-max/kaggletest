import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Build a compressed task361: C4-symmetrization about the solid-block core center.
# Algorithm (same as deployed, compressed):
#  1. Conv collapse one-hot input [1,10,10,10] -> color grid ci [1,1,10,10] (weight [0..9]).
#  2. Detect core center via solid-block detection:
#       M = (ci>0.5) mask of nonzero cells.
#       P3 = centers of 3x3 solid blocks (AveragePool 3x3 ~ 1.0)  -> odd core, integer center
#       Q  = centers of 2x2 solid blocks (AveragePool 2x2 ~ 1.0)  -> even core, half-int center
#     centroid of detected centers gives (cr,cc); use 2*center as integer "two_cr/two_cc".
#  3. Build rot180 (Fr @ ci @ Fc) and rot90 (A @ ci^T @ B) permutation matrices from 2*center.
#  4. final = max(ci, rot180, rot90, rot270-equiv) ; C4 union.
#  5. Pad final color grid to 30x30 with sentinel 255, Equal vs [0..9] -> output one-hot (excluded).

def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)

nodes = []
inits = []

# ---- constants ----
# Conv weight to collapse 10 channels to color value: channel k -> value k
W = np.arange(10, dtype=np.float32).reshape(1,10,1,1)
inits.append(c('W_ci', W))
inits.append(c('half', np.array([0.5], np.float32)))
inits.append(c('almost_one', np.array([0.9999], np.float32)))
inits.append(c('one_s', np.array(1.0, np.float32)))
inits.append(c('two_s', np.array(2.0, np.float32)))
inits.append(c('bias_cast', np.array(0.1, np.float32)))
inits.append(c('zero_f', np.array(0.0, np.float32)))
# row/col coordinate grids
r_grid = np.arange(10, dtype=np.float32).reshape(1,1,10,1)
c_grid = np.arange(10, dtype=np.float32).reshape(1,1,1,10)
inits.append(c('r_grid_f', r_grid))
inits.append(c('c_grid_f', c_grid))
# sigma_rr[i,j] = 2(i+j), delta2[i,j] = 2(j-i)
ii = np.arange(10).reshape(10,1); jj = np.arange(10).reshape(1,10)
sigma = (2*(ii+jj)).astype(np.int32)
delta = (2*(jj-ii)).astype(np.int32)
inits.append(c('sigma_rr', sigma))
inits.append(c('delta2', delta))
inits.append(c('two_int', np.array(2, np.int32)))
inits.append(c('reshape_1110', np.array([1,1,10,10], np.int64)))
# k for one-hot
inits.append(c('k_arr', np.arange(10, dtype=np.int32).reshape(1,10,1,1)))
inits.append(c('k_u8', np.arange(10, dtype=np.uint8).reshape(1,10,1,1)))
# pad specs
inits.append(c('pad_Q', np.array([0,0,0,0, 0,0,1,1], np.int64)))
inits.append(c('zero_b', np.array(False)))
# pad final color grid 10x10 -> 30x30 with sentinel 255
inits.append(c('pad_final', np.array([0,0,0,0, 0,0,20,20], np.int64)))
inits.append(c('sent255', np.array(255, np.int32)))
# spatial crop 30x30 -> 10x10 (grids live in top-left)
inits.append(c('crop_starts', np.array([0,0], np.int64)))
inits.append(c('crop_ends', np.array([10,10], np.int64)))
inits.append(c('crop_axes', np.array([2,3], np.int64)))

# ---- collapse input ----
# Conv full one-hot -> color grid [1,1,30,30], then crop to [1,1,10,10]
nodes.append(helper.make_node('Conv', ['input','W_ci'], ['ci30'], kernel_shape=[1,1], pads=[0,0,0,0]))
nodes.append(helper.make_node('Slice', ['ci30','crop_starts','crop_ends','crop_axes'], ['ci']))

# ---- core center detection ----
nodes.append(helper.make_node('Greater', ['ci','half'], ['M_bool']))
nodes.append(helper.make_node('Cast', ['M_bool'], ['M'], to=TensorProto.FLOAT))
# 3x3 solid blocks
nodes.append(helper.make_node('AveragePool', ['M'], ['P3_raw'], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node('Greater', ['P3_raw','almost_one'], ['P3_bool']))
nodes.append(helper.make_node('Cast', ['P3_bool'], ['P3'], to=TensorProto.FLOAT))
# 2x2 solid blocks
nodes.append(helper.make_node('AveragePool', ['M'], ['Q_raw'], kernel_shape=[2,2], pads=[0,0,0,0], strides=[1,1]))
nodes.append(helper.make_node('Greater', ['Q_raw','almost_one'], ['Q_bool']))
nodes.append(helper.make_node('Cast', ['Q_bool'], ['Q_9'], to=TensorProto.FLOAT))
nodes.append(helper.make_node('Pad', ['Q_9','pad_Q','zero_f'], ['Q'], mode='constant'))

# has_3 = sum(P3) (0 or >0)
nodes.append(helper.make_node('ReduceSum', ['P3'], ['has_3'], keepdims=0))
nodes.append(helper.make_node('Sub', ['one_s','has_3'], ['not_has_3']))
# centroid of P3 cells (sum r, sum c)
nodes.append(helper.make_node('Where', ['P3_bool','r_grid_f','zero_f'], ['P3_r']))
nodes.append(helper.make_node('ReduceSum', ['P3_r'], ['sum_r_p'], keepdims=0))
nodes.append(helper.make_node('Where', ['P3_bool','c_grid_f','zero_f'], ['P3_c']))
nodes.append(helper.make_node('ReduceSum', ['P3_c'], ['sum_c_p'], keepdims=0))
# centroid of Q cells (2x2 block centers are at half positions -> Q grid already shifted)
nodes.append(helper.make_node('Mul', ['Q','r_grid_f'], ['Q_r']))
nodes.append(helper.make_node('ReduceSum', ['Q_r'], ['sum_r_q'], keepdims=0))
nodes.append(helper.make_node('Mul', ['Q','c_grid_f'], ['Q_c']))
nodes.append(helper.make_node('ReduceSum', ['Q_c'], ['sum_c_q'], keepdims=0))
# select centroid
nodes.append(helper.make_node('Mul', ['has_3','sum_r_p'], ['hp_r']))
nodes.append(helper.make_node('Mul', ['not_has_3','sum_r_q'], ['nq_r']))
nodes.append(helper.make_node('Add', ['hp_r','nq_r'], ['sel_r']))
nodes.append(helper.make_node('Mul', ['sel_r','two_s'], ['two_sel_r']))
nodes.append(helper.make_node('Mul', ['has_3','sum_c_p'], ['hp_c']))
nodes.append(helper.make_node('Mul', ['not_has_3','sum_c_q'], ['nq_c']))
nodes.append(helper.make_node('Add', ['hp_c','nq_c'], ['sel_c']))
nodes.append(helper.make_node('Mul', ['sel_c','two_s'], ['two_sel_c']))
# two_cr = round(2*center_r) ; add not_has_3 (the Q centroid is offset by -0.5 in each axis,
#   so 2*center is integer minus 1; adding not_has_3 corrects by +1 when using Q path) + 0.1 bias for rounding
nodes.append(helper.make_node('Sum', ['two_sel_r','not_has_3','bias_cast'], ['two_cr_b']))
nodes.append(helper.make_node('Sum', ['two_sel_c','not_has_3','bias_cast'], ['two_cc_b']))
nodes.append(helper.make_node('Cast', ['two_cr_b'], ['two_cr'], to=TensorProto.INT32))
nodes.append(helper.make_node('Cast', ['two_cc_b'], ['two_cc'], to=TensorProto.INT32))
nodes.append(helper.make_node('Mul', ['two_cr','two_int'], ['four_cr']))
nodes.append(helper.make_node('Mul', ['two_cc','two_int'], ['four_cc']))

# rot180 permutation matrices
nodes.append(helper.make_node('Equal', ['sigma_rr','four_cr'], ['Fr_bool']))
nodes.append(helper.make_node('Equal', ['sigma_rr','four_cc'], ['Fc_bool']))
# rot90 matrices
nodes.append(helper.make_node('Sub', ['two_cc','two_cr'], ['twoDiff']))
nodes.append(helper.make_node('Equal', ['delta2','twoDiff'], ['A_bool']))
nodes.append(helper.make_node('Add', ['two_cr','two_cc'], ['twoSum']))
nodes.append(helper.make_node('Equal', ['sigma_rr','twoSum'], ['B_bool']))

# permutation matrices as float16 [1,1,10,10]
for nm in ['Fr','Fc','A','B']:
    nodes.append(helper.make_node('Reshape', [nm+'_bool','reshape_1110'], [nm+'4_pre_bool']))
    nodes.append(helper.make_node('Cast', [nm+'4_pre_bool'], [nm+'4'], to=TensorProto.FLOAT16))

# color grid to float16
nodes.append(helper.make_node('Cast', ['ci'], ['ci16'], to=TensorProto.FLOAT16))
# rot180 = Fr @ ci @ Fc
nodes.append(helper.make_node('MatMul', ['Fr4','ci16'], ['Frc']))
nodes.append(helper.make_node('MatMul', ['Frc','Fc4'], ['rot180_ci']))
nodes.append(helper.make_node('Max', ['ci16','rot180_ci'], ['y']))
# rot90 = A @ y^T @ B
nodes.append(helper.make_node('Transpose', ['y'], ['yT'], perm=[0,1,3,2]))
nodes.append(helper.make_node('MatMul', ['A4','yT'], ['AyT']))
nodes.append(helper.make_node('MatMul', ['AyT','B4'], ['rot90_y']))
nodes.append(helper.make_node('Max', ['y','rot90_y'], ['final_ci']))

# output: final color grid (fp16) -> uint8, Equal vs [0..9] on 10x10 -> bool one-hot [1,10,10,10],
#         then Pad spatially to 30x30 (the one-hot is interior at 1000B; padded output excluded).
nodes.append(helper.make_node('Cast', ['final_ci'], ['ci_u8'], to=TensorProto.UINT8))
nodes.append(helper.make_node('Equal', ['ci_u8','k_u8'], ['eq_bool']))
nodes.append(helper.make_node('Pad', ['eq_bool','pad_final','zero_b'], ['output'], mode='constant'))

graph = helper.make_graph(
    nodes, 'task361_compact',
    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,10,30,30])],
    [helper.make_tensor_value_info('output', TensorProto.BOOL, [1,10,30,30])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, '/tmp/b13_task361.onnx')
print('saved')
