import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Optimized: keep ALL interior tensors [1,1,30,30] in 1-byte dtypes; make Equal the OUTPUT.
# g_f = Conv(input,[0..9]) f32 [1,1,30,30]  (3600 B, unavoidable f32 conv out)
# g   = Cast(g_f, uint8)                    (900)
# P   = ReduceMax(g) scalar uint8
# s   = row + col (uint8, r+c max 30 fits in uint8)  [1,1,30,30] (900)
# m   = Mod(s, P) uint8 ; gm = m + 1 uint8           (900,900)
# inside via ReduceSum(input,ch) f32 (3600) -> Cast bool? use Greater -> bool (900)
#   then masked = Where(inside, gm, 255)   uint8 (900)
# output = Equal(masked[1,1,30,30], colors[1,10,1,1]) -> bool [1,10,30,30] OUTPUT (excluded)

H = W = 30
g_in = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, H, W])
g_out = helper.make_tensor_value_info('output', TensorProto.BOOL, [1, 10, H, W])

inits = []
nodes = []

w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(w, 'cw'))
nodes.append(helper.make_node('Conv', ['input', 'cw'], ['g_f'], kernel_shape=[1, 1]))
nodes.append(helper.make_node('Cast', ['g_f'], ['g'], to=TensorProto.UINT8))

nodes.append(helper.make_node('ReduceMax', ['g'], ['P'], keepdims=0))  # uint8 scalar

# s = row + col, uint8 (values 0..58 -> wait max r+c = 29+29=58, fits uint8). Use uint8.
row = np.arange(H, dtype=np.uint8).reshape(1, 1, H, 1)
col = np.arange(W, dtype=np.uint8).reshape(1, 1, 1, W)
inits.append(numpy_helper.from_array(row, 'row'))
inits.append(numpy_helper.from_array(col, 'col'))
nodes.append(helper.make_node('Add', ['row', 'col'], ['s']))  # uint8 [1,1,30,30]

nodes.append(helper.make_node('Mod', ['s', 'P'], ['m']))  # uint8
one = np.array(1, dtype=np.uint8)
inits.append(numpy_helper.from_array(one, 'one_u'))
nodes.append(helper.make_node('Add', ['m', 'one_u'], ['gm']))  # uint8 1..P

# inside mask: ReduceSum(input over ch) > 0
inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), 'axis1'))
nodes.append(helper.make_node('ReduceSum', ['input', 'axis1'], ['chsum'], keepdims=1))  # f32 [1,1,30,30]
inits.append(numpy_helper.from_array(np.array(0, dtype=np.float32), 'zero_f'))
nodes.append(helper.make_node('Greater', ['chsum', 'zero_f'], ['inside']))  # bool

# masked = Where(inside, gm, 255)
inits.append(numpy_helper.from_array(np.array(255, dtype=np.uint8), 'sentinel'))
nodes.append(helper.make_node('Where', ['inside', 'gm', 'sentinel'], ['masked']))  # uint8

colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(colors, 'colors'))
nodes.append(helper.make_node('Equal', ['masked', 'colors'], ['output']))  # bool [1,10,30,30] OUTPUT

graph = helper.make_graph(nodes, 'task305', [g_in], [g_out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, '/tmp/p1f_task305.onnx')
print('saved')

import onnxruntime as ort, sys
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng
d = ng.load_examples(305)
sess = ort.InferenceSession('/tmp/p1f_task305.onnx')

def to_canon(grid):
    a = np.zeros((1, 10, 30, 30), np.float32)
    g = np.array(grid)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            a[0, int(g[r, c]), r, c] = 1.0
    return a

def from_canon(arr):
    b = (arr > 0.0).astype(float)[0]
    out = []
    for r in range(30):
        cells = []
        for c in range(30):
            cs = [ch for ch in range(10) if b[ch, r, c] == 1]
            cells.append(cs[0] if len(cs) == 1 else (11 if cs else 10))
        while cells and cells[-1] == 10:
            cells.pop()
        out.append(cells)
    while out and not out[-1]:
        out.pop()
    return out

bad = 0; tot = 0
for split in ['train', 'test', 'arc-gen']:
    for ex in d[split]:
        r = sess.run(['output'], {'input': to_canon(ex['input'])})[0]
        if from_canon(r) != ex['output']:
            bad += 1
        tot += 1
print('canonical mismatch', bad, '/', tot)
