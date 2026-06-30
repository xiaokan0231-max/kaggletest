import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Clean floor: 2 inits (s_const 900p, grid_mask 900p) + 2 interior (Mod m 900, Where masked 900) = 3600.
H = W = 30
GRID = 16
g_in = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, H, W])
g_out = helper.make_tensor_value_info('output', TensorProto.BOOL, [1, 10, H, W])
inits = []
nodes = []

# P extraction (tiny [1,10,1,1])
inits.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), 'sp_axes'))
nodes.append(helper.make_node('ReduceMax', ['input', 'sp_axes'], ['pres'], keepdims=1))
idx = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(idx, 'idx'))
nodes.append(helper.make_node('Mul', ['pres', 'idx'], ['pidx']))
nodes.append(helper.make_node('ReduceMax', ['pidx'], ['P_f'], keepdims=0))
nodes.append(helper.make_node('Cast', ['P_f'], ['P'], to=TensorProto.UINT8))

# s_const initializer (r+c) — direct Mod input, NOT a node output
rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
s = (rr + cc).astype(np.uint8).reshape(1, 1, H, W)
inits.append(numpy_helper.from_array(s, 's_const'))
nodes.append(helper.make_node('Mod', ['s_const', 'P'], ['m']))  # interior 900

# grid_mask initializer (bool, 1 byte) — Where condition
gm = np.zeros((1, 1, H, W), dtype=bool)
gm[0, 0, :GRID, :GRID] = True
inits.append(numpy_helper.from_array(gm, 'grid_mask'))
inits.append(numpy_helper.from_array(np.array(254, dtype=np.uint8), 'sentinel'))
nodes.append(helper.make_node('Where', ['grid_mask', 'm', 'sentinel'], ['m_masked']))  # interior 900

cvals = np.array([253, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(cvals, 'cvals'))
nodes.append(helper.make_node('Equal', ['m_masked', 'cvals'], ['output']))

graph = helper.make_graph(nodes, 'task305', [g_in], [g_out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, '/tmp/p1f_task305_e.onnx')
print('saved')

import onnxruntime as ort, sys
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng
d = ng.load_examples(305)
sess = ort.InferenceSession('/tmp/p1f_task305_e.onnx')

def to_canon(grid):
    a = np.zeros((1, 10, 30, 30), np.float32); g = np.array(grid)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            a[0, int(g[r, c]), r, c] = 1.0
    return a

def from_canon(arr):
    b = (arr > 0.0).astype(float)[0]; out = []
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
