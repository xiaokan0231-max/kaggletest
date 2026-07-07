import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Canonical: input/output [1,10,30,30]. Real grid is top-left 16x16; outside = all-zero channels.
# Rule: P = max(real color); out[r,c]=((r+c) mod P)+1 for r,c in grid; outside stays all-zero.
# Pipeline (all interior tensors [1,1,30,30] kept uint8/int8/bool = small):
#   g = Conv(input, [0..9])            -> [1,1,30,30] color grid (0 outside & for noise cells)
#   P = ReduceMax(g)                   -> scalar
#   s = row + col                      -> [1,1,30,30] r+c
#   gm = (s mod P) + 1                 -> [1,1,30,30] uint8 pattern color (everywhere)
#   inside = ReduceSum(input,ch)>0     -> [1,1,30,30] bool  (1 inside the 16x16 grid)
#   eq = Equal(gm, colors[1,10,1,1])   -> would be [1,10,30,30]; combine with inside as AND -> output
#   output = And(eq, inside)           -> bool [1,10,30,30] = graph output (EXCLUDED from memory)

H = W = 30
g_in = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, H, W])
g_out = helper.make_tensor_value_info('output', TensorProto.BOOL, [1, 10, H, W])

inits = []
nodes = []

# Conv collapse weight [1,10,1,1] = 0..9  -> color grid g_f f32
w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(w, 'cw'))
nodes.append(helper.make_node('Conv', ['input', 'cw'], ['g_f'], kernel_shape=[1, 1]))
nodes.append(helper.make_node('Cast', ['g_f'], ['g'], to=TensorProto.INT32))

# P = ReduceMax(g)
nodes.append(helper.make_node('ReduceMax', ['g'], ['P'], keepdims=0))

# row/col index vectors -> s = r+c
row = np.arange(H, dtype=np.int32).reshape(1, 1, H, 1)
col = np.arange(W, dtype=np.int32).reshape(1, 1, 1, W)
inits.append(numpy_helper.from_array(row, 'row'))
inits.append(numpy_helper.from_array(col, 'col'))
nodes.append(helper.make_node('Add', ['row', 'col'], ['s']))  # [1,1,30,30] int32

# m = Mod(s,P); gm_i = m+1 ; gm uint8
nodes.append(helper.make_node('Mod', ['s', 'P'], ['m']))
one = np.array(1, dtype=np.int32)
inits.append(numpy_helper.from_array(one, 'one_i'))
nodes.append(helper.make_node('Add', ['m', 'one_i'], ['gm_i']))
nodes.append(helper.make_node('Cast', ['gm_i'], ['gm'], to=TensorProto.UINT8))

# inside mask = ReduceSum(input over channels axis=1) > 0
nodes.append(helper.make_node('ReduceSum', ['input', 'axis1'], ['chsum'], keepdims=1))
inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), 'axis1'))
zero_f = np.array(0, dtype=np.float32)
inits.append(numpy_helper.from_array(zero_f, 'zero_f'))
nodes.append(helper.make_node('Greater', ['chsum', 'zero_f'], ['inside']))  # bool [1,1,30,30]

# colors [1,10,1,1] uint8 0..9 ; eq = Equal(gm,colors) bool [1,10,30,30]
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(colors, 'colors'))
nodes.append(helper.make_node('Equal', ['gm', 'colors'], ['eq']))
# output = And(eq, inside)
nodes.append(helper.make_node('And', ['eq', 'inside'], ['output']))

graph = helper.make_graph(nodes, 'task305', [g_in], [g_out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, '/tmp/p1f_task305.onnx')
print('saved')

# numeric self-test in canonical 30x30 format
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
    b = (arr > 0.0).astype(float)[0]  # [10,30,30]
    H = 30; out = []
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
        pred = from_canon(r)
        tot += 1
        if pred != ex['output']:
            bad += 1
            if bad <= 2:
                print('MISMATCH split', split)
print('canonical mismatch', bad, '/', tot)
