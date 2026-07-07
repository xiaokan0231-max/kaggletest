import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Goal: only THREE [1,1,30,30] quantities total (s, m, m_masked), plus tiny.
# iota row/col encode out-of-grid as big:
#   row[r] = r if r<16 else 100 ; col[c] = c if c<16 else 100  (each 30 params)
#   s = row+col : in-grid <=30, out-of-grid >=100   (interior 900)
#   m = Mod(s,P)  (interior 900)
#   inside = Less(s, 60)  (interior 900 bool)  <-- this is a 4th 900 tensor :(
# To stay at 3, fold mask via: m_masked = Where(Less(s,60), m, 254). The Less output is interior 900.
# That's 4 (s, m, less, masked). Instead reuse: compute mask from s WITHOUT extra full tensor by
# using Min: m2 = Min(m, ... ) no.
# Alternative: out-of-grid -> Mod(big,P) gives junk; but we then Greater on s to mask.
# Accept 3 BIG tensors by removing 's' as a separate tensor: feed an s_const initializer (900 param)
# directly to BOTH Mod and a Less. Then interior = m(900)+less(900)+masked(900)=2700, param s=900 => 3600.
# Hmm same. Try: param s_const(900) + interior m(900) + interior masked(900); derive mask cheaply
#   from row/col broadcast (tiny) AND-ed -> but And output is 900 interior.
#
# Best realistic: s_const initializer(900p) ; m=Mod(s_const,P) interior(900) ;
#   m_masked = Where(grid_mask_broadcast, m, 254) where grid_mask built from two small vectors:
#   rmask[1,1,30,1] bool (30p) , cmask[1,1,1,30] bool(30p); And -> [1,1,30,30] interior(900).
# total grid tensors: s_const(p900)+m(i900)+andmask(i900)+masked(i900) = 4. worse.
#
# CONCLUSION attempt: minimize to s_const(p900) + m(i900) + masked(i900) = 2700 by deriving the
# Where condition directly as a comparison on s_const that ORT may fuse... but it still outputs 900.
#
# Pragmatic best: keep v3 structure but replace grid_mask(900 param) by encoding sentinel INTO s_const
# and gate with Greater reusing s_const, trading the 900-param grid_mask for a 900-interior 'inside'.
# Net identical. So v3 (3708) seems near-floor. We instead shave PARAMS elsewhere & try fp-free.

# --- Implement: s_const has 254 baked OUT-of-grid won't survive Mod, so compute inside from s_const ---
H = W = 30
GRID = 16
g_in = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, H, W])
g_out = helper.make_tensor_value_info('output', TensorProto.BOOL, [1, 10, H, W])
inits = []
nodes = []

# P extraction (tiny)
inits.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), 'sp_axes'))
nodes.append(helper.make_node('ReduceMax', ['input', 'sp_axes'], ['pres'], keepdims=1))
idx = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(idx, 'idx'))
nodes.append(helper.make_node('Mul', ['pres', 'idx'], ['pidx']))
nodes.append(helper.make_node('ReduceMax', ['pidx'], ['P_f'], keepdims=0))
nodes.append(helper.make_node('Cast', ['P_f'], ['P'], to=TensorProto.UINT8))

# s_const: (r+c) in-grid, 200 out-of-grid  (single 900-param initializer; serves s AND mask)
rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
s = (rr + cc).astype(np.uint8)
s[GRID:, :] = 200
s[:, GRID:] = 200
s = s.reshape(1, 1, H, W)
inits.append(numpy_helper.from_array(s, 's_const'))

# inside = Less(s_const, 100) -> bool (interior 900)
inits.append(numpy_helper.from_array(np.array(100, dtype=np.uint8), 'thresh'))
nodes.append(helper.make_node('Less', ['s_const', 'thresh'], ['inside']))

# m = Mod(s_const, P) (interior 900)
nodes.append(helper.make_node('Mod', ['s_const', 'P'], ['m']))

# m_masked = Where(inside, m, 254) (interior 900)
inits.append(numpy_helper.from_array(np.array(254, dtype=np.uint8), 'sentinel'))
nodes.append(helper.make_node('Where', ['inside', 'm', 'sentinel'], ['m_masked']))

# cvals: channel ch matches m==ch-1. ch0 -> 253 (never).
cvals = np.array([253, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(numpy_helper.from_array(cvals, 'cvals'))
nodes.append(helper.make_node('Equal', ['m_masked', 'cvals'], ['output']))

graph = helper.make_graph(nodes, 'task305', [g_in], [g_out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, '/tmp/p1f_task305_d.onnx')
print('saved')

import onnxruntime as ort, sys
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng
d = ng.load_examples(305)
sess = ort.InferenceSession('/tmp/p1f_task305_d.onnx')

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
