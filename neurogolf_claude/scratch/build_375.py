import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

N, C, H, W = 1, 10, 30, 30

nodes = []
inits = []
def ini(name, arr):
    inits.append(numpy_helper.from_array(arr, name))

# all constants are INITIALIZERS (counted only in params, NOT in memory)
ini("axall", np.array([0,1,2,3], dtype=np.int64))
ini("one", np.array(1.0, dtype=np.float16).reshape(1,1,1,1))
ini("row2", (2*np.arange(H, dtype=np.float16)).reshape(1,1,H,1))
ini("col2", (2*np.arange(W, dtype=np.float16)).reshape(1,1,1,W))
ini("rowidxh", np.arange(H, dtype=np.float16).reshape(1,1,H,1))
ini("sent", np.array(99.0, dtype=np.float16).reshape(1,1,1,1))
e0 = np.zeros((1,10,1,1), dtype=np.float32); e0[0,0,0,0]=1.0
ini("e0", e0)

# ---------- H = sqrt(total hot cells) ; sumdiag = H-1 (only scalar intermediates) ----------
nodes.append(helper.make_node("ReduceSum", ["input","axall"], ["total"], keepdims=1))  # [1,1,1,1] f32 = H*W = H^2
nodes.append(helper.make_node("Sqrt", ["total"], ["Hf"]))  # [1,1,1,1] f32 = H
nodes.append(helper.make_node("Cast", ["Hf"], ["Hh"], to=TensorProto.FLOAT16))  # fp16 H
nodes.append(helper.make_node("Sub", ["Hh","one"], ["sumdiag"]))  # fp16 [1,1,1,1]=H-1

# ---------- au = |2i-(H-1)| with out-of-grid rows set to sentinel 99 (fp16) ----------
nodes.append(helper.make_node("Sub", ["row2","sumdiag"], ["u"]))   # fp16 [1,1,30,1] 60B
nodes.append(helper.make_node("Abs", ["u"], ["au"]))              # fp16 [1,1,30,1] 60B
nodes.append(helper.make_node("LessOrEqual", ["rowidxh","sumdiag"], ["inrow"]))  # bool [1,1,30,1]
nodes.append(helper.make_node("Where", ["inrow","au","sent"], ["aum"]))  # fp16 [1,1,30,1] 60B

# ---------- av = |2j-(H-1)| (fp16) ----------
nodes.append(helper.make_node("Sub", ["col2","sumdiag"], ["v"]))   # fp16 [1,1,1,30] 60B
nodes.append(helper.make_node("Abs", ["v"], ["av"]))              # fp16 [1,1,1,30] 60B

# cond = (aum == av) broadcast -> [1,1,30,30] bool (single big tensor 900B)
nodes.append(helper.make_node("Equal", ["aum","av"], ["cond"]))

# ---------- output = Where(cond, e0_onehot, input) ----------
nodes.append(helper.make_node("Where", ["cond","e0","input"], ["output"]))  # [1,10,30,30] graph output

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N,C,H,W])
out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N,C,H,W])
graph = helper.make_graph(nodes, "task375", [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",18)], ir_version=9)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b15_task375.onnx")
print("saved")

import onnxruntime as ort
import sys
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng
d = ng.load_examples(375)
sess = ort.InferenceSession("/tmp/b15_task375.onnx")
def to_oh(grid):
    x = np.zeros((1,10,30,30), np.float32)
    g = np.array(grid)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            x[0,g[r,c],r,c]=1.0
    return x
bad=0
for split in ['train','test','arc-gen']:
    for ex in d[split]:
        xi=to_oh(ex['input'])
        r=(sess.run(["output"],{"input":xi})[0]>0.0).astype(float)
        if not np.array_equal(r, to_oh(ex['output'])):
            bad+=1
print("mismatches", bad)
