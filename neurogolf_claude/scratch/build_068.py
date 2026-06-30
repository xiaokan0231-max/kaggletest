import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def t(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)

nodes = []
inits = []

# ---- input one-hot [1,10,30,30] f32 (excluded) ----

# per-color counts -> [1,10,1,1]
inits.append(t("ax23", np.array([2,3], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["input","ax23"], ["cnt"], keepdims=1))
inits.append(t("one_f", np.array(1.0, dtype=np.float32)))
nodes.append(helper.make_node("Equal", ["cnt","one_f"], ["isuniqc"]))       # bool [1,10,1,1]
nodes.append(helper.make_node("Cast", ["isuniqc"], ["isuniqc_f"], to=TensorProto.FLOAT))
colvec = np.arange(10, dtype=np.float32).reshape(1,10,1,1)
inits.append(t("colvec", colvec))
nodes.append(helper.make_node("Mul", ["isuniqc_f","colvec"], ["uvc"]))      # [1,10,1,1]
inits.append(t("ax1", np.array([1], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["uvc","ax1"], ["uniqval"], keepdims=1))  # [1,1,1,1] f32 = unique color

# color grid via Conv [[0..9]]
W_collapse = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(t("Wc", W_collapse))
nodes.append(helper.make_node("Conv", ["input", "Wc"], ["g30"]))            # [1,1,30,30] f32 (3600B)
# crop to 10x10
inits.append(t("s0", np.array([0,0], dtype=np.int64)))
inits.append(t("e10", np.array([10,10], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["g30","s0","e10","ax23"], ["g"]))   # [1,1,10,10] f32

# unique cell mask
nodes.append(helper.make_node("Equal", ["g","uniqval"], ["umaskb"]))        # bool [1,1,10,10] (100B)
nodes.append(helper.make_node("Cast", ["umaskb"], ["umasku"], to=TensorProto.UINT8))  # uint8 0/1 (100B)

# 3x3 box footprint via MaxPool (pad 1) on uint8 -> uint8 (100B), then *2 for border color
nodes.append(helper.make_node("MaxPool", ["umasku"], ["boxu"],
                              kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))  # uint8 [1,1,10,10]
inits.append(t("two_u", np.array(2, dtype=np.uint8)))
nodes.append(helper.make_node("Mul", ["boxu","two_u"], ["box2u"]))          # uint8 0/2 (100B)

# center color = uniqval (uint8)
nodes.append(helper.make_node("Cast", ["uniqval"], ["uvu"], to=TensorProto.UINT8))  # [1,1,1,1] uint8
# gm10 = where(umaskb, uvu, box2u)
nodes.append(helper.make_node("Where", ["umaskb","uvu","box2u"], ["gm10u"]))  # uint8 [1,1,10,10] (100B)

# pad to 30x30 with sentinel 255
inits.append(t("pads", np.array([0,0,0,0, 0,0,20,20], dtype=np.int64)))
inits.append(t("sent", np.array(255, dtype=np.uint8)))
nodes.append(helper.make_node("Pad", ["gm10u","pads","sent"], ["gm30"], mode="constant"))  # uint8 [1,1,30,30] (900B)

# one-hot output
colors = np.arange(10, dtype=np.uint8).reshape(1,10,1,1)
inits.append(t("colors", colors))
nodes.append(helper.make_node("Equal", ["gm30","colors"], ["output"]))      # bool [1,10,30,30] OUTPUT

graph = helper.make_graph(
    nodes, "task068",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1,10,30,30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b14_task068.onnx")
print("saved")

import onnxruntime as ort
import sys
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng
sess = ort.InferenceSession("/tmp/b14_task068.onnx")
d = ng.load_examples(68)
def to_oh(grid):
    grid=np.array(grid); a=np.zeros((1,10,30,30),np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            a[0,grid[r,c],r,c]=1.0
    return a
def decode(out):
    out=out[0]; res=np.full((30,30),10,dtype=int)
    for r in range(30):
        for c in range(30):
            ch=[k for k in range(10) if out[k,r,c]==1]
            res[r,c]=ch[0] if len(ch)==1 else (11 if ch else 10)
    return res
bad=0
for split in ['train','test','arc-gen']:
    for i,ex in enumerate(d[split]):
        oh=to_oh(ex['input'])
        o=sess.run(["output"],{"input":oh})[0].astype(int)
        pred=decode(o); exp=np.array(ex['output'])
        if not np.array_equal(pred[:exp.shape[0],:exp.shape[1]], exp):
            bad+=1
            if bad<=3: print('MISMATCH',split,i)
print('numpy-decode bad:', bad)
