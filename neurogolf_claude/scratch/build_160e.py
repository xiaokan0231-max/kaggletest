import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def const(name, arr):
    return numpy_helper.from_array(arr, name=name)

nodes, inits = [], []
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8

# 1. Slice input channel-1 (foreground mask), rows/cols 0:10  -> b1 [1,1,10,10] f32 (400B)
#    Foreground color is always 1, so channel 1 of the one-hot is exactly the binary mask.
inits.append(const("s_start", np.array([1, 0, 0], dtype=np.int64)))
inits.append(const("s_end",   np.array([2, 10, 10], dtype=np.int64)))
inits.append(const("s_axes",  np.array([1, 2, 3], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["input", "s_start", "s_end", "s_axes"], ["b32"]))

# 2. ring detection: kernel ones with center -8.  cv==8 <=> ring center.  (Conv in f32 on b32)
hollow = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
inits.append(const("hollow", hollow))
nodes.append(helper.make_node("Conv", ["b32", "hollow"], ["cv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
inits.append(const("c8", np.array(8.0, dtype=np.float32)))
nodes.append(helper.make_node("Equal", ["cv", "c8"], ["R"]))

# Ru = 2 where ring else 0
inits.append(const("two_u8", np.array(2, dtype=np.uint8)))
inits.append(const("zero_u8", np.array(0, dtype=np.uint8)))
nodes.append(helper.make_node("Where", ["R", "two_u8", "zero_u8"], ["Ru"]))

# 3. box dilation (3x3) and plus dilation (vert+horz) via MaxPool, carrying value 2
nodes.append(helper.make_node("MaxPool", ["Ru"], ["B"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
nodes.append(helper.make_node("MaxPool", ["Ru"], ["Pv"], kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1]))
nodes.append(helper.make_node("MaxPool", ["Ru"], ["Ph"], kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1]))
nodes.append(helper.make_node("Max", ["Pv", "Ph"], ["P"]))

# 4. gm = where(box) -> P (2 on plus, 0 on corner) ; else b
nodes.append(helper.make_node("Cast", ["b32"], ["b"], to=U8))
nodes.append(helper.make_node("Cast", ["B"], ["Bb"], to=TensorProto.BOOL))
nodes.append(helper.make_node("Where", ["Bb", "P", "b"], ["gm"]))

# 5. Pad to 30x30 sentinel 255, Equal colors -> output
inits.append(const("pads", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64)))
inits.append(const("sent", np.array(255, dtype=np.uint8)))
nodes.append(helper.make_node("Pad", ["gm", "pads", "sent"], ["gm30"], mode="constant"))
inits.append(const("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "task160", [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/p1e_task160.onnx")
print("saved")

import sys
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng
import onnxruntime as ort
sess = ort.InferenceSession("/tmp/p1e_task160.onnx")
d = ng.load_examples(160)
bad = 0
for split in ['train', 'test', 'arc-gen']:
    for ex in d[split]:
        bm = ng.convert_to_numpy(ex)
        r = (sess.run(["output"], {"input": bm["input"]})[0] > 0.0).astype(float)
        if not np.array_equal(r, bm["output"]):
            bad += 1
print("bad", bad)
