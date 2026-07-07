import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def const(name, arr):
    return numpy_helper.from_array(arr, name=name)

nodes = []
inits = []
F16 = TensorProto.FLOAT16

# 1. collapse one-hot -> color grid g30 [1,1,30,30] f32 (irreducible 3600B)
W1 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(const("W1", W1))
nodes.append(helper.make_node("Conv", ["input", "W1"], ["g30"], kernel_shape=[1, 1]))

# 2. slice to 10x10 then cast to fp16
inits.append(const("s_start", np.array([0, 0], dtype=np.int64)))
inits.append(const("s_end", np.array([10, 10], dtype=np.int64)))
inits.append(const("s_axes", np.array([2, 3], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["g30", "s_start", "s_end", "s_axes"], ["g32"]))
nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=F16))  # g [1,1,10,10] f16 (200B)

# 4. neighbor sum via Conv ones3x3 pad=1 (f16)
ones3 = np.ones((1, 1, 3, 3), dtype=np.float16)
inits.append(const("ones3", ones3))
nodes.append(helper.make_node("Conv", ["g", "ones3"], ["s"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))

# 5. ring center R = (s==8) & (g==0)
inits.append(const("c8", np.array(8.0, dtype=np.float16)))
inits.append(const("c0", np.array(0.0, dtype=np.float16)))
nodes.append(helper.make_node("Equal", ["s", "c8"], ["eq8"]))
nodes.append(helper.make_node("Equal", ["g", "c0"], ["g0"]))
nodes.append(helper.make_node("And", ["eq8", "g0"], ["R"]))
nodes.append(helper.make_node("Cast", ["R"], ["Rf"], to=F16))  # 200B

# 6. plus mask via plus kernel conv (f16)
plus = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float16).reshape(1, 1, 3, 3)
inits.append(const("plusk", plus))
nodes.append(helper.make_node("Conv", ["Rf", "plusk"], ["pc"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
inits.append(const("half", np.array(0.5, dtype=np.float16)))
nodes.append(helper.make_node("Greater", ["pc", "half"], ["P"]))

# 7. box mask via maxpool 3x3 pad1 (f16)
nodes.append(helper.make_node("MaxPool", ["Rf"], ["Bmax"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
nodes.append(helper.make_node("Greater", ["Bmax", "half"], ["B"]))

# 9. gm color uint8 via arithmetic: gm = b*(1-B) + 2*P  (uint8)
#    where B (the 9 box cells) -> b cleared to 0, then plus cells (P) set to 2.
nodes.append(helper.make_node("Cast", ["g"], ["b"], to=TensorProto.UINT8))   # 0/1
nodes.append(helper.make_node("Where", ["B", "u0", "b"], ["cleared"]))        # b cleared in box
nodes.append(helper.make_node("Where", ["P", "u2", "cleared"], ["gm"]))       # plus -> 2
inits.append(const("u0", np.array(0, dtype=np.uint8)))
inits.append(const("u2", np.array(2, dtype=np.uint8)))

# 10. Pad gm to 30x30 with sentinel 255, then Equal with colors -> output (excluded)
inits.append(const("pads", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64)))
inits.append(const("sent", np.array(255, dtype=np.uint8)))
nodes.append(helper.make_node("Pad", ["gm", "pads", "sent"], ["gm30"], mode="constant"))  # [1,1,30,30] uint8 900B
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(const("colors", colors))
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
        r = sess.run(["output"], {"input": bm["input"]})[0]
        r = (r > 0.0).astype(float)
        if not np.array_equal(r, bm["output"]):
            bad += 1
print("bad", bad)
