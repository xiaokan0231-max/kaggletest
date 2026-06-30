import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import sys
sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng
import onnxruntime as ort

H = W = 30

def const(name, arr):
    return numpy_helper.from_array(arr, name=name)

nodes = []
inits = []

# 1) slice input to channel 5, rows 0-2, cols 0-2 -> is5 [1,1,3,3] f32 (36B) = the "5" mask
inits.append(const("cs", np.array([5, 0, 0], dtype=np.int64)))
inits.append(const("ce", np.array([6, 3, 3], dtype=np.int64)))
inits.append(const("cax", np.array([1, 2, 3], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["input", "cs", "ce", "cax"], ["is5"]))

# 2) cv = Conv(is5, [2,4,3]) -> [1,1,3,1] f32 mapped color per row
w_row = np.array([2.0, 4.0, 3.0], dtype=np.float32).reshape(1, 1, 1, 3)
inits.append(const("w_row", w_row))
nodes.append(helper.make_node("Conv", ["is5", "w_row"], ["cv"], kernel_shape=[1, 3]))

# 3) cast cv to uint8 first (3B) then tile to 3 cols -> c3u [1,1,3,3] uint8 (9B)
nodes.append(helper.make_node("Cast", ["cv"], ["cvu"], to=TensorProto.UINT8))
nodes.append(helper.make_node("Concat", ["cvu", "cvu", "cvu"], ["c3u"], axis=3))

# 4) small one-hot for 3x3 -> oh3 [1,10,3,3] bool (90B)
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(const("colors", colors))
nodes.append(helper.make_node("Equal", ["c3u", "colors"], ["oh3"]))

# 5) Pad-as-output -> output [1,10,30,30] bool
pads = np.array([0, 0, 0, 0, 0, 0, H - 3, W - 3], dtype=np.int64)
inits.append(const("pads", pads))
inits.append(const("padF", np.array(False, dtype=bool)))
nodes.append(helper.make_node("Pad", ["oh3", "pads", "padF"], ["output"], mode="constant"))

graph = helper.make_graph(
    nodes, "task262",
    inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])],
    outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, H, W])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "/tmp/p1f_task262.onnx")
print("saved /tmp/p1f_task262.onnx")

ex = ng.load_examples(262)
sess = ort.InferenceSession("/tmp/p1f_task262.onnx")
ok = True
n = 0
for split in ["train", "test", "arc-gen"]:
    for p in ex[split]:
        b = ng.convert_to_numpy(p)
        r = sess.run(["output"], {"input": b["input"]})[0]
        r = (r > 0.0).astype(float)
        n += 1
        if not np.array_equal(r, b["output"]):
            ok = False
            print("MISMATCH", split)
print("ALL OK", ok, "n=", n)
