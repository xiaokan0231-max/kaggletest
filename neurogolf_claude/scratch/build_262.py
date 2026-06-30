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

# 1) collapse one-hot input -> color grid g [1,1,30,30] f32
w_collapse = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(const("w_collapse", w_collapse))
nodes.append(helper.make_node("Conv", ["input", "w_collapse"], ["g"], kernel_shape=[1, 1]))

# 2) crop to top-left 3x3 -> g3 [1,1,3,3]
inits.append(const("cs", np.array([0, 0], dtype=np.int64)))
inits.append(const("ce", np.array([3, 3], dtype=np.int64)))
inits.append(const("cax", np.array([2, 3], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["g", "cs", "ce", "cax"], ["g3"]))

# 3) is5 = (g3 == 5) -> fp16
inits.append(const("five", np.array(5.0, dtype=np.float32)))
nodes.append(helper.make_node("Equal", ["g3", "five"], ["is5b"]))
nodes.append(helper.make_node("Cast", ["is5b"], ["is5"], to=TensorProto.FLOAT16))

# 4) cv = Conv(is5, [2,4,3]) -> [1,1,3,1] fp16, mapped color per row
w_row = np.array([2.0, 4.0, 3.0], dtype=np.float16).reshape(1, 1, 1, 3)
inits.append(const("w_row", w_row))
nodes.append(helper.make_node("Conv", ["is5", "w_row"], ["cv"], kernel_shape=[1, 3]))

# 5) tile to 3 cols -> c3 [1,1,3,3]
nodes.append(helper.make_node("Concat", ["cv", "cv", "cv"], ["c3"], axis=3))
# cast to uint8
nodes.append(helper.make_node("Cast", ["c3"], ["c3u"], to=TensorProto.UINT8))

# 6) pad to [1,1,30,30] with sentinel 255 outside the 3x3
pads = np.array([0, 0, 0, 0, 0, 0, H - 3, W - 3], dtype=np.int64)
inits.append(const("pads", pads))
inits.append(const("pad255", np.array(255, dtype=np.uint8)))
nodes.append(helper.make_node("Pad", ["c3u", "pads", "pad255"], ["gm"], mode="constant"))

# 7) Equal-as-output
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(const("colors", colors))
nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

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
