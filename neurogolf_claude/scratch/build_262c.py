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

# 1) crop input spatially to top-left 3x3 -> in3 [1,10,3,3] f32 (360B)
inits.append(const("cs", np.array([0, 0], dtype=np.int64)))
inits.append(const("ce", np.array([3, 3], dtype=np.int64)))
inits.append(const("cax", np.array([2, 3], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["input", "cs", "ce", "cax"], ["in3"]))

# 2) collapse one-hot -> color grid g3 [1,1,3,3] f32 (36B)
w_collapse = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(const("w_collapse", w_collapse))
nodes.append(helper.make_node("Conv", ["in3", "w_collapse"], ["g3"], kernel_shape=[1, 1]))

# 3) is5 = (g3 == 5) -> fp16
inits.append(const("five", np.array(5.0, dtype=np.float32)))
nodes.append(helper.make_node("Equal", ["g3", "five"], ["is5b"]))
nodes.append(helper.make_node("Cast", ["is5b"], ["is5"], to=TensorProto.FLOAT16))

# 4) cv = Conv(is5, [2,4,3]) -> [1,1,3,1] fp16 mapped color per row
w_row = np.array([2.0, 4.0, 3.0], dtype=np.float16).reshape(1, 1, 1, 3)
inits.append(const("w_row", w_row))
nodes.append(helper.make_node("Conv", ["is5", "w_row"], ["cv"], kernel_shape=[1, 3]))

# 5) tile to 3 cols -> c3 [1,1,3,3], cast uint8
nodes.append(helper.make_node("Concat", ["cv", "cv", "cv"], ["c3"], axis=3))
nodes.append(helper.make_node("Cast", ["c3"], ["c3u"], to=TensorProto.UINT8))

# 6) small one-hot for 3x3 -> oh3 [1,10,3,3] bool (90B interior)
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(const("colors", colors))
nodes.append(helper.make_node("Equal", ["c3u", "colors"], ["oh3"]))

# 7) Pad-as-output: pad spatial dims (axes 2,3) bottom/right by 27 with False -> output [1,10,30,30]
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
