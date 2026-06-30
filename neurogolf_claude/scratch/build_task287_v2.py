import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import sys
sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng

IR_VERSION = 10
OPSET = [helper.make_opsetid("", 13)]

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])

nodes = []
inits = []

# 1) Conv collapse one-hot -> f32 color grid g [1,1,30,30]  (3600B, irreducible)
w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(helper.make_tensor("W", TensorProto.FLOAT, [1, 10, 1, 1], w.flatten().tolist()))
nodes.append(helper.make_node("Conv", ["input", "W"], ["g"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

# 2) cast whole grid to uint8 once: gu8 [1,1,30,30] (900B), replaces the f32 slice
nodes.append(helper.make_node("Cast", ["g"], ["gu8"], to=TensorProto.UINT8))

# 3) slice real 16x16 region (uint8, 256B)
inits.append(helper.make_tensor("starts16", TensorProto.INT64, [2], [0, 0]))
inits.append(helper.make_tensor("ends16", TensorProto.INT64, [2], [16, 16]))
inits.append(helper.make_tensor("axesHW", TensorProto.INT64, [2], [2, 3]))
nodes.append(helper.make_node("Slice", ["gu8", "starts16", "ends16", "axesHW"], ["g16"]))

# 4) rot180 of g16 via negative-step slice (256B)
inits.append(helper.make_tensor("rstarts", TensorProto.INT64, [2], [15, 15]))
inits.append(helper.make_tensor("rends", TensorProto.INT64, [2], [-17, -17]))
inits.append(helper.make_tensor("rsteps", TensorProto.INT64, [2], [-1, -1]))
nodes.append(helper.make_node("Slice", ["g16", "rstarts", "rends", "axesHW", "rsteps"], ["rot"]))

# 5) mask = (g16 == 4)  (bool 256B)
inits.append(helper.make_tensor("c4", TensorProto.UINT8, [], [4]))
nodes.append(helper.make_node("Equal", ["g16", "c4"], ["mask4"]))

# 6) gm16 = where(mask4, rot, g16)  (uint8 256B)
nodes.append(helper.make_node("Where", ["mask4", "rot", "g16"], ["gm16"]))

# 7) pad to 30x30 with sentinel 255 (uint8 900B)
inits.append(helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 0, 0, 0, 0, 14, 14]))
inits.append(helper.make_tensor("padval", TensorProto.UINT8, [], [255]))
nodes.append(helper.make_node("Pad", ["gm16", "pads", "padval"], ["gm"], mode="constant"))

# 8) Equal(gm, colors[1,10,1,1]) -> output one-hot bool [1,10,30,30] (graph output, excluded)
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(helper.make_tensor("colors", TensorProto.UINT8, [1, 10, 1, 1], colors.flatten().tolist()))
nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

graph = helper.make_graph(nodes, "task287", [inp], [out], inits)
model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b8_task287.onnx")
print("saved")

d = ng.load_examples(287)
sess = ort.InferenceSession("/tmp/b8_task287.onnx")
bad = 0; tot = 0
for split in ["train", "test", "arc-gen"]:
    for ex in d[split]:
        b = ng.convert_to_numpy(ex)
        r = sess.run(["output"], {"input": b["input"]})[0]
        if not np.array_equal((r > 0.0).astype(float), b["output"]):
            bad += 1
        tot += 1
print("ORT tot", tot, "bad", bad)
