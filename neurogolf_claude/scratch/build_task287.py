import numpy as np
import onnx
from onnx import helper, TensorProto

IR_VERSION = 10
OPSET = [helper.make_opsetid("", 13)]

# ---- input/output value infos ----
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])

nodes = []
inits = []

# 1) collapse one-hot -> single color grid g [1,1,30,30] f32 via Conv weight [1,10,1,1]=[0..9]
w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
inits.append(helper.make_tensor("W", TensorProto.FLOAT, [1, 10, 1, 1], w.flatten().tolist()))
nodes.append(helper.make_node("Conv", ["input", "W"], ["g"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

# 2) slice to the real 16x16 region: g16 [1,1,16,16]
inits.append(helper.make_tensor("starts16", TensorProto.INT64, [2], [0, 0]))
inits.append(helper.make_tensor("ends16", TensorProto.INT64, [2], [16, 16]))
inits.append(helper.make_tensor("axesHW", TensorProto.INT64, [2], [2, 3]))
nodes.append(helper.make_node("Slice", ["g", "starts16", "ends16", "axesHW"], ["g16f"]))

# cast to uint8
nodes.append(helper.make_node("Cast", ["g16f"], ["g16"], to=TensorProto.UINT8))

# 3) rot180 of g16 via Slice with negative steps over H and W
inits.append(helper.make_tensor("rstarts", TensorProto.INT64, [2], [15, 15]))
inits.append(helper.make_tensor("rends", TensorProto.INT64, [2], [-17, -17]))  # exclusive, go to before 0
inits.append(helper.make_tensor("rsteps", TensorProto.INT64, [2], [-1, -1]))
nodes.append(helper.make_node("Slice", ["g16", "rstarts", "rends", "axesHW", "rsteps"], ["rot"]))

# 4) mask = (g16 == 4)
inits.append(helper.make_tensor("c4", TensorProto.UINT8, [], [4]))
nodes.append(helper.make_node("Equal", ["g16", "c4"], ["mask4"]))

# 5) gm16 = where(mask4, rot, g16)  [1,1,16,16] uint8 (restored colors 1..9)
nodes.append(helper.make_node("Where", ["mask4", "rot", "g16"], ["gm16"]))

# 6) pad gm16 -> [1,1,30,30] with sentinel 255 (out-of-grid cells match no color 0..9)
# pads format for opset>=11: [b0,c0,h0,w0, b1,c1,h1,w1]
inits.append(helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 0, 0, 0, 0, 14, 14]))
inits.append(helper.make_tensor("padval", TensorProto.UINT8, [], [255]))
nodes.append(helper.make_node("Pad", ["gm16", "pads", "padval"], ["gm"], mode="constant"))

# 7) colors [1,10,1,1] uint8 = 0..9 ; Equal(gm, colors) broadcasts -> output one-hot bool [1,10,30,30]
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(helper.make_tensor("colors", TensorProto.UINT8, [1, 10, 1, 1], colors.flatten().tolist()))
nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))

graph = helper.make_graph(nodes, "task287", [inp], [out], inits)
model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b8_task287.onnx")
print("saved /tmp/b8_task287.onnx")

# quick local functional check with onnxruntime
import onnxruntime as ort
import sys
sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng
d = ng.load_examples(287)
sess = ort.InferenceSession("/tmp/b8_task287.onnx")
bad = 0
tot = 0
for split in ["train", "test", "arc-gen"]:
    for ex in d[split]:
        b = ng.convert_to_numpy(ex)
        r = sess.run(["output"], {"input": b["input"]})[0]
        pred = (r > 0.0).astype(float)
        if not np.array_equal(pred, b["output"]):
            bad += 1
        tot += 1
print("ORT check: tot", tot, "bad", bad)
