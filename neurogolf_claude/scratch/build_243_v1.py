"""Build compact ONNX for task243: 4-conn flood-fill from color-1 cells through color-0 cells.
Rule: every 0 (black) cell 4-connected to a 1 cell (through 0 cells) becomes 1.
"""
import numpy as np
import onnx
from onnx import helper as h, TensorProto as TP, numpy_helper as nh

CROP = 18          # all train/test/arc-gen grids are <= 18x18
NSTEPS = 42        # alternating 1D dilation steps (data needs <=39; margin)

def const(name, arr):
    return nh.from_array(np.asarray(arr), name=name)

nodes = []
inits = []

# ---- color grid via Conv ----
w_color = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)   # [0..9]
inits.append(const("wcolor", w_color))
nodes.append(h.make_node("Conv", ["input", "wcolor"], ["g30"], kernel_shape=[1, 1]))

# slice color grid to CROP
inits.append(const("s4", np.array([0, 0, 0, 0], np.int64)))
inits.append(const("eC", np.array([1, 1, CROP, CROP], np.int64)))
inits.append(const("ax4", np.array([0, 1, 2, 3], np.int64)))
nodes.append(h.make_node("Slice", ["g30", "s4", "eC", "ax4"], ["gC_f"]))
nodes.append(h.make_node("Cast", ["gC_f"], ["gC"], to=TP.UINT8))

# ---- free (ch0) and seed (ch1) from input ----
inits.append(const("s1", np.array([0, 1, 0, 0], np.int64)))
inits.append(const("e1", np.array([1, 2, CROP, CROP], np.int64)))
nodes.append(h.make_node("Slice", ["input", "s4", "eC", "ax4"], ["ch0_f"]))
nodes.append(h.make_node("Slice", ["input", "s1", "e1", "ax4"], ["ch1_f"]))
nodes.append(h.make_node("Cast", ["ch0_f"], ["free"], to=TP.UINT8))
nodes.append(h.make_node("Cast", ["ch1_f"], ["seed"], to=TP.UINT8))
nodes.append(h.make_node("Max", ["free", "seed"], ["allowed"]))

# ---- presence (in-grid) mask via Conv with all-ones channel weight ----
w_pres = np.ones((1, 10, 1, 1), dtype=np.float32)
inits.append(const("wpres", w_pres))
nodes.append(h.make_node("Conv", ["input", "wpres"], ["pres30"], kernel_shape=[1, 1]))
nodes.append(h.make_node("Slice", ["pres30", "s4", "eC", "ax4"], ["presC_f"]))
nodes.append(h.make_node("Cast", ["presC_f"], ["ingrid"], to=TP.UINT8))   # 1 in-grid, 0 out

# ---- flood: alternating 1D MaxPool + Mul(allowed) ----
cur = "seed"
for i in range(NSTEPS):
    if i % 2 == 0:
        ks, pads = [3, 1], [1, 0, 1, 0]
    else:
        ks, pads = [1, 3], [0, 1, 0, 1]
    p = f"p{i}"
    d = f"d{i}"
    nodes.append(h.make_node("MaxPool", [cur], [p], kernel_shape=ks, pads=pads, strides=[1, 1]))
    nodes.append(h.make_node("Mul", [p, "allowed"], [d]))
    cur = d
reached = cur

# ---- final color grid: gm = gC*(1-reached) + reached ----
inits.append(const("one_u", np.array(1, np.uint8)))
nodes.append(h.make_node("Sub", ["one_u", reached], ["notr"]))
nodes.append(h.make_node("Mul", ["gC", "notr"], ["gkeep"]))
nodes.append(h.make_node("Add", ["gkeep", reached], ["gm0"]))

# ---- mask out-of-grid cells (inside the crop) to sentinel 255 ----
# gm = ingrid ? gm0 : 255  ; using arithmetic: gm0*ingrid + 255*(1-ingrid)
nodes.append(h.make_node("Sub", ["one_u", "ingrid"], ["notgrid"]))
nodes.append(h.make_node("Mul", ["gm0", "ingrid"], ["gm_in"]))
inits.append(const("c255", np.array(255, np.uint8)))
nodes.append(h.make_node("Mul", ["notgrid", "c255"], ["gm_out"]))
nodes.append(h.make_node("Add", ["gm_in", "gm_out"], ["gm"]))

# ---- pad gm back to 30x30 with sentinel 255 (out-of-grid) ----
pad_amt = 30 - CROP
inits.append(const("padv", np.array([0, 0, 0, 0, 0, 0, pad_amt, pad_amt], np.int64)))
inits.append(const("pad255", np.array(255, np.uint8)))
nodes.append(h.make_node("Pad", ["gm", "padv", "pad255"], ["gm30"], mode="constant"))

# ---- Equal vs colors -> output one-hot [1,10,30,30] (EXCLUDED graph output) ----
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
inits.append(const("colors", colors))
nodes.append(h.make_node("Equal", ["gm30", "colors"], ["output"]))

inp = h.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
out = h.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])
graph = h.make_graph(nodes, "task243", [inp], [out], inits)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b9_task243.onnx")
print("saved, nodes:", len(nodes), "inits:", len(inits))
