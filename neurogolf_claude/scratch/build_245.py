"""task245: move the 5x5 2-shape so its bbox-center aligns with the 3-box center.
Only colors {0,2,3} occur -> slice channels 0..3 of input (skip Conv). Window 10x10.
shift via Mr @ in2 @ McT, McT=Equal(D,-dc). Output = Equal(gm[1,1,30,30], colors).
"""
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H

W = 10
nodes = []
inits = []


def const(name, arr):
    inits.append(onnx.numpy_helper.from_array(np.asarray(arr), name))
    return name


# ---- slice channel 0 (window) and channels 2..3 (window) separately ----
const("half", np.array(0.5, np.float32))
const("s0beg", np.array([0, 0, 0, 0], np.int64))
const("s0end", np.array([1, 1, 10, 10], np.int64))
nodes.append(H.make_node("Slice", ["input", "s0beg", "s0end"], ["s0"]))  # [1,1,10,10] f32
const("s23beg", np.array([0, 2, 0, 0], np.int64))
const("s23end", np.array([1, 4, 10, 10], np.int64))
nodes.append(H.make_node("Slice", ["input", "s23beg", "s23end"], ["s23"]))  # [1,2,10,10] f32

# is0 mask, and is2/is3 from one bool tensor s23b
nodes.append(H.make_node("Greater", ["s0", "half"], ["is0_b"]))  # [1,1,10,10] bool
nodes.append(H.make_node("Greater", ["s23", "half"], ["s23b"]))  # [1,2,10,10] bool
const("c2beg", np.array([0, 0, 0, 0], np.int64))
const("c2end", np.array([1, 1, 10, 10], np.int64))
const("c3beg", np.array([0, 1, 0, 0], np.int64))
const("c3end", np.array([1, 2, 10, 10], np.int64))
nodes.append(H.make_node("Slice", ["s23b", "c2beg", "c2end"], ["is2_b"]))  # [1,1,10,10] bool
nodes.append(H.make_node("Slice", ["s23b", "c3beg", "c3end"], ["is3_b"]))

# presence = is0 OR is2 OR is3
nodes.append(H.make_node("Or", ["is0_b", "is2_b"], ["p02"]))
nodes.append(H.make_node("Or", ["p02", "is3_b"], ["pres_b"]))  # [1,1,10,10] bool

# squeeze + fp16 for matmul/reduce
const("ax01", np.array([0, 1], np.int64))
nodes.append(H.make_node("Squeeze", ["is2_b", "ax01"], ["is2_2d_b"]))
nodes.append(H.make_node("Squeeze", ["is3_b", "ax01"], ["is3_2d_b"]))
nodes.append(H.make_node("Cast", ["is2_2d_b"], ["is2_2d"], to=TP.FLOAT16))   # [10,10]
nodes.append(H.make_node("Cast", ["is3_2d_b"], ["is3_2d"], to=TP.FLOAT16))

# ---- per-row/col has-mask -> [10]; 3-box always 7x7, 2-shape always 5x5 so
#      dr = min3r - min2r + 1 ;  dc = min3c - min2c + 1   (mins only) ----
const("ax0", np.array([0], np.int64))
const("ax1b", np.array([1], np.int64))
nodes.append(H.make_node("ReduceMax", ["is2_2d", "ax1b"], ["row2"], keepdims=0))
nodes.append(H.make_node("ReduceMax", ["is2_2d", "ax0"], ["col2"], keepdims=0))
nodes.append(H.make_node("ReduceMax", ["is3_2d", "ax1b"], ["row3"], keepdims=0))
nodes.append(H.make_node("ReduceMax", ["is3_2d", "ax0"], ["col3"], keepdims=0))

const("idx", np.arange(W, dtype=np.float16))
const("bigpos", np.array(100.0, np.float16))


def minidx(rowhas, prefix):
    b = prefix + "_b"
    nodes.append(H.make_node("Cast", [rowhas], [b], to=TP.BOOL))
    nodes.append(H.make_node("Where", [b, "idx", "bigpos"], [prefix + "_mn_v"]))
    nodes.append(H.make_node("ReduceMin", [prefix + "_mn_v"], [prefix + "_mn"], keepdims=0))
    return prefix + "_mn"


m2r = minidx("row2", "r2")  # min row of 2-shape
m2c = minidx("col2", "c2")
m3r = minidx("row3", "r3")
m3c = minidx("col3", "c3")

const("one_f16", np.array(1.0, np.float16))
nodes.append(H.make_node("Sub", [m3r, m2r], ["dr0"]))
nodes.append(H.make_node("Add", ["dr0", "one_f16"], ["dr"]))   # min3r-min2r+1
nodes.append(H.make_node("Sub", [m3c, m2c], ["dc0"]))
nodes.append(H.make_node("Add", ["dc0", "one_f16"], ["dc"]))

# ---- shift matrices without a shared [10,10] D ----
# Mr[a,b]=(a-b==dr)=(a==b+dr); McT[a,b]=(a-b==-dc)=(a==b-dc)
const("idx_col", np.arange(W, dtype=np.float16).reshape(W, 1))
const("idx_row", np.arange(W, dtype=np.float16).reshape(1, W))
nodes.append(H.make_node("Add", ["idx_row", "dr"], ["rhs_r"]))   # [1,10]
nodes.append(H.make_node("Sub", ["idx_row", "dc"], ["rhs_c"]))   # [1,10]
nodes.append(H.make_node("Equal", ["idx_col", "rhs_r"], ["Mr_b"]))   # [10,10]
nodes.append(H.make_node("Equal", ["idx_col", "rhs_c"], ["McT_b"]))
nodes.append(H.make_node("Cast", ["Mr_b"], ["Mr"], to=TP.FLOAT16))
nodes.append(H.make_node("Cast", ["McT_b"], ["McT"], to=TP.FLOAT16))

nodes.append(H.make_node("MatMul", ["Mr", "is2_2d"], ["tmp1"]))
nodes.append(H.make_node("MatMul", ["tmp1", "McT"], ["out2"]))   # [10,10] f16
const("half16", np.array(0.5, np.float16))
nodes.append(H.make_node("Greater", ["out2", "half16"], ["out2_b2d"]))  # [10,10] bool
const("shp4", np.array([1, 1, W, W], np.int64))
nodes.append(H.make_node("Reshape", ["out2_b2d", "shp4"], ["out2_b"]))  # [1,1,10,10] bool

# ---- gm10 uint8 ----
const("u255", np.array(255, np.uint8))
const("u3", np.array(3, np.uint8))
const("u2", np.array(2, np.uint8))
const("u0", np.array(0, np.uint8))
nodes.append(H.make_node("Where", ["out2_b", "u2", "u0"], ["lay2"]))
nodes.append(H.make_node("Where", ["is3_b", "u3", "lay2"], ["lay3"]))
nodes.append(H.make_node("Where", ["pres_b", "lay3", "u255"], ["gm10"]))  # [1,1,10,10] uint8

# pad to [1,1,30,30] with 255
const("pads", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64))
nodes.append(H.make_node("Pad", ["gm10", "pads", "u255"], ["gm"]))  # [1,1,30,30] uint8

const("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
nodes.append(H.make_node("Equal", ["gm", "colors"], ["output"]))  # [1,10,30,30] bool

graph = H.make_graph(
    nodes, "task245",
    [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
    [H.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])],
    inits,
)
model = H.make_model(graph, opset_imports=[H.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b15_task245.onnx")
print("saved")

import onnxruntime as ort
import sys
sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng
sess = ort.InferenceSession("/tmp/b15_task245.onnx")
d = ng.load_examples(245)
bad = 0
tot = 0
for split in ["train", "test", "arc-gen"]:
    for ex in d[split]:
        bm = ng.convert_to_numpy(ex)
        out = sess.run(["output"], {"input": bm["input"]})[0]
        out = (out > 0.0).astype(np.float32)
        if not np.array_equal(out, bm["output"]):
            bad += 1
        tot += 1
print("self-test bad", bad, "/", tot)
