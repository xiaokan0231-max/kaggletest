"""Leaner ONNX for task061. Reuse fill for validity. opset10 fp16."""
import math
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

H, G = 18, 30
PERIODS = [4, 5, 6, 7, 8, 9]
F16 = TP.FLOAT16
nodes, inits, vinfo = [], [], []


def add_init(name, a):
    tp = F16 if a.dtype == np.float16 else onnx.helper.np_dtype_to_tensor_dtype(a.dtype)
    inits.append(helper.make_tensor(name, tp, a.shape, a.flatten().tolist())); return name


def vi(n, d, s):
    vinfo.append(helper.make_tensor_value_info(n, d, s)); return n


def node(op, i, o, **a):
    nodes.append(helper.make_node(op, i, o, **a))


# Collapse one-hot -> color grid fp32 [1,1,30,30]
add_init("cw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
node("Conv", ["input", "cw"], ["g30"]); vi("g30", TP.FLOAT, [1, 1, G, G])
# slice 18x18, cast fp16, reshape to [18,18]
add_init("s0", np.array([0, 0], np.int64)); add_init("s18", np.array([H, H], np.int64))
add_init("ax23", np.array([2, 3], np.int64))
node("Slice", ["g30", "s0", "s18", "ax23"], ["g18a"]); vi("g18a", TP.FLOAT, [1, 1, H, H])
node("Cast", ["g18a"], ["g18h"], to=F16); vi("g18h", F16, [1, 1, H, H])
add_init("shp_hh", np.array([H, H], np.int64))
node("Reshape", ["g18h", "shp_hh"], ["g"]); vi("g", F16, [H, H])
add_init("zf", np.array(0, np.float16))
node("Greater", ["g", "zf"], ["g_nz"]); vi("g_nz", TP.BOOL, [H, H])

filled, valid = [], []
for P in PERIODS:
    B = math.ceil(H / P); pad = B * P - H
    if pad > 0:
        node("Pad", ["g"], [f"gp_{P}"], mode="constant", pads=[0, 0, pad, pad], value=0.0)
        vi(f"gp_{P}", F16, [B * P, B * P]); src = f"gp_{P}"
    else:
        src = "g"
    add_init(f"shp4_{P}", np.array([B, P, B, P], np.int64))
    node("Reshape", [src, f"shp4_{P}"], [f"gr_{P}"]); vi(f"gr_{P}", F16, [B, P, B, P])
    node("ReduceMax", [f"gr_{P}"], [f"mx_{P}"], keepdims=0, axes=[0, 2]); vi(f"mx_{P}", F16, [P, P])
    # tile mx [P,P] -> [B*P,B*P] via Tile, slice to 18x18
    add_init(f"reps_{P}", np.array([B, B], np.int64))
    node("Tile", [f"mx_{P}", f"reps_{P}"], [f"til_{P}"]); vi(f"til_{P}", F16, [B * P, B * P])
    add_init(f"fs_{P}", np.array([0, 0], np.int64)); add_init(f"fe_{P}", np.array([H, H], np.int64))
    add_init(f"fax_{P}", np.array([0, 1], np.int64))
    node("Slice", [f"til_{P}", f"fs_{P}", f"fe_{P}", f"fax_{P}"], [f"fill_{P}"]); vi(f"fill_{P}", F16, [H, H])
    filled.append(f"fill_{P}")
    # validity: bad = (fill>g) AND (g>0); valid = sum(bad)==0
    node("Greater", [f"fill_{P}", "g"], [f"gt_{P}"]); vi(f"gt_{P}", TP.BOOL, [H, H])
    node("And", [f"gt_{P}", "g_nz"], [f"bad_{P}"]); vi(f"bad_{P}", TP.BOOL, [H, H])
    node("Cast", [f"bad_{P}"], [f"badf_{P}"], to=F16); vi(f"badf_{P}", F16, [H, H])
    node("ReduceSum", [f"badf_{P}"], [f"bsum_{P}"], keepdims=0, axes=[0, 1]); vi(f"bsum_{P}", F16, [])
    add_init(f"half_{P}", np.array(0.5, np.float16))
    node("Less", [f"bsum_{P}", f"half_{P}"], [f"valid_{P}"]); vi(f"valid_{P}", TP.BOOL, [])
    valid.append(f"valid_{P}")

# select smallest valid P
add_init("zero_hh", np.zeros((H, H), np.float16))
prev_result, prev_any = "zero_hh", None
for idx, P in enumerate(PERIODS):
    if prev_any is None:
        chosen = f"valid_{P}"
    else:
        node("Not", [prev_any], [f"na_{P}"]); vi(f"na_{P}", TP.BOOL, [])
        node("And", [f"valid_{P}", f"na_{P}"], [f"ch_{P}"]); vi(f"ch_{P}", TP.BOOL, [])
        chosen = f"ch_{P}"
    node("Where", [chosen, filled[idx], prev_result], [f"res_{P}"]); vi(f"res_{P}", F16, [H, H])
    prev_result = f"res_{P}"
    if prev_any is None:
        prev_any = f"valid_{P}"
    else:
        node("Or", [prev_any, f"valid_{P}"], [f"any_{P}"]); vi(f"any_{P}", TP.BOOL, [])
        prev_any = f"any_{P}"

# output: reshape [1,1,18,18], pad to 30 with 255 fp16, cast int32, Equal
add_init("shp1118", np.array([1, 1, H, H], np.int64))
node("Reshape", [prev_result, "shp1118"], ["fc4"]); vi("fc4", F16, [1, 1, H, H])
padG = G - H
node("Pad", ["fc4"], ["grid30f"], mode="constant", pads=[0, 0, 0, 0, 0, 0, padG, padG], value=255.0)
vi("grid30f", F16, [1, 1, G, G])
node("Cast", ["grid30f"], ["grid30i"], to=TP.INT32); vi("grid30i", TP.INT32, [1, 1, G, G])
add_init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
node("Equal", ["grid30i", "colors"], ["output"])

graph = helper.make_graph(nodes, "task061",
    [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, G, G])],
    [helper.make_tensor_value_info("output", TP.BOOL, [1, 10, G, G])],
    inits, value_info=vinfo)
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b8_task061.onnx")
import numpy as np
tot = 0
for v in vinfo:
    if v.name in ("input", "output"): continue
    ne = 1
    for d in v.type.tensor_type.shape.dim: ne *= d.dim_value
    tot += ne * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type)).itemsize
print("nodes", len(nodes), "static mem est", tot)
