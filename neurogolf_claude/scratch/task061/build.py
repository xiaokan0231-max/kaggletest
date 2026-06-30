"""Build compact ONNX for task061: periodic pattern reconstruction. opset 10, fp16 pipeline."""
import math
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

H = 18
G = 30
PERIODS = [4, 5, 6, 7, 8, 9]
F16 = TP.FLOAT16

nodes, inits, vinfo = [], [], []


def add_init(name, np_arr):
    tp = onnx.helper.np_dtype_to_tensor_dtype(np_arr.dtype) if np_arr.dtype != np.float16 else F16
    inits.append(helper.make_tensor(name, tp, np_arr.shape, np_arr.flatten().tolist()))
    return name


def vi(name, dtype, shape):
    vinfo.append(helper.make_tensor_value_info(name, dtype, shape))
    return name


def node(op, ins, outs, **attrs):
    nodes.append(helper.make_node(op, ins, outs, **attrs))


# Collapse one-hot -> color grid [1,1,30,30] fp32 via Conv weight [1,10,1,1]=0..9
add_init("collapse_w", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
node("Conv", ["input", "collapse_w"], ["g30f32"]); vi("g30f32", TP.FLOAT, [1, 1, G, G])

# Slice top-left 18x18 (still fp32), then cast to fp16
add_init("s0", np.array([0, 0], np.int64))
add_init("s18", np.array([H, H], np.int64))
add_init("ax23", np.array([2, 3], np.int64))
node("Slice", ["g30f32", "s0", "s18", "ax23"], ["g18c"]); vi("g18c", TP.FLOAT, [1, 1, H, H])
add_init("shp_hh", np.array([H, H], np.int64))
node("Reshape", ["g18c", "shp_hh"], ["g32"]); vi("g32", TP.FLOAT, [H, H])
node("Cast", ["g32"], ["g"], to=F16); vi("g", F16, [H, H])

add_init("zf", np.array(0, np.float16))

filled_names, valid_names = [], []
for P in PERIODS:
    L = H - P
    # horizontal
    add_init(f"ax1_{P}", np.array([1], np.int64))
    add_init(f"hsa_{P}", np.array([0], np.int64)); add_init(f"hea_{P}", np.array([L], np.int64))
    node("Slice", ["g", f"hsa_{P}", f"hea_{P}", f"ax1_{P}"], [f"ha_{P}"]); vi(f"ha_{P}", F16, [H, L])
    add_init(f"hsb_{P}", np.array([P], np.int64)); add_init(f"heb_{P}", np.array([H], np.int64))
    node("Slice", ["g", f"hsb_{P}", f"heb_{P}", f"ax1_{P}"], [f"hb_{P}"]); vi(f"hb_{P}", F16, [H, L])
    node("Greater", [f"ha_{P}", "zf"], [f"han_{P}"]); vi(f"han_{P}", TP.BOOL, [H, L])
    node("Greater", [f"hb_{P}", "zf"], [f"hbn_{P}"]); vi(f"hbn_{P}", TP.BOOL, [H, L])
    # a!=b -> use Less/Greater on fp16 (Equal unsupported on fp16). neq = (a>b) OR (b>a)
    node("Greater", [f"ha_{P}", f"hb_{P}"], [f"hgt_{P}"]); vi(f"hgt_{P}", TP.BOOL, [H, L])
    node("Less", [f"ha_{P}", f"hb_{P}"], [f"hlt_{P}"]); vi(f"hlt_{P}", TP.BOOL, [H, L])
    node("Or", [f"hgt_{P}", f"hlt_{P}"], [f"hne_{P}"]); vi(f"hne_{P}", TP.BOOL, [H, L])
    node("And", [f"han_{P}", f"hbn_{P}"], [f"hboth_{P}"]); vi(f"hboth_{P}", TP.BOOL, [H, L])
    node("And", [f"hne_{P}", f"hboth_{P}"], [f"hbad_{P}"]); vi(f"hbad_{P}", TP.BOOL, [H, L])
    # vertical
    add_init(f"ax0_{P}", np.array([0], np.int64))
    add_init(f"vsa_{P}", np.array([0], np.int64)); add_init(f"vea_{P}", np.array([L], np.int64))
    node("Slice", ["g", f"vsa_{P}", f"vea_{P}", f"ax0_{P}"], [f"va_{P}"]); vi(f"va_{P}", F16, [L, H])
    add_init(f"vsb_{P}", np.array([P], np.int64)); add_init(f"veb_{P}", np.array([H], np.int64))
    node("Slice", ["g", f"vsb_{P}", f"veb_{P}", f"ax0_{P}"], [f"vb_{P}"]); vi(f"vb_{P}", F16, [L, H])
    node("Greater", [f"va_{P}", "zf"], [f"van_{P}"]); vi(f"van_{P}", TP.BOOL, [L, H])
    node("Greater", [f"vb_{P}", "zf"], [f"vbn_{P}"]); vi(f"vbn_{P}", TP.BOOL, [L, H])
    node("Greater", [f"va_{P}", f"vb_{P}"], [f"vgt_{P}"]); vi(f"vgt_{P}", TP.BOOL, [L, H])
    node("Less", [f"va_{P}", f"vb_{P}"], [f"vlt_{P}"]); vi(f"vlt_{P}", TP.BOOL, [L, H])
    node("Or", [f"vgt_{P}", f"vlt_{P}"], [f"vne_{P}"]); vi(f"vne_{P}", TP.BOOL, [L, H])
    node("And", [f"van_{P}", f"vbn_{P}"], [f"vboth_{P}"]); vi(f"vboth_{P}", TP.BOOL, [L, H])
    node("And", [f"vne_{P}", f"vboth_{P}"], [f"vbad_{P}"]); vi(f"vbad_{P}", TP.BOOL, [L, H])
    # any bad? cast to fp16, reducesum
    node("Cast", [f"hbad_{P}"], [f"hbf_{P}"], to=F16); vi(f"hbf_{P}", F16, [H, L])
    node("Cast", [f"vbad_{P}"], [f"vbf_{P}"], to=F16); vi(f"vbf_{P}", F16, [L, H])
    node("ReduceSum", [f"hbf_{P}"], [f"hsum_{P}"], keepdims=0, axes=[0, 1]); vi(f"hsum_{P}", F16, [])
    node("ReduceSum", [f"vbf_{P}"], [f"vsum_{P}"], keepdims=0, axes=[0, 1]); vi(f"vsum_{P}", F16, [])
    node("Add", [f"hsum_{P}", f"vsum_{P}"], [f"tot_{P}"]); vi(f"tot_{P}", F16, [])
    # valid if tot < 0.5
    add_init(f"half_{P}", np.array(0.5, np.float16))
    node("Less", [f"tot_{P}", f"half_{P}"], [f"valid_{P}"]); vi(f"valid_{P}", TP.BOOL, [])
    valid_names.append(f"valid_{P}")

    # fill: pad to B*P, reshape [B,P,B,P], reducemax axes 0,2, expand, reshape, slice 18x18
    B = math.ceil(H / P)
    pad = B * P - H
    if pad > 0:
        node("Pad", ["g"], [f"gp_{P}"], mode="constant", pads=[0, 0, pad, pad], value=0.0)
        vi(f"gp_{P}", F16, [B * P, B * P]); src = f"gp_{P}"
    else:
        src = "g"
    add_init(f"shp4_{P}", np.array([B, P, B, P], np.int64))
    node("Reshape", [src, f"shp4_{P}"], [f"gr_{P}"]); vi(f"gr_{P}", F16, [B, P, B, P])
    node("ReduceMax", [f"gr_{P}"], [f"mx_{P}"], keepdims=1, axes=[0, 2]); vi(f"mx_{P}", F16, [1, P, 1, P])
    add_init(f"exp_{P}", np.array([B, P, B, P], np.int64))
    node("Expand", [f"mx_{P}", f"exp_{P}"], [f"til_{P}"]); vi(f"til_{P}", F16, [B, P, B, P])
    add_init(f"shpbb_{P}", np.array([B * P, B * P], np.int64))
    node("Reshape", [f"til_{P}", f"shpbb_{P}"], [f"flat_{P}"]); vi(f"flat_{P}", F16, [B * P, B * P])
    add_init(f"fs_{P}", np.array([0, 0], np.int64)); add_init(f"fe_{P}", np.array([H, H], np.int64))
    add_init(f"fax_{P}", np.array([0, 1], np.int64))
    node("Slice", [f"flat_{P}", f"fs_{P}", f"fe_{P}", f"fax_{P}"], [f"fill_{P}"]); vi(f"fill_{P}", F16, [H, H])
    filled_names.append(f"fill_{P}")

# select smallest valid
add_init("zero_hh", np.zeros((H, H), np.float16))
prev_result = "zero_hh"
prev_any = None
for idx, P in enumerate(PERIODS):
    if prev_any is None:
        chosen = f"valid_{P}"
    else:
        node("Not", [prev_any], [f"notany_{P}"]); vi(f"notany_{P}", TP.BOOL, [])
        node("And", [f"valid_{P}", f"notany_{P}"], [f"chosen_{P}"]); vi(f"chosen_{P}", TP.BOOL, [])
        chosen = f"chosen_{P}"
    node("Where", [chosen, filled_names[idx], prev_result], [f"res_{P}"]); vi(f"res_{P}", F16, [H, H])
    prev_result = f"res_{P}"
    if prev_any is None:
        prev_any = f"valid_{P}"
    else:
        node("Or", [prev_any, f"valid_{P}"], [f"any_{P}"]); vi(f"any_{P}", TP.BOOL, [])
        prev_any = f"any_{P}"

# build output. reshape final to [1,1,18,18], pad to 30 with sentinel 255 (fp16), cast int32, Equal
add_init("shp1118", np.array([1, 1, H, H], np.int64))
node("Reshape", [prev_result, "shp1118"], ["fc4"]); vi("fc4", F16, [1, 1, H, H])
padG = G - H
node("Pad", ["fc4"], ["grid30f"], mode="constant", pads=[0, 0, 0, 0, 0, 0, padG, padG], value=255.0)
vi("grid30f", F16, [1, 1, G, G])
node("Cast", ["grid30f"], ["grid30"], to=TP.INT32); vi("grid30", TP.INT32, [1, 1, G, G])
add_init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
node("Equal", ["grid30", "colors"], ["output"])

graph = helper.make_graph(
    [n for n in nodes if n is not None], "task061",
    [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, G, G])],
    [helper.make_tensor_value_info("output", TP.BOOL, [1, 10, G, G])],
    inits, value_info=vinfo,
)
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b8_task061.onnx")
print("saved nodes", len([n for n in nodes if n]), "inits", len(inits))
