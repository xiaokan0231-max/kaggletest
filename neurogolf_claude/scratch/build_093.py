import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

H = W = 14
FP16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL

nodes = []
inits = []


def add_init(name, arr):
    inits.append(numpy_helper.from_array(arr, name))


# ---- input collapse: Conv input[1,10,30,30] with [1,10,1,1] weights = color grid ----
add_init("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))

# crop to 14x14 (top-left). Slice on axes 2,3.
add_init("cs", np.array([0, 0], dtype=np.int64))
add_init("ce", np.array([H, W], dtype=np.int64))
add_init("ca", np.array([2, 3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["gf"]))  # [1,1,14,14] f32

# masks
add_init("c5", np.array(5.0, dtype=np.float32))
add_init("c0", np.array(0.0, dtype=np.float32))
nodes.append(helper.make_node("Equal", ["gf", "c5"], ["Wm_b"]))      # wall bool
nodes.append(helper.make_node("Equal", ["gf", "c0"], ["bg_b"]))      # background bool
nodes.append(helper.make_node("Or", ["Wm_b", "bg_b"], ["w_or_bg"]))
nodes.append(helper.make_node("Not", ["w_or_bg"], ["M_b"]))          # marker bool
# fp16 versions for arithmetic
nodes.append(helper.make_node("Cast", ["Wm_b"], ["Wm"], to=FP16))
nodes.append(helper.make_node("Cast", ["M_b"], ["M"], to=FP16))


def stack(axis, tag):
    """markers fall along `axis` onto a wall band. returns fill bool tensor name."""
    a = axis
    # forward cumsum of wall
    cumW = f"cumW{tag}"
    nodes.append(helper.make_node("CumSum", ["Wm", f"ax{a}"], [cumW]))
    # reverse cumsum of wall
    rcumW = f"rcumW{tag}"
    nodes.append(helper.make_node("CumSum", ["Wm", f"ax{a}"], [rcumW], reverse=1))
    # wallAbove = (cumW - Wm) > 0 ; wallBelow = (rcumW - Wm) > 0
    nodes.append(helper.make_node("Sub", [cumW, "Wm"], [f"caW{tag}"]))
    nodes.append(helper.make_node("Sub", [rcumW, "Wm"], [f"crW{tag}"]))
    nodes.append(helper.make_node("Greater", [f"caW{tag}", "zero16"], [f"wAb{tag}"]))  # bool
    nodes.append(helper.make_node("Greater", [f"crW{tag}", "zero16"], [f"wBe{tag}"]))  # bool
    # nonwall fp16
    nodes.append(helper.make_node("Not", ["Wm_b"], [f"nw_b{tag}"]))
    # region_above = nonwall & wallBelow ; region_below = nonwall & wallAbove
    nodes.append(helper.make_node("And", [f"nw_b{tag}", f"wBe{tag}"], [f"rab_b{tag}"]))
    nodes.append(helper.make_node("And", [f"nw_b{tag}", f"wAb{tag}"], [f"rbe_b{tag}"]))
    nodes.append(helper.make_node("Cast", [f"rab_b{tag}"], [f"rab{tag}"], to=FP16))
    nodes.append(helper.make_node("Cast", [f"rbe_b{tag}"], [f"rbe{tag}"], to=FP16))
    # Aup = M * wallBelow(fp16); but use boolean: markers strictly above wall
    nodes.append(helper.make_node("Cast", [f"wBe{tag}"], [f"wBe16{tag}"], to=FP16))
    nodes.append(helper.make_node("Cast", [f"wAb{tag}"], [f"wAb16{tag}"], to=FP16))
    nodes.append(helper.make_node("Mul", ["M", f"wBe16{tag}"], [f"Aup{tag}"]))
    nodes.append(helper.make_node("Mul", ["M", f"wAb16{tag}"], [f"Adn{tag}"]))
    # nAbove = reducesum over axis a (keepdims)
    nodes.append(helper.make_node("ReduceSum", [f"Aup{tag}", f"axarr{a}"], [f"nAb{tag}"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", [f"Adn{tag}", f"axarr{a}"], [f"nBe{tag}"], keepdims=1))
    # distDown = reverse cumsum of region_above ; distUp = forward cumsum region_below
    nodes.append(helper.make_node("CumSum", [f"rab{tag}", f"ax{a}"], [f"distD{tag}"], reverse=1))
    nodes.append(helper.make_node("CumSum", [f"rbe{tag}", f"ax{a}"], [f"distU{tag}"]))
    # fillAbove = (region_above>0) & (distDown <= nAbove)
    nodes.append(helper.make_node("LessOrEqual", [f"distD{tag}", f"nAb{tag}"], [f"leD{tag}"]))
    nodes.append(helper.make_node("LessOrEqual", [f"distU{tag}", f"nBe{tag}"], [f"leU{tag}"]))
    nodes.append(helper.make_node("And", [f"rab_b{tag}", f"leD{tag}"], [f"fAb{tag}"]))
    nodes.append(helper.make_node("And", [f"rbe_b{tag}", f"leU{tag}"], [f"fBe{tag}"]))
    nodes.append(helper.make_node("Or", [f"fAb{tag}", f"fBe{tag}"], [f"fill{tag}"]))  # bool
    return f"fill{tag}"


add_init("ax0", np.array([2], dtype=np.int64))   # axis 2 = rows
add_init("ax1", np.array([3], dtype=np.int64))   # axis 3 = cols
add_init("axarr0", np.array([2], dtype=np.int64))
add_init("axarr1", np.array([3], dtype=np.int64))
add_init("zero16", np.array(0.0, dtype=np.float16))

fH = stack(0, "H")   # vertical fall (horizontal wall)
fV = stack(1, "V")   # horizontal fall (vertical wall)

# orientation: wall is horizontal if some full row of wall exists.
# rowWallCount = reducesum Wm over axis 3 (cols) ; isH if any == W
nodes.append(helper.make_node("ReduceSum", ["Wm", "axarr1"], ["rowWsum"], keepdims=1))  # [1,1,14,1]
add_init("Wf", np.array(float(W), dtype=np.float16))
nodes.append(helper.make_node("GreaterOrEqual", ["rowWsum", "Wf"], ["rowFull"]))  # [1,1,14,1] bool
nodes.append(helper.make_node("ReduceMax", ["rowFull"], ["isH_b"], keepdims=1))  # scalar-ish bool [1,1,1,1]

# select fill = isH ? fH : fV   (Where needs same shape; isH_b broadcasts)
nodes.append(helper.make_node("Where", ["isH_b", fH, fV], ["fillsel"]))  # bool

# answer color grid: wall(5) OR fill(5) -> 5 else 0
nodes.append(helper.make_node("Or", ["Wm_b", "fillsel"], ["five_b"]))  # bool: where color should be 5
# uint8 color grid: 5 where five_b else 0
nodes.append(helper.make_node("Cast", ["five_b"], ["five_u"], to=U8))  # 1/0
add_init("c5u", np.array(5, dtype=np.uint8))
nodes.append(helper.make_node("Mul", ["five_u", "c5u"], ["gm14"]))  # uint8 0 or 5

# pad to 30x30 with 0 (background outside grid stays color 0)
add_init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 30 - W], dtype=np.int64))
add_init("sent", np.array(0, dtype=np.uint8))
nodes.append(helper.make_node("Pad", ["gm14", "pads", "sent"], ["gm30"]))  # [1,1,30,30] uint8

# output one-hot: Equal(gm30, colors[1,10,1,1])
add_init("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))  # bool [1,10,30,30]

inp = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
g = helper.make_graph(nodes, "task093", [inp], [out], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 16)])
m.ir_version = 9
onnx.save(m, "/tmp/b11_task093.onnx")
print("saved")

# quick self-test with onnxruntime
import onnxruntime as ort
import sys
sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng
ng._NEUROGOLF_DIR = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"
d = ng.load_examples(93)
sess = ort.InferenceSession("/tmp/b11_task093.onnx")
ok = bad = 0
for split in ["train", "test", "arc-gen"]:
    for ex in d[split]:
        bm = ng.convert_to_numpy(ex)
        r = sess.run(["output"], {"input": bm["input"]})[0]
        r = (r > 0).astype(float)
        if np.array_equal(r, bm["output"]):
            ok += 1
        else:
            bad += 1
print("selftest ok", ok, "bad", bad)
