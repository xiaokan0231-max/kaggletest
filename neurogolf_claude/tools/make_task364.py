"""Build compact ONNX for task364.

Rule (verified on all 266 train+test+arc-gen examples):
  Input grid has only color 3 (foreground) on background 0. Each 4-connected
  foreground component is a 1-pixel-wide path. Recolor the whole component by:
    - branch cell present (degree>=3, T/Y junction)        -> color 2
    - else exactly 2 corner cells (U/S/C/Z shape)          -> color 6
    - else exactly 1 corner cell (L shape)                 -> color 1
  Corner cell = degree-2 fg cell with one vertical AND one horizontal neighbor.

ONNX algorithm (no labeling, no Loop):
  fg = input[:,3]. Components never touch diagonally (verified) so a 3x3 MaxPool
  masked by fg propagates strictly inside one component (~2 cells / iter).
  deg via 4 neighbor shifts; branch=fg*(deg>=3); corner=fg*(deg==2)*hasV*hasH.
  Wave 1 (float16 cmax): propagate max corner-index. A component has 2 corners iff
    some corner cell has idx < cmax -> 'second'.
  Wave 2 (uint8 typeval): per cell type = 2*branch + 1*second; MaxPool propagates the
    component type (branch=2 beats 2corner=1 beats L=0).
  Decode: m2=fg&(type==2), m6=fg&(type==1), m1=fg&(type==0).
  Output one-hot [1,10,30,30]: ch0=active&~fg, ch1=m1, ch2=m2, ch6=m6 via fixed Conv.

Memory-minimised: the two propagation waves run in float16 / uint8 (2 / 1 byte).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/wf3_task364.onnx"
H = W = 30
ITER = 7

nodes, inits = [], []

def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name

def n(op, i, o, **kw):
    if isinstance(o, str): o = [o]
    if isinstance(i, str): i = [i]
    nodes.append(helper.make_node(op, i, o, **kw)); return o[0]

# ---------- constants ----------
C("ax1", np.array([1], np.int64))
C("zero", np.array(0.0, np.float32))
C("one", np.array(1.0, np.float32))
C("two", np.array(2.0, np.float32))
# select channel 3: weight [1,10,1,1] one-hot
w_ch3 = np.zeros((1, 10, 1, 1), np.float32); w_ch3[0, 3, 0, 0] = 1.0
C("w_ch3", w_ch3)
# pixel index 1..900 (float32)
idx32 = (np.arange(H * W).reshape(1, 1, H, W) + 1).astype(np.float32)
C("idx32", idx32)
C("half", np.array(0.5, np.float32))
# output assembler weight [10,4,1,1]: (bg,m1,m2,m6) -> channels (0,1,2,6)
w_out = np.zeros((10, 4, 1, 1), np.float32)
w_out[0, 0, 0, 0] = 1.0; w_out[1, 1, 0, 0] = 1.0
w_out[2, 2, 0, 0] = 1.0; w_out[6, 3, 0, 0] = 1.0
C("w_out", w_out)

# ---------- fg + active ----------
n("Conv", ["input", "w_ch3"], "fg")                       # [1,1,30,30] 0/1 f32
n("ReduceSum", ["input", "ax1"], "active", keepdims=1)    # 1 inside real grid

# ---------- 4-neighbor shifts ----------
def shift(src, dy, dx, name):
    pt, pb = max(dy, 0), max(-dy, 0)
    pl, pr = max(dx, 0), max(-dx, 0)
    C(name + "_p", np.array([0, 0, pt, pl, 0, 0, pb, pr], np.int64))
    n("Pad", [src, name + "_p"], name + "_pad")
    C(name + "_s", np.array([pb, pr], np.int64))
    C(name + "_e", np.array([pb + H, pr + W], np.int64))
    C(name + "_a", np.array([2, 3], np.int64))
    n("Slice", [name + "_pad", name + "_s", name + "_e", name + "_a"], name)
    return name

shift("fg", -1, 0, "up"); shift("fg", 1, 0, "dn")
shift("fg", 0, -1, "lf"); shift("fg", 0, 1, "rt")
n("Add", ["up", "dn"], "vsum")
n("Add", ["lf", "rt"], "hsum")
n("Add", ["vsum", "hsum"], "deg")

# branch = fg*(deg>2)
n("Greater", ["deg", "two"], "br_b"); n("Cast", ["br_b"], "br_f", to=TensorProto.FLOAT)
n("Mul", ["fg", "br_f"], "branch")                        # f32 0/1
# corner = fg*(deg==2)*(vsum>0)*(hsum>0)
n("Equal", ["deg", "two"], "d2_b"); n("Cast", ["d2_b"], "d2", to=TensorProto.FLOAT)
n("Greater", ["vsum", "zero"], "hv_b"); n("Cast", ["hv_b"], "hv", to=TensorProto.FLOAT)
n("Greater", ["hsum", "zero"], "hh_b"); n("Cast", ["hh_b"], "hh", to=TensorProto.FLOAT)
n("Mul", ["fg", "d2"], "c0"); n("Mul", ["c0", "hv"], "c1"); n("Mul", ["c1", "hh"], "corner")

# ---------- wave 1: max corner index (float32) ----------
n("Mul", ["corner", "idx32"], "cmax")                     # f32 seed
for k in range(ITER):
    n("MaxPool", ["cmax"], f"cmp{k}", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    cmax = n("Mul", [f"cmp{k}", "fg"], f"cmax{k}")
# second corner = corner & (idx32 < cmax)
n("Sub", [cmax, "half"], "cmax_lo")
n("Less", ["idx32", "cmax_lo"], "lt_b"); n("Cast", ["lt_b"], "lt", to=TensorProto.FLOAT)
n("Mul", ["corner", "lt"], "second")                      # f32 0/1

# ---------- wave 2: type propagation (uint8) ----------
# typeval = 2*branch + 1*second, as uint8
n("Mul", ["branch", "two"], "br2")
n("Add", ["br2", "second"], "typef")                      # f32 in {0,1,2}
n("Cast", ["typef"], "type", to=TensorProto.UINT8)        # u8 seed
n("Cast", ["fg"], "fg8", to=TensorProto.UINT8)
for k in range(ITER):
    n("MaxPool", ["type"], f"tmp{k}", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    type = n("Mul", [f"tmp{k}", "fg8"], f"type{k}")       # noqa: A001
n("Cast", [type], "typeout", to=TensorProto.FLOAT)        # back to f32 {0,1,2}

# ---------- decode masks ----------
# m2 = fg*(type==2); m6 = fg*(type==1); m1 = fg*(type==0)
n("Equal", ["typeout", "two"], "is2_b"); n("Cast", ["is2_b"], "is2", to=TensorProto.FLOAT)
n("Mul", ["fg", "is2"], "m2")
n("Equal", ["typeout", "one"], "is1_b"); n("Cast", ["is1_b"], "is1", to=TensorProto.FLOAT)
n("Mul", ["fg", "is1"], "m6")
n("Add", ["m2", "m6"], "m26"); n("Sub", ["fg", "m26"], "m1")   # remaining fg -> color 1

# ---------- assemble output ----------
n("Sub", ["one", "fg"], "notfg"); n("Mul", ["active", "notfg"], "bg")
n("Concat", ["bg", "m1", "m2", "m6"], "mstack", axis=1)
n("Conv", ["mstack", "w_out"], "output")

# ---------- graph ----------
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, H, W])
graph = helper.make_graph(nodes, "task364", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
