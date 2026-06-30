"""Build compact ONNX for task143 -- single-grid hybrid (tiny [1,1,10,10] tensors).
Rule: template footprint Tb in box rows0-2/cols0-2; any single-color object whose
footprint exactly matches Tb -> recolor to 5.

Algorithm (all static ops, work on uint8 [1,1,10,10]):
  filled = work>0
  countgrid[y,x] = #cells sharing color of (y,x)   (via 8-ch masks * per-color count)
  For each anchor (i,j), over the 9 box offsets (a,b in 0..2) weighted by Tb[a,b]:
    corr   = sum  Tb*filled(i+a,j+b)         -> template fully present where corr==K
    cmax   = max  Tb*work(i+a,j+b)           -> color at Tb cells
    cmin   = min  (Tb*work + (1-Tb)*BIG)     -> min color (BIG ignores Tb-off)
    cntmin = min  (Tb*cg   + (1-Tb)*BIG)     -> object's total count
  anchor_ok = (corr==K) & (cmin==cmax) & (cntmin==K) & (cmax>0)
  recolor   = dilate anchor_ok by Tb (reflected)  -> footprint cells to set to 5
  gm = where(recolor, 5, g) ; pad 30 with 255 ; Equal vs colors -> output
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

C, N = [], []
BIG = 100


def const(name, arr):
    C.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def node(op, ins, outs, **kw):
    N.append(helper.make_node(op, ins, outs, **kw))
    return outs[0] if isinstance(outs, list) else outs


def shift(src_padded, a, b, H, W, tag):
    """Slice a HxW window from a padded grid at offset (a,b)."""
    s = const(f"s_{tag}", np.array([a, b], dtype=np.int64))
    e = const(f"e_{tag}", np.array([a + H, b + W], dtype=np.int64))
    out = f"sh_{tag}"
    node("Slice", [src_padded, s, e, "ax23"], [out])
    return out


inp = "input"
H = W = 10
const("ax23", np.array([2, 3], dtype=np.int64))
const("s0", np.array([0, 0], dtype=np.int64))

# color grid g30 [1,1,30,30] f32 -> uint8 -> crop 10x10
const("w_color", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
node("Conv", [inp, "w_color"], ["g30"])
node("Cast", ["g30"], ["g30u"], to=TensorProto.UINT8)
const("e10", np.array([10, 10], dtype=np.int64))
node("Slice", ["g30u", "s0", "e10", "ax23"], ["g"])  # [1,1,10,10] uint8

# template box Tb
const("e3", np.array([3, 3], dtype=np.int64))
node("Slice", ["g", "s0", "e3", "ax23"], ["tbox"])  # [1,1,3,3] uint8
const("u0", np.array(0, dtype=np.uint8))
node("Greater", ["tbox", "u0"], ["Tb_b"])           # bool
node("Cast", ["Tb_b"], ["Tb"], to=TensorProto.INT32)  # int32 0/1 [1,1,3,3]
node("ReduceSum", ["Tb"], ["K"], keepdims=1)          # [1,1,1,1] int32

# work = g * boxmask
bm = np.ones((1, 1, 10, 10), np.uint8); bm[:, :, :4, :4] = 0
const("boxmask", bm)
node("Mul", ["g", "boxmask"], ["work_u"])             # uint8
node("Cast", ["work_u"], ["work"], to=TensorProto.INT32)  # int32 [1,1,10,10]
node("Greater", ["work", "i0"], ["filled_b"])
const("i0", np.array(0, dtype=np.int32))
node("Cast", ["filled_b"], ["filled"], to=TensorProto.INT32)  # int32 0/1

# countgrid via 8 color channels
colors = [1, 2, 3, 4, 6, 7, 8, 9]
const("cvec", np.array(colors, dtype=np.int32).reshape(1, 8, 1, 1))
node("Equal", ["work", "cvec"], ["cmask_b"])          # bool [1,8,10,10]
node("Cast", ["cmask_b"], ["cmask"], to=TensorProto.INT32)  # int32 [1,8,10,10]
const("ax23b", np.array([2, 3], dtype=np.int64))
node("ReduceSum", ["cmask", "ax23b"], ["ccount"], keepdims=1)  # [1,8,1,1] count per color
node("Mul", ["cmask", "ccount"], ["cweighted"])       # [1,8,10,10]
const("ax1", np.array([1], dtype=np.int64))
node("ReduceSum", ["cweighted", "ax1"], ["countgrid"], keepdims=1)  # [1,1,10,10] int32

# pad grids: pad bottom/right by 2 for forward shifts
const("pad_br2", np.array([0, 0, 0, 0, 0, 0, 2, 2], dtype=np.int64))
node("Pad", ["filled", "pad_br2"], ["filled_p"])      # [1,1,12,12]
node("Pad", ["work", "pad_br2"], ["work_p"])          # int32 [1,1,12,12]
node("Pad", ["countgrid", "pad_br2"], ["cg_p"])       # int32 [1,1,12,12]

# accumulate over 9 offsets
const("BIG", np.array(BIG, dtype=np.int32))
corr = cmax = cmin = cntmin = None
k = 0
for a in range(3):
    for b in range(3):
        # Tb[a,b] scalar
        sa = const(f"ts{k}", np.array([a, b], dtype=np.int64))
        ea = const(f"te{k}", np.array([a + 1, b + 1], dtype=np.int64))
        node("Slice", ["Tb", f"ts{k}", f"te{k}", "ax23"], [f"t{k}"])  # [1,1,1,1] int32
        node("Sub", ["one", f"t{k}"], [f"nt{k}"])                     # 1-Tb
        # shifted windows
        fpw = shift("filled_p", a, b, H, W, f"f{k}")
        wpw = shift("work_p", a, b, H, W, f"w{k}")
        cgw = shift("cg_p", a, b, H, W, f"c{k}")
        # corr += Tb*filled
        node("Mul", [f"t{k}", fpw], [f"cf{k}"])
        corr = (f"cf{k}") if corr is None else node("Add", [corr, f"cf{k}"], [f"corr{k}"])
        # tvw = Tb*work
        node("Mul", [f"t{k}", wpw], [f"tvw{k}"])
        # cmax = max(cmax, tvw)
        cmax = (f"tvw{k}") if cmax is None else node("Max", [cmax, f"tvw{k}"], [f"cmax{k}"])
        # cminterm = tvw + (1-Tb)*BIG
        node("Mul", [f"nt{k}", "BIG"], [f"ntb{k}"])
        node("Add", [f"tvw{k}", f"ntb{k}"], [f"cmt{k}"])
        cmin = (f"cmt{k}") if cmin is None else node("Min", [cmin, f"cmt{k}"], [f"cmin{k}"])
        # cntterm = Tb*cg + (1-Tb)*BIG
        node("Mul", [f"t{k}", cgw], [f"tcg{k}"])
        node("Add", [f"tcg{k}", f"ntb{k}"], [f"cnt{k}"])
        cntmin = (f"cnt{k}") if cntmin is None else node("Min", [cntmin, f"cnt{k}"], [f"cntm{k}"])
        k += 1
const("one", np.array(1, dtype=np.int32))

# anchor_ok = (corr==K)&(cmin==cmax)&(cntmin==K)&(cmax>0)
node("Equal", [corr, "K"], ["eqK"])         # bool
node("Equal", [cmin, cmax], ["uni"])        # bool
node("Equal", [cntmin, "K"], ["cntok"])     # bool
node("Greater", [cmax, "i0"], ["pos"])      # bool
node("And", ["eqK", "uni"], ["a1"])
node("And", ["cntok", "pos"], ["a2"])
node("And", ["a1", "a2"], ["anchor_b"])
node("Cast", ["anchor_b"], ["anchor"], to=TensorProto.INT32)  # [1,1,10,10]

# dilate anchor by Tb: recolor[y,x] = max over (a,b) Tb[a,b]*anchor[y-a,x-b]
# pad anchor top/left by 2
const("pad_tl2", np.array([0, 0, 2, 2, 0, 0, 0, 0], dtype=np.int64))
node("Pad", ["anchor", "pad_tl2"], ["anchor_p"])  # [1,1,12,12]
recolor = None
k = 0
for a in range(3):
    for b in range(3):
        # window at (2-a, 2-b)
        s = const(f"ds{k}", np.array([2 - a, 2 - b], dtype=np.int64))
        e = const(f"de{k}", np.array([2 - a + H, 2 - b + W], dtype=np.int64))
        node("Slice", ["anchor_p", f"ds{k}", f"de{k}", "ax23"], [f"aw{k}"])
        node("Mul", [f"t{k}", f"aw{k}"], [f"rt{k}"])  # reuse t{k} (Tb[a,b])
        recolor = (f"rt{k}") if recolor is None else node("Max", [recolor, f"rt{k}"], [f"rc{k}"])
        k += 1
node("Greater", [recolor, "i0"], ["recolor_b"])  # bool [1,1,10,10]

# gm = where(recolor, 5, g)
const("five", np.array(5, dtype=np.uint8))
node("Where", ["recolor_b", "five", "g"], ["gm10"])  # uint8 [1,1,10,10]
const("pad_to30", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64))
const("sentinel", np.array(255, dtype=np.uint8))
node("Pad", ["gm10", "pad_to30", "sentinel"], ["gm30"])
const("allcolors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
node("Equal", ["gm30", "allcolors"], ["output"])

gin = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
gout = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
graph = helper.make_graph(N, "task143", [gin], [gout], C)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b8_task143b.onnx")
print("saved b, nodes:", len(N))
