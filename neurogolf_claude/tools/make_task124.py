"""task124: vertical-period extension. Input Hx10 (H in 5..8) is the top of a
pattern with vertical period p and horizontal shift s: row[r+p] = shift_right_s(row[r]).
Output is the 10x10 downward extension.

ONNX plan (candidate enumeration over (p,s) in {1,2,3}x{0,1,2}, smallest p first):
  X9   = Slice(input, ch 1..9, rows 0..9, cols 0..9)        [1,9,10,10]
  Xb   = Reshape -> [9,1,10,10]   (colored channels as batch)
  rows = ReduceSum(input, axes ch+col) sliced to 10 rows     -> 10 if r<H else 0
  shifts = Conv(Xb, Wshift[9,1,4,3]) -> [9,9,10,10], channel k = X shifted down p_k right s_k
  per-row mismatch via binary identity (a-b)^2 = a + b - 2ab, reduced over (color, col)
  BEFORE masking, so masks live in tiny [9,10] space:
    sqrow[k,r] = XbRow[r] + shiftRow[k,r] - 2*crossRow[k,r],  cross = Xb*shifts
    m_k = sum_r sqrow[k,r] * (r >= p_k) * (r < H)
  (colored channels only: ch0 would false-mismatch at left zero-fill)
  valid = 1 - Sign(m);  w = valid * (1 - Sign(exclusive CumSum(valid)))   first valid wins
  Xsel = Slice(Xb, rows 0..2) * (topmask_k * w)   topmask_k static: r < p_k (p<=3)
  gen  = depthwise Conv(Xsel, Wgen[9,1,10,19], group=9): channel k sums shifts (j*p_k, j*s_k)
  out  = ReduceSum(gen, k) -> [1,9,10,10]; bg = 1 - sum(colored); Concat; Pad to 30x30.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT
CANDS = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]

inits = []


def init_f(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))


def init_i(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))


# --- constants ---
init_i("sl_starts", [1, 0, 0])
init_i("sl_ends", [10, 10, 10])
init_i("sl_axes", [1, 2, 3])
init_i("shape_b", [9, 1, 10, 10])
init_i("rs_axes", [1, 3])
init_i("i0", [0])
init_i("i3", [3])
init_i("i10", [10])
init_i("ax2", [2])
init_i("shape_r2", [1, 10])
init_i("red03", [0, 3])
init_i("ax1", [1])

Wsh = np.zeros((9, 1, 4, 3), np.float32)  # out[y,x] = in[y-p, x-s] with pads top=3,left=2
for k, (p, s) in enumerate(CANDS):
    Wsh[k, 0, 3 - p, 2 - s] = 1.0
init_f("Wshift", Wsh)

rmask2d = np.zeros((9, 10), np.float32)   # rows r >= p_k participate in the check
tmask3 = np.zeros((1, 9, 3, 1), np.float32)  # rows r < p_k generate the output (p<=3)
for k, (p, s) in enumerate(CANDS):
    rmask2d[k, p:] = 1.0
    tmask3[0, k, :p, 0] = 1.0
init_f("rmask2d", rmask2d)
init_f("tmask3", tmask3)

init_f("two", np.float32(2.0))
init_f("one9", np.ones(9))
init_i("cs_axis", 0)
init_i("shape_w", [1, 9, 1, 1])

Wg = np.zeros((9, 1, 10, 19), np.float32)  # out[y,x] = sum_j in[y-j*p, x-j*s], pads top=9,left=18
for k, (p, s) in enumerate(CANDS):
    j = 0
    while j * p <= 9:
        Wg[k, 0, 9 - j * p, 18 - j * s] = 1.0
        j += 1
init_f("Wgen", Wg)

init_i("shape_c", [1, 9, 10, 10])
init_f("one_s", np.ones((1, 1, 1, 1)))
init_i("padspec", [0, 0, 0, 0, 0, 0, 20, 20])

n = helper.make_node
nodes = [
    # crop to colored channels, 10x10
    n("Slice", ["input", "sl_starts", "sl_ends", "sl_axes"], ["X9"]),       # [1,9,10,10]
    n("Reshape", ["X9", "shape_b"], ["Xb"]),                                # [9,1,10,10]
    # dynamic row indicator (10 if r<H else 0); from all 10 channels (blank rows possible)
    n("ReduceSum", ["input", "rs_axes"], ["rowsum"], keepdims=1),           # [1,1,30,1]
    n("Slice", ["rowsum", "i0", "i10", "ax2"], ["rows10"]),                 # [1,1,10,1]
    n("Reshape", ["rows10", "shape_r2"], ["rows2d"]),                       # [1,10]
    # candidate shifts; per-row squared mismatch without fat intermediates:
    # sum_{c,x} (Xb - shifts)^2 = XbRow + shiftRow - 2*crossRow  (entries are 0/1)
    n("Conv", ["Xb", "Wshift"], ["shifts"], kernel_shape=[4, 3], pads=[3, 2, 0, 0]),  # [9,9,10,10]
    n("Mul", ["Xb", "shifts"], ["cross"]),                                  # [9,9,10,10]
    n("ReduceSum", ["cross", "red03"], ["crossRow"], keepdims=0),           # [9,10]
    n("ReduceSum", ["Xb", "red03"], ["XbRow"], keepdims=0),                 # [1,10]
    n("ReduceSum", ["shifts", "red03"], ["shiftRow"], keepdims=0),          # [9,10]
    n("Add", ["XbRow", "shiftRow"], ["t1"]),                                # [9,10]
    n("Mul", ["crossRow", "two"], ["cross2"]),                              # [9,10]
    n("Sub", ["t1", "cross2"], ["sqrow"]),                                  # [9,10]
    # mask rows: r >= p_k (static) AND r < H (dynamic), then total mismatch per candidate
    n("Mul", ["rmask2d", "rows2d"], ["mask2d"]),                            # [9,10]
    n("Mul", ["sqrow", "mask2d"], ["masked"]),                              # [9,10]
    n("ReduceSum", ["masked", "ax1"], ["m"], keepdims=0),                   # [9]
    # first-valid (smallest p, then s in 0,1,2) one-hot selector
    n("Sign", ["m"], ["sm"]),
    n("Sub", ["one9", "sm"], ["valid"]),
    n("CumSum", ["valid", "cs_axis"], ["prefix"], exclusive=1),
    n("Sign", ["prefix"], ["sp"]),
    n("Sub", ["one9", "sp"], ["gate"]),
    n("Mul", ["valid", "gate"], ["w"]),
    n("Reshape", ["w", "shape_w"], ["w4"]),                                 # [1,9,1,1]
    # generation from winner's top-p rows (p <= 3, so rows 0..2 suffice)
    n("Mul", ["tmask3", "w4"], ["msel"]),                                   # [1,9,3,1]
    n("Slice", ["Xb", "i0", "i3", "ax2"], ["Xtop3"]),                       # [9,1,3,10]
    n("Mul", ["Xtop3", "msel"], ["Xsel"]),                                  # [9,9,3,10]
    n("Conv", ["Xsel", "Wgen"], ["gen"],
      kernel_shape=[10, 19], pads=[9, 18, 7, 0], group=9),                  # [9,9,10,10]
    n("ReduceSum", ["gen", "ax1"], ["selsum"], keepdims=1),                 # [9,1,10,10]
    n("Reshape", ["selsum", "shape_c"], ["C"]),                             # [1,9,10,10]
    # background channel + assemble + pad to 30x30
    n("ReduceSum", ["C", "ax1"], ["sumc"], keepdims=1),                     # [1,1,10,10]
    n("Sub", ["one_s", "sumc"], ["bg"]),                                    # [1,1,10,10]
    n("Concat", ["bg", "C"], ["out10"], axis=1),                            # [1,10,10,10]
    n("Pad", ["out10", "padspec"], ["output"]),                             # [1,10,30,30]
]

graph = helper.make_graph(
    nodes, "task124",
    [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
out_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task124.onnx"
onnx.save(model, out_path)
params = sum(int(np.prod(t.dims)) if t.dims else 1 for t in inits)
print("saved", out_path, "params:", params)
