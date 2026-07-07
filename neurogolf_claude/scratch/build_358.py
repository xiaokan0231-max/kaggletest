"""Compact ONNX for task358: cross-arm periodic fill of center row & column.

Rule: a plus/cross shape sits in the grid. Its center-column contiguous block
tiles periodically (period = block length) down the whole center column; the
center-row block tiles across the whole center row. Out-of-grid cells stay
unset (all-zero one-hot). Grid extent recovered from input one-hot: in-grid
cells have some channel set, out-of-grid cells are all-zero.

Memory discipline: the only f32 [1,1,30,30] tensor is the collapsed color grid
(input cost). All other interior grid tensors are uint8 (900 B). Index math
runs on tiny [30] int64 vectors. The final one-hot is the graph OUTPUT (excluded
from the memory metric).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

N = 30
nodes, inits = [], []
def K(name, arr):  # constant initializer
    inits.append(nh.from_array(np.asarray(arr), name))

# constants
K("wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))    # collapse one-hot -> color
K("colors10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))  # final Equal
K("ar30", np.arange(N, dtype=np.int64))                            # [30] index vector
K("ax1", np.array([1], np.int64))
K("ax0", np.array([0], np.int64))
K("ax13", np.array([1, 3], np.int64))   # reduce channel+width -> per-row indicator
K("ax12", np.array([1, 2], np.int64))   # reduce channel+height -> per-col indicator
K("ax2", np.array([2], np.int64))       # reduce rows
K("ax3", np.array([3], np.int64))       # reduce cols
K("zero_u8", np.array(0, np.uint8))
K("sentinel", np.array(255, np.uint8))
K("shp30", np.array([N, N], np.int64))
K("shp30v1", np.array([N], np.int64))
K("shp4_h", np.array([1, 1, N, 1], np.int64))   # column-fill broadcast shape
K("shp4_w", np.array([1, 1, 1, N], np.int64))   # row-fill broadcast shape
K("zero4_u8", np.array(0, np.uint8))   # scalar, broadcasts (saves 900 params vs full grid)
K("ar30_v4", np.arange(N, dtype=np.int64).reshape(1, 1, N, 1))  # row indices
K("ar30_h4", np.arange(N, dtype=np.int64).reshape(1, 1, 1, N))  # col indices

def nd(op, i, o, **kw):
    nodes.append(helper.make_node(op, i, o, **kw))

# --- color grid (f32 input cost) ---
nd("Conv", ["input", "wcol"], ["g_f"])                 # [1,1,30,30] f32  (input cost)
nd("Cast", ["g_f"], ["g_u8_4d"], to=TensorProto.UINT8)  # [1,1,30,30] uint8 color grid

# --- grid extent without a full-grid f32 mask tensor ---
# per-row indicator [1,30] and per-col indicator [1,30] (tiny f32), then counts H,W.
nd("ReduceMax", ["input", "ax13"], ["rowind"], keepdims=0)  # [1,30] f32 (1 if row in-grid)
nd("ReduceMax", ["input", "ax12"], ["colind"], keepdims=0)  # [1,30] f32
nd("ReduceSum", ["rowind", "ax1"], ["H_1"], keepdims=1)     # [1] f32 grid height
nd("ReduceSum", ["colind", "ax1"], ["W_1"], keepdims=1)     # [1] f32 grid width
nd("Cast", ["H_1"], ["H_i"], to=TensorProto.INT64)          # [1]
nd("Cast", ["W_1"], ["W_i"], to=TensorProto.INT64)          # [1]
# grid-bound conditions (kept as broadcastable 1-D masks; no full [1,1,30,30] AND)
nd("Less", ["ar30_v4", "H_i"], ["row_in"])   # [1,1,30,1] bool : row < H
nd("Less", ["ar30_h4", "W_i"], ["col_in"])   # [1,1,1,30] bool : col < W

# nonzero cross mask (4D) -> fp16 for reductions (ReduceSum/ArgMax need float/int)
nd("Greater", ["g_u8_4d", "zero_u8"], ["nz_b"])             # [1,1,30,30] bool
nd("Cast", ["nz_b"], ["nz_f16"], to=TensorProto.FLOAT16)     # [1,1,30,30] fp16 {0,1}

# per-column count (sum over rows=axis2) and per-row count (sum over cols=axis3)
nd("ReduceSum", ["nz_f16", "ax2"], ["colsum4"], keepdims=0)  # [1,1,30] fp16 (per col)
nd("ReduceSum", ["nz_f16", "ax3"], ["rowsum4"], keepdims=0)  # [1,1,30] fp16 (per row)
nd("Reshape", ["colsum4", "shp30v1"], ["colsum"])           # [30]
nd("Reshape", ["rowsum4", "shp30v1"], ["rowsum"])           # [30]
nd("ArgMax", ["colsum"], ["cc_1"], axis=0, keepdims=1)      # [1] int64
nd("ArgMax", ["rowsum"], ["cr_1"], axis=0, keepdims=1)      # [1] int64

# center column / row vectors (uint8 [30]) gathered from the 4D color grid
nd("Gather", ["g_u8_4d", "cc_1"], ["colvec_g"], axis=3)     # [1,1,30,1] uint8
nd("Reshape", ["colvec_g", "shp30v1"], ["colvec"])          # [30] uint8
nd("Gather", ["g_u8_4d", "cr_1"], ["rowvec_g"], axis=2)     # [1,1,1,30] uint8
nd("Reshape", ["rowvec_g", "shp30v1"], ["rowvec"])          # [30] uint8

# top/pv for column, left/ph for row  (vectors are tiny [30])
nd("Greater", ["colvec", "zero_u8"], ["colnz_b"])          # [30] bool
nd("Cast", ["colnz_b"], ["colnz_f16"], to=TensorProto.FLOAT16)  # [30] fp16
nd("ArgMax", ["colnz_f16"], ["top_1"], axis=0, keepdims=1)  # [1] int64
nd("ReduceSum", ["colnz_f16", "ax0"], ["pv_f16"], keepdims=1)  # [1] fp16
nd("Cast", ["pv_f16"], ["pv_1"], to=TensorProto.INT64)       # [1] int64

nd("Greater", ["rowvec", "zero_u8"], ["rownz_b"])
nd("Cast", ["rownz_b"], ["rownz_f16"], to=TensorProto.FLOAT16)
nd("ArgMax", ["rownz_f16"], ["left_1"], axis=0, keepdims=1)  # [1] int64
nd("ReduceSum", ["rownz_f16", "ax0"], ["ph_f16"], keepdims=1)
nd("Cast", ["ph_f16"], ["ph_1"], to=TensorProto.INT64)       # [1] int64

# colidx = top + ((ar30-top) mod pv) ; rowidx similarly  -> [30] int64
nd("Sub", ["ar30", "top_1"], ["r_top"])
nd("Mod", ["r_top", "pv_1"], ["r_mod"], fmod=0)
nd("Add", ["r_mod", "top_1"], ["colidx"])
nd("Gather", ["colvec", "colidx"], ["colfill"], axis=0)     # [30] uint8

nd("Sub", ["ar30", "left_1"], ["c_left"])
nd("Mod", ["c_left", "ph_1"], ["c_mod"], fmod=0)
nd("Add", ["c_mod", "left_1"], ["rowidx"])
nd("Gather", ["rowvec", "rowidx"], ["rowfill"], axis=0)     # [30] uint8

# Build the answer directly in 4D [1,1,30,30].  row axis -> dim2, col axis -> dim3.
nd("Reshape", ["colfill", "shp4_h"], ["colfill_4d"])   # [1,1,30,1] uint8
nd("Reshape", ["rowfill", "shp4_w"], ["rowfill_4d"])   # [1,1,1,30] uint8
# clip the two fills to grid bounds (tiny tensors): out-of-grid -> sentinel 255
nd("Where", ["row_in", "colfill_4d", "sentinel"], ["colfill_m"])  # [1,1,30,1] uint8
nd("Where", ["col_in", "rowfill_4d", "sentinel"], ["rowfill_m"])  # [1,1,1,30] uint8
# background: in-grid off-cross -> 0, off-grid -> 255  (built via nested Where; inner is tiny)
nd("Where", ["col_in", "zero4_u8", "sentinel"], ["bg_inner"])     # [1,1,1,30] uint8
nd("Where", ["row_in", "bg_inner", "sentinel"], ["bg"])           # [1,1,30,30] uint8
# place center column then center row (intersection consistent)
nd("Equal", ["ar30_h4", "cc_1"], ["is_cc4"])   # [1,1,1,30] bool : col == cc
nd("Equal", ["ar30_v4", "cr_1"], ["is_cr4"])   # [1,1,30,1] bool : row == cr
nd("Where", ["is_cc4", "colfill_m", "bg"], ["o_col"])           # [1,1,30,30] uint8
nd("Where", ["is_cr4", "rowfill_m", "o_col"], ["out_color_4d"]) # [1,1,30,30] uint8
nd("Equal", ["out_color_4d", "colors10"], ["output"])           # [1,10,30,30] bool (output)

graph = helper.make_graph(
    nodes, "task358",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, N, N])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, N, N])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b20_task358.onnx")
print("saved /tmp/b20_task358.onnx")
