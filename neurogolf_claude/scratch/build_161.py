import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H = W = 30
NC = 10

def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)

nodes = []
inits = []

# ---- 1. color grid g (fp32) via Conv weight [0..9] ----
w_color = np.arange(NC, dtype=np.float32).reshape(1, NC, 1, 1)
inits.append(const("w_color", w_color))
nodes.append(helper.make_node("Conv", ["input", "w_color"], ["g_f"]))  # [1,1,30,30] fp32

# ---- 2. grid count -> gridmask via Conv with all-ones weight ----
w_ones = np.ones((1, NC, 1, 1), dtype=np.float32)
inits.append(const("w_ones", w_ones))
nodes.append(helper.make_node("Conv", ["input", "w_ones"], ["cnt_f"]))  # [1,1,30,30] fp32
inits.append(const("half", np.array(0.5, dtype=np.float32)))
nodes.append(helper.make_node("Greater", ["cnt_f", "half"], ["gmask"]))  # bool gridmask

# ---- cast g to uint8 ----
nodes.append(helper.make_node("Cast", ["g_f"], ["gu"], to=TensorProto.UINT8))  # uint8 color grid

# ---- 3. interior via erosion of gmask (MaxPool on (1-gmask)) ----
# erosion(gmask) = NOT dilation(NOT gmask); dilation via MaxPool 3x3 pad1
gmask_f = "gmask_f"
nodes.append(helper.make_node("Cast", ["gmask"], [gmask_f], to=TensorProto.FLOAT))
inits.append(const("one_f", np.array(1.0, dtype=np.float32)))
nodes.append(helper.make_node("Sub", ["one_f", gmask_f], ["not_gmask_f"]))
# Pad not_gmask_f with value 1.0 (out-of-array treated as NON-grid) on all sides, then valid MaxPool
inits.append(const("pads1", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)))
nodes.append(helper.make_node("Pad", ["not_gmask_f", "pads1", "one_f"], ["ngm_pad"], mode="constant"))
nodes.append(helper.make_node(
    "MaxPool", ["ngm_pad"], ["dil_notgmask"],
    kernel_shape=[3, 3], strides=[1, 1]))
# interior = 1 - dil_notgmask  (==1 where all 3x3 nbrs are grid)
nodes.append(helper.make_node("Greater", ["half", "dil_notgmask"], ["interior"]))  # 0.5 > dil -> interior bool
nodes.append(helper.make_node("Cast", ["interior"], ["interior_f"], to=TensorProto.FLOAT))

# ---- 4. special color detection (cheap: per-color [1,10] reductions) ----
# special color = unique color present on grid with ZERO interior count.
# grid_count[c] = ReduceSum(input, spatial)  (input excluded from memory)
inits.append(const("axes23", np.array([2, 3], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["input", "axes23"], ["grid_count"], keepdims=0))  # [1,10] fp32
# interior one-hot then reduce: input * interior  -> [1,10,30,30] (single masked tensor)
nodes.append(helper.make_node("Mul", ["input", "interior_f"], ["int_oh"]))  # [1,10,30,30]
nodes.append(helper.make_node("ReduceSum", ["int_oh", "axes23"], ["int_count"], keepdims=0))  # [1,10] fp32
# is_special = (grid_count>0.5) AND (int_count<0.5)
nodes.append(helper.make_node("Greater", ["grid_count", "half"], ["present10"]))   # bool [1,10]
nodes.append(helper.make_node("Greater", ["half", "int_count"], ["noint10"]))      # bool [1,10] (int_count<0.5)
nodes.append(helper.make_node("And", ["present10", "noint10"], ["is_special10"]))   # bool [1,10]
nodes.append(helper.make_node("Cast", ["is_special10"], ["is_special10_f"], to=TensorProto.FLOAT))
colorvals = np.arange(NC, dtype=np.float32).reshape(1, NC)  # [1,10] values 0..9
inits.append(const("colorvals10", colorvals))
nodes.append(helper.make_node("Mul", ["is_special10_f", "colorvals10"], ["scvec_f"]))  # [1,10] fp32
inits.append(const("axis1", np.array([1], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["scvec_f", "axis1"], ["sc_f"], keepdims=1))  # [1,1] fp32
nodes.append(helper.make_node("Cast", ["sc_f"], ["sc"], to=TensorProto.UINT8))  # [1,1] uint8
inits.append(const("u0", np.array(0, dtype=np.uint8)))

# ---- 5. special-cell mask sp = (gu==sc)&gmask ----
# need sc broadcastable to [1,1,30,30]; reshape sc [1,1] -> [1,1,1,1]
inits.append(const("shp1111", np.array([1, 1, 1, 1], dtype=np.int64)))
nodes.append(helper.make_node("Reshape", ["sc", "shp1111"], ["sc4"]))  # uint8 [1,1,1,1]
nodes.append(helper.make_node("Equal", ["gu", "sc4"], ["eq_sc"]))  # bool [1,1,30,30]
nodes.append(helper.make_node("And", ["eq_sc", "gmask"], ["sp"]))  # special cells

# ---- 6. edge classification ----
# A grid cell is on LR edge if left OR right neighbor is not grid.
# Compute neighbor-not-grid via MaxPool of (1-gmask) along horizontal/vertical only.
# horizontal dilation kernel [1,3]: dilates not_gmask horizontally -> cell has non-grid left/right nbr (or self)
# reuse padded ngm_pad (padded by 1 on all sides) for directional pools too
nodes.append(helper.make_node(
    "MaxPool", ["ngm_pad"], ["h_dil_pad"],
    kernel_shape=[1, 3], strides=[1, 1]))  # output [1,1,32,30]: valid over W, full over padded H
# crop the extra padded H rows (1 top, 1 bottom)
inits.append(const("starts_h", np.array([1], dtype=np.int64)))
inits.append(const("ends_h", np.array([1 + H], dtype=np.int64)))
inits.append(const("axis2_s", np.array([2], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["h_dil_pad", "starts_h", "ends_h", "axis2_s"], ["h_dil"]))
nodes.append(helper.make_node(
    "MaxPool", ["ngm_pad"], ["v_dil_pad"],
    kernel_shape=[3, 1], strides=[1, 1]))  # output [1,1,30,32]
inits.append(const("starts_w", np.array([1], dtype=np.int64)))
inits.append(const("ends_w", np.array([1 + W], dtype=np.int64)))
inits.append(const("axis3_s", np.array([3], dtype=np.int64)))
nodes.append(helper.make_node("Slice", ["v_dil_pad", "starts_w", "ends_w", "axis3_s"], ["v_dil"]))
# LR_edge = gmask AND h_dil>0.5  (cell is grid but has a horizontal non-grid neighbor or is itself... self is grid so not)
nodes.append(helper.make_node("Greater", ["h_dil", "half"], ["h_edge_b"]))  # bool
nodes.append(helper.make_node("Greater", ["v_dil", "half"], ["v_edge_b"]))
nodes.append(helper.make_node("And", ["sp", "h_edge_b"], ["sp_LR"]))  # special on left/right edge -> fill row
nodes.append(helper.make_node("And", ["sp", "v_edge_b"], ["sp_TB"]))  # special on top/bottom edge -> fill col

# ---- 7. rows/cols to fill ----
# fill_rows[r] = any over c of sp_LR  -> ReduceMax over axis W (=3), keepdims to broadcast
nodes.append(helper.make_node("Cast", ["sp_LR"], ["sp_LR_u"], to=TensorProto.UINT8))
nodes.append(helper.make_node("Cast", ["sp_TB"], ["sp_TB_u"], to=TensorProto.UINT8))
inits.append(const("axis3", np.array([3], dtype=np.int64)))
inits.append(const("axis2", np.array([2], dtype=np.int64)))
nodes.append(helper.make_node("ReduceMax", ["sp_LR_u", "axis3"], ["fill_rows"], keepdims=1))  # [1,1,30,1]
nodes.append(helper.make_node("ReduceMax", ["sp_TB_u", "axis2"], ["fill_cols"], keepdims=1))  # [1,1,1,30]
# cross = (fill_rows OR fill_cols) broadcast & gmask
nodes.append(helper.make_node("Cast", ["fill_rows"], ["fr_b"], to=TensorProto.BOOL))
nodes.append(helper.make_node("Cast", ["fill_cols"], ["fc_b"], to=TensorProto.BOOL))
nodes.append(helper.make_node("Or", ["fr_b", "fc_b"], ["cross_raw"]))  # [1,1,30,30] bool
nodes.append(helper.make_node("And", ["cross_raw", "gmask"], ["cross"]))  # bool

# ---- 8. build color grid gm: padding=255, cross=sc, else 0 ----
# gm = where(gmask, where(cross, sc, 0), 255)
inits.append(const("u255", np.array(255, dtype=np.uint8)))
# where(cross, sc4, 0)
nodes.append(helper.make_node("Where", ["cross", "sc4", "u0"], ["incolor"]))  # uint8 [1,1,30,30]
nodes.append(helper.make_node("Where", ["gmask", "incolor", "u255"], ["gm_color"]))  # uint8

# ---- 9. output one-hot = Equal(gm_color, colors[1,10,1,1]) ----
colorsout = np.arange(NC, dtype=np.uint8).reshape(1, NC, 1, 1)
inits.append(const("colorsout", colorsout))
nodes.append(helper.make_node("Equal", ["gm_color", "colorsout"], ["output"]))  # bool [1,10,30,30] OUTPUT

graph = helper.make_graph(
    nodes, "task161",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, NC, H, W])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, NC, H, W])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "/tmp/b10_task161.onnx")
print("saved")
