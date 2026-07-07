"""task093 v2: fp16 interior to halve memory on the [1,1,14,14] numeric tensors."""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

G = 14
F16 = TensorProto.FLOAT16
nodes = []
inits = []
def add_const(name, arr):
    inits.append(numpy_helper.from_array(arr, name))

# collapse one-hot -> color grid (must be f32 Conv)
w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
add_const("collapse_w", w)
nodes.append(helper.make_node("Conv", ["input", "collapse_w"], ["g30"]))  # f32 [1,1,30,30] 3600B
add_const("crop_starts", np.array([0, 0], np.int64))
add_const("crop_ends", np.array([G, G], np.int64))
add_const("crop_axes", np.array([2, 3], np.int64))
nodes.append(helper.make_node("Slice", ["g30", "crop_starts", "crop_ends", "crop_axes"], ["gf32"]))
nodes.append(helper.make_node("Cast", ["gf32"], ["g"], to=F16))  # fp16 color grid

add_const("c5", np.array(5, np.float16))
add_const("c0", np.array(0, np.float16))
nodes.append(helper.make_node("Equal", ["g", "c5"], ["b_bool"]))
nodes.append(helper.make_node("Equal", ["g", "c0"], ["isz_bool"]))
nodes.append(helper.make_node("Not", ["b_bool"], ["nb_bool"]))
nodes.append(helper.make_node("Not", ["isz_bool"], ["nz_bool"]))
nodes.append(helper.make_node("And", ["nb_bool", "nz_bool"], ["m_bool"]))
nodes.append(helper.make_node("Cast", ["b_bool"], ["bf"], to=F16))
nodes.append(helper.make_node("Cast", ["m_bool"], ["mf"], to=F16))

add_const("axW", np.array([3], np.int64))
add_const("axH", np.array([2], np.int64))
nodes.append(helper.make_node("ReduceSum", ["bf", "axW"], ["rowsum"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["bf", "axH"], ["colsum"], keepdims=1))
add_const("cW", np.array(float(G), np.float16))
nodes.append(helper.make_node("Equal", ["rowsum", "cW"], ["rowfull_b"]))
nodes.append(helper.make_node("Equal", ["colsum", "cW"], ["colfull_b"]))
nodes.append(helper.make_node("Cast", ["rowfull_b"], ["rowfull"], to=F16))
nodes.append(helper.make_node("Cast", ["colfull_b"], ["colfull"], to=F16))

add_const("axis2", np.array(2, np.int64))
add_const("axis3", np.array(3, np.int64))
nodes.append(helper.make_node("CumSum", ["rowfull", "axis2"], ["bbelow_s"], reverse=1, exclusive=1))
nodes.append(helper.make_node("CumSum", ["rowfull", "axis2"], ["babove_s"], reverse=0, exclusive=1))
nodes.append(helper.make_node("CumSum", ["colfull", "axis3"], ["bright_s"], reverse=1, exclusive=1))
nodes.append(helper.make_node("CumSum", ["colfull", "axis3"], ["bleft_s"], reverse=0, exclusive=1))

add_const("zf", np.array(0.0, np.float16))
nodes.append(helper.make_node("Greater", ["bbelow_s", "zf"], ["bbelow_b"]))
nodes.append(helper.make_node("Greater", ["babove_s", "zf"], ["babove_b"]))
nodes.append(helper.make_node("Greater", ["bright_s", "zf"], ["bright_b"]))
nodes.append(helper.make_node("Greater", ["bleft_s", "zf"], ["bleft_b"]))
nodes.append(helper.make_node("Not", ["rowfull_b"], ["nrf_b"]))
nodes.append(helper.make_node("Not", ["colfull_b"], ["ncf_b"]))
nodes.append(helper.make_node("And", ["bbelow_b", "nrf_b"], ["above_row_b"]))
nodes.append(helper.make_node("And", ["babove_b", "nrf_b"], ["below_row_b"]))
nodes.append(helper.make_node("And", ["bright_b", "ncf_b"], ["left_col_b"]))
nodes.append(helper.make_node("And", ["bleft_b", "ncf_b"], ["right_col_b"]))
for nm in ["above_row", "below_row", "left_col", "right_col"]:
    nodes.append(helper.make_node("Cast", [nm + "_b"], [nm], to=F16))

add_const("ones_grid", np.ones((1, 1, G, G), np.float16))
nodes.append(helper.make_node("Mul", ["above_row", "ones_grid"], ["A"]))
nodes.append(helper.make_node("Mul", ["below_row", "ones_grid"], ["Bl"]))
nodes.append(helper.make_node("Mul", ["left_col", "ones_grid"], ["L"]))
nodes.append(helper.make_node("Mul", ["right_col", "ones_grid"], ["R"]))

nodes.append(helper.make_node("CumSum", ["A", "axis2"], ["Da"], reverse=1, exclusive=0))
nodes.append(helper.make_node("CumSum", ["Bl", "axis2"], ["Db"], reverse=0, exclusive=0))
nodes.append(helper.make_node("CumSum", ["L", "axis3"], ["Dl"], reverse=1, exclusive=0))
nodes.append(helper.make_node("CumSum", ["R", "axis3"], ["Dr"], reverse=0, exclusive=0))

nodes.append(helper.make_node("Mul", ["mf", "A"], ["mA"]))
nodes.append(helper.make_node("Mul", ["mf", "Bl"], ["mB"]))
nodes.append(helper.make_node("Mul", ["mf", "L"], ["mL"]))
nodes.append(helper.make_node("Mul", ["mf", "R"], ["mR"]))
nodes.append(helper.make_node("ReduceSum", ["mA", "axH"], ["Na"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["mB", "axH"], ["Nb"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["mL", "axW"], ["Nl"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["mR", "axW"], ["Nr"], keepdims=1))

add_const("half", np.array(0.5, np.float16))
def fill(D, N, out):
    nodes.append(helper.make_node("Greater", [D, "half"], [out + "_ge1"]))
    nodes.append(helper.make_node("LessOrEqual", [D, N], [out + "_leN"]))
    nodes.append(helper.make_node("And", [out + "_ge1", out + "_leN"], [out]))
fill("Da", "Na", "fa")
fill("Db", "Nb", "fb")
fill("Dl", "Nl", "fl")
fill("Dr", "Nr", "fr")

nodes.append(helper.make_node("Or", ["b_bool", "fa"], ["o1"]))
nodes.append(helper.make_node("Or", ["o1", "fb"], ["o2"]))
nodes.append(helper.make_node("Or", ["o2", "fl"], ["o3"]))
nodes.append(helper.make_node("Or", ["o3", "fr"], ["out5_b"]))

nodes.append(helper.make_node("Cast", ["out5_b"], ["out5_u8"], to=TensorProto.UINT8))
add_const("five_u8", np.array(5, np.uint8))
nodes.append(helper.make_node("Mul", ["out5_u8", "five_u8"], ["gm14"]))
add_const("pads", np.array([0, 0, 0, 0, 0, 0, 30 - G, 30 - G], np.int64))
add_const("padval_u8", np.array(255, np.uint8))
nodes.append(helper.make_node("Pad", ["gm14", "pads", "padval_u8"], ["gm30"]))
add_const("colors_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Equal", ["gm30", "colors_u8"], ["output"]))

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "task093", [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b12_task093.onnx")
print("saved nodes=", len(nodes), "inits=", len(inits))
