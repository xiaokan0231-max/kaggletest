"""task093 v5: trim bool-grid count (merge marker mask, Where for gm, fold combine)."""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

G = 14
F16 = TensorProto.FLOAT16
nodes = []; inits = []
def add_const(name, arr): inits.append(numpy_helper.from_array(arr, name))

w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
add_const("collapse_w", w)
nodes.append(helper.make_node("Conv", ["input", "collapse_w"], ["g30"]))
add_const("cs", np.array([0, 0], np.int64)); add_const("ce", np.array([G, G], np.int64)); add_const("ca", np.array([2, 3], np.int64))
nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["gf32"]))  # f32 [1,1,14,14]

add_const("c5", np.array(5, np.float32)); add_const("c0", np.array(0, np.float32))
nodes.append(helper.make_node("Equal", ["gf32", "c5"], ["b_bool"]))
nodes.append(helper.make_node("Equal", ["gf32", "c0"], ["isz_bool"]))
nodes.append(helper.make_node("Or", ["b_bool", "isz_bool"], ["bg_bool"]))   # is band-or-zero
nodes.append(helper.make_node("Not", ["bg_bool"], ["m_bool"]))               # marker
nodes.append(helper.make_node("Cast", ["b_bool"], ["bf"], to=F16))
nodes.append(helper.make_node("Cast", ["m_bool"], ["mf"], to=F16))

add_const("axW", np.array([3], np.int64)); add_const("axH", np.array([2], np.int64))
nodes.append(helper.make_node("ReduceSum", ["bf", "axW"], ["rowsum"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["bf", "axH"], ["colsum"], keepdims=1))
add_const("cW", np.array(float(G), np.float16))
nodes.append(helper.make_node("Equal", ["rowsum", "cW"], ["rowfull_b"]))
nodes.append(helper.make_node("Equal", ["colsum", "cW"], ["colfull_b"]))
nodes.append(helper.make_node("Cast", ["rowfull_b"], ["rowfull"], to=F16))
nodes.append(helper.make_node("Cast", ["colfull_b"], ["colfull"], to=F16))

add_const("axis2", np.array(2, np.int64)); add_const("axis3", np.array(3, np.int64))
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

nodes.append(helper.make_node("CumSum", ["above_row", "axis2"], ["Da"], reverse=1))
nodes.append(helper.make_node("CumSum", ["below_row", "axis2"], ["Db"], reverse=0))
nodes.append(helper.make_node("CumSum", ["left_col", "axis3"], ["Dl"], reverse=1))
nodes.append(helper.make_node("CumSum", ["right_col", "axis3"], ["Dr"], reverse=0))

nodes.append(helper.make_node("Mul", ["mf", "above_row"], ["mA"]))
nodes.append(helper.make_node("Mul", ["mf", "below_row"], ["mB"]))
nodes.append(helper.make_node("Mul", ["mf", "left_col"], ["mL"]))
nodes.append(helper.make_node("Mul", ["mf", "right_col"], ["mR"]))
nodes.append(helper.make_node("ReduceSum", ["mA", "axH"], ["Na"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["mB", "axH"], ["Nb"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["mL", "axW"], ["Nl"], keepdims=1))
nodes.append(helper.make_node("ReduceSum", ["mR", "axW"], ["Nr"], keepdims=1))

add_const("half", np.array(0.5, np.float16))
def fill(D, N, out):
    nodes.append(helper.make_node("Greater", [D, "half"], [out + "_ge1"]))    # 1D
    nodes.append(helper.make_node("LessOrEqual", [D, N], [out + "_leN"]))       # grid
    nodes.append(helper.make_node("And", [out + "_ge1", out + "_leN"], [out]))   # grid
fill("Da", "Na", "fa"); fill("Db", "Nb", "fb"); fill("Dl", "Nl", "fl"); fill("Dr", "Nr", "fr")

nodes.append(helper.make_node("Or", ["b_bool", "fa"], ["o1"]))
nodes.append(helper.make_node("Or", ["fb", "fl"], ["o2"]))
nodes.append(helper.make_node("Or", ["o1", "o2"], ["o3"]))
nodes.append(helper.make_node("Or", ["o3", "fr"], ["out5_b"]))

# Where(out5_b, 5, 0) -> gm14 uint8 (drops out5_u8 + Mul)
add_const("five_u8", np.array(5, np.uint8)); add_const("zero_u8", np.array(0, np.uint8))
nodes.append(helper.make_node("Where", ["out5_b", "five_u8", "zero_u8"], ["gm14"]))
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
