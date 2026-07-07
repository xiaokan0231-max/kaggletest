"""task335 v3: make vert/horiz/seg uint8 (was f32) -> save ~8100B."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b20_task335.onnx"
H = W = 30
nodes, inits = [], []


def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name))
    return name


def n(op, i, o, **kw):
    if isinstance(o, str):
        o = [o]
    if isinstance(i, str):
        i = [i]
    nodes.append(helper.make_node(op, i, o, **kw))
    return o[0]


C("w_col", np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1))
C("ax1", np.array([1], np.int64))
C("ax2", np.array([2], np.int64))
C("ax3", np.array([3], np.int64))
C("tri_d", np.tril(np.ones((H, H), np.float32)))
C("tri_u", np.triu(np.ones((H, H), np.float32)))
C("zerof", np.array(0.0, np.float32))
C("one_u8", np.array(1, np.uint8))
C("four_u8", np.array(4, np.uint8))
C("zero_u8", np.array(0, np.uint8))
C("colors_u", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
C("ch8_s", np.array([8], np.int64)); C("ch8_e", np.array([9], np.int64))
C("ch2_s", np.array([2], np.int64)); C("ch2_e", np.array([3], np.int64))

# collapse
n("Conv", ["input", "w_col"], "g1")                  # f32, ext=0

# presence vectors
n("ReduceMax", ["input", "ax2"], "pmax_col")
n("ReduceMax", ["input", "ax3"], "pmax_row")
n("Slice", ["pmax_col", "ch8_s", "ch8_e", "ax1"], "col8")
n("Slice", ["pmax_col", "ch2_s", "ch2_e", "ax1"], "col2")
n("Slice", ["pmax_row", "ch8_s", "ch8_e", "ax1"], "row8")
n("Slice", ["pmax_row", "ch2_s", "ch2_e", "ax1"], "row2")
n("Max", ["row8", "row2"], "rowmark")
n("Max", ["col8", "col2"], "colmark")

# bands (tiny vectors, keep f32 then cast to u8 vectors)
C("rs30_1", np.array([H, 1], np.int64))
n("Reshape", ["rowmark", "rs30_1"], "rowmark2")
n("MatMul", ["tri_d", "rowmark2"], "rd"); n("MatMul", ["tri_u", "rowmark2"], "ru")
n("Greater", ["rd", "zerof"], "rd_b"); n("Greater", ["ru", "zerof"], "ru_b")
n("And", ["rd_b", "ru_b"], "rband_b")
n("Cast", ["rband_b"], "rband_u8", to=TensorProto.UINT8)
C("rs_r4", np.array([1, 1, H, 1], np.int64))
n("Reshape", ["rband_u8", "rs_r4"], "rowband")       # [1,1,30,1] u8

n("Reshape", ["colmark", "rs30_1"], "colmark2")
n("MatMul", ["tri_d", "colmark2"], "cd"); n("MatMul", ["tri_u", "colmark2"], "cu")
n("Greater", ["cd", "zerof"], "cd_b"); n("Greater", ["cu", "zerof"], "cu_b")
n("And", ["cd_b", "cu_b"], "cband_b")
n("Cast", ["cband_b"], "cband_u8", to=TensorProto.UINT8)
C("rs_c4", np.array([1, 1, 1, W], np.int64))
n("Reshape", ["cband_u8", "rs_c4"], "colband")       # [1,1,1,30] u8

# col8/row2 as u8 vectors for the broadcast product
n("Cast", ["col8"], "col8u", to=TensorProto.UINT8)   # [1,1,1,30]
n("Cast", ["row2"], "row2u", to=TensorProto.UINT8)   # [1,1,30,1]

# segment mask in uint8 (900B each)
n("Mul", ["col8u", "rowband"], "vert")               # [1,1,30,30] u8 0/1
n("Mul", ["row2u", "colband"], "horiz")              # [1,1,30,30] u8 0/1
n("Max", ["vert", "horiz"], "seg")                   # u8 0/1

# colbase uint8: Cast(g1)-1, exterior 0->255
n("Cast", ["g1"], "g1u8", to=TensorProto.UINT8)
n("Sub", ["g1u8", "one_u8"], "colbase")              # u8

# new4 = seg & original-empty (colbase==0)
n("Equal", ["colbase", "zero_u8"], "empty_b")        # bool
n("Cast", ["seg"], "seg_b", to=TensorProto.BOOL)
n("And", ["seg_b", "empty_b"], "new4_b")
n("Where", ["new4_b", "four_u8", "colbase"], "gm")   # u8

n("Equal", ["gm", "colors_u"], "output")             # bool one-hot = graph output

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
outp = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, H, W])
graph = helper.make_graph(nodes, "task335", [inp], [outp], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print("saved", OUT, "nodes", len(nodes), "inits", len(inits))
