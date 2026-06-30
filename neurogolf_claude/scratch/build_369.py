"""task369: 0-cells colored by 4-connected component size (max 3).
color = 3 - maxnb, maxnb = max over closed 4-nbhd of deg (deg=#4-conn 0-neighbors),
counting only 0-cells. bg(5) stays 5. All grids 10x10.

KEY: input channel-0 cropped to 10x10 IS the 0-cell mask -> one Slice
(axes=[1,2,3]) gives m [1,1,10,10] fp32 (400B), NO 30x30 fp32 tensor needed.
Interior arithmetic fp16/uint8. 30x30 color tensor uint8 (only big tensor). opset 13.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET = 13
IR = 10
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8

nodes = []
inits = []


def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


# 1) m_f = input[:, 0:1, 0:10, 0:10]  == 0-cell mask at 10x10, fp32 (400B)
C("m_starts", np.array([0, 0, 0], dtype=np.int64))
C("m_ends", np.array([1, 10, 10], dtype=np.int64))
C("m_axes", np.array([1, 2, 3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["input", "m_starts", "m_ends", "m_axes"], ["m_f"], name="getmask"))
# m_f [1,1,10,10] fp32 : 1.0 on 0-cells, 0.0 on bg

# 2) m_bool = (m_f > 0.5)
C("half", np.array([[[[0.5]]]], dtype=np.float32))
nodes.append(helper.make_node("Greater", ["m_f", "half"], ["m_bool"], name="is0"))
# m_bool [1,1,10,10] bool

# 3) deg = Conv(m_f, plus-kernel pad1) : #4-conn 0-neighbors (fp32)
C("w_plus", np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.float32))
nodes.append(helper.make_node("Conv", ["m_f", "w_plus"], ["deg"], name="degree",
                              pads=[1, 1, 1, 1], kernel_shape=[3, 3]))
# deg [1,1,10,10] fp32 (values 0..4)

# 4) cast deg -> uint8, q = Where(m_bool, deg_u, 0) : deg on 0-cells, 0 on bg
nodes.append(helper.make_node("Cast", ["deg"], ["deg_u"], name="deg_to_u8", to=U8))
C("zero_u", np.array([[[[0]]]], dtype=np.uint8))
nodes.append(helper.make_node("Where", ["m_bool", "deg_u", "zero_u"], ["q"], name="q"))
# q [1,1,10,10] uint8 (100B)

# 5) maxnb = max over closed 4-nbhd of q (pad 0, 4 shifts, Max) -- uint8
C("ax23", np.array([2, 3], dtype=np.int64))
C("padpads", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64))
C("padval0", np.array(0, dtype=np.uint8))
nodes.append(helper.make_node("Pad", ["q", "padpads", "padval0"], ["qp"], name="padq", mode="constant"))
C("up_s", np.array([0, 1], dtype=np.int64)); C("up_e", np.array([10, 11], dtype=np.int64))
C("dn_s", np.array([2, 1], dtype=np.int64)); C("dn_e", np.array([12, 11], dtype=np.int64))
C("lf_s", np.array([1, 0], dtype=np.int64)); C("lf_e", np.array([11, 10], dtype=np.int64))
C("rt_s", np.array([1, 2], dtype=np.int64)); C("rt_e", np.array([11, 12], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["qp", "up_s", "up_e", "ax23"], ["s_up"], name="s_up"))
nodes.append(helper.make_node("Slice", ["qp", "dn_s", "dn_e", "ax23"], ["s_dn"], name="s_dn"))
nodes.append(helper.make_node("Slice", ["qp", "lf_s", "lf_e", "ax23"], ["s_lf"], name="s_lf"))
nodes.append(helper.make_node("Slice", ["qp", "rt_s", "rt_e", "ax23"], ["s_rt"], name="s_rt"))
nodes.append(helper.make_node("Max", ["q", "s_up", "s_dn", "s_lf", "s_rt"], ["maxnb"], name="maxnb"))
# maxnb [1,1,10,10] uint8 : 0/1/2 on 0-cells

# 6) color (pure uint8): 0-cells -> 3-maxnb (0->3,1->2,2->1); bg -> 5
C("two_u", np.array([[[[2]]]], dtype=np.uint8))
C("one_u", np.array([[[[1]]]], dtype=np.uint8))
nodes.append(helper.make_node("Equal", ["maxnb", "two_u"], ["mx_eq2"], name="mx_eq2"))
nodes.append(helper.make_node("Equal", ["maxnb", "one_u"], ["mx_eq1"], name="mx_eq1"))
C("c1", np.array([[[[1]]]], dtype=np.uint8))
C("c2", np.array([[[[2]]]], dtype=np.uint8))
C("c3", np.array([[[[3]]]], dtype=np.uint8))
nodes.append(helper.make_node("Where", ["mx_eq1", "c2", "c3"], ["t01"], name="t01"))
nodes.append(helper.make_node("Where", ["mx_eq2", "c1", "t01"], ["tab"], name="tab"))
C("five_u", np.array([[[[5]]]], dtype=np.uint8))
nodes.append(helper.make_node("Where", ["m_bool", "tab", "five_u"], ["col10u"], name="colmix"))
# col10u [1,1,10,10] uint8

# 7) pad to 30x30 with 255 (out-of-grid -> empty)
C("padpads2", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64))
C("padval2", np.array(255, dtype=np.uint8))
nodes.append(helper.make_node("Pad", ["col10u", "padpads2", "padval2"], ["col30u"], name="pad30", mode="constant"))
# col30u [1,1,30,30] uint8 (900B) -- only large interior tensor

# 8) output one-hot: Equal(col30u, colors 0..9) -> bool [1,10,30,30] = OUTPUT (excluded)
C("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
nodes.append(helper.make_node("Equal", ["col30u", "colors"], ["output"], name="onehot"))

graph = helper.make_graph(
    nodes, "task369",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
model.ir_version = IR
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/p1c_task369.onnx")
print("saved. nodes:", len(nodes))
