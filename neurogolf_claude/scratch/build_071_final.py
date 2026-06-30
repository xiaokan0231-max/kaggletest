import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

N = 16
F16 = TensorProto.FLOAT16
nodes = []; inits = []
def add_init(name, arr):
    inits.append(nh.from_array(np.asarray(arr), name=name)); return name

# ===== identify the two present colors from free input counts =====
add_init("axes_hw", np.array([2,3], dtype=np.int64))
nodes.append(helper.make_node("ReduceSum", ["input","axes_hw"], ["cnt"], keepdims=0))  # [1,10] f32
add_init("f0", np.array(0.0, dtype=np.float32))
nodes.append(helper.make_node("Greater", ["cnt","f0"], ["present"]))  # [1,10] bool (ch0 also true)
add_init("colors_vec", np.arange(10, dtype=np.float32).reshape(1,10))
# c_max = max present color (ch0 contributes 0)
nodes.append(helper.make_node("Cast", ["present"], ["present_f"], to=TensorProto.FLOAT))
nodes.append(helper.make_node("Mul", ["present_f","colors_vec"], ["pcolors"]))  # [1,10]
add_init("axis1v", np.array([1], dtype=np.int64))
nodes.append(helper.make_node("ReduceMax", ["pcolors","axis1v"], ["c_max"], keepdims=1))  # [1,1] f32
# c_min = min present nonzero color: where(present & nonzero, color, 99) then ReduceMin
add_init("chan_nz", np.array([[0]+[1]*9], dtype=bool))
nodes.append(helper.make_node("And", ["present","chan_nz"], ["present_nz"]))  # exclude ch0
add_init("big99", np.array([[99.0]*10], dtype=np.float32))
nodes.append(helper.make_node("Where", ["present_nz","colors_vec","big99"], ["pcolors_min"]))
nodes.append(helper.make_node("ReduceMin", ["pcolors_min","axis1v"], ["c_min"], keepdims=1))  # [1,1] f32

# ===== color grid via Conv, crop to g16 =====
add_init("convw", np.arange(10, dtype=np.float32).reshape(1,10,1,1))
nodes.append(helper.make_node("Conv", ["input","convw"], ["g30"]))  # [1,1,30,30] f32 =3600B
add_init("gstarts", np.array([0,0,0,0], dtype=np.int64))
add_init("gends", np.array([1,1,N,N], dtype=np.int64))
add_init("gaxes", np.array([0,1,2,3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["g30","gstarts","gends","gaxes"], ["g16"]))  # [1,1,16,16] f32 =1024B
add_init("shape4", np.array([1,1,1,1], dtype=np.int64))
nodes.append(helper.make_node("Reshape", ["c_min","shape4"], ["cmin4"]))
nodes.append(helper.make_node("Reshape", ["c_max","shape4"], ["cmax4"]))
nodes.append(helper.make_node("Equal", ["g16","cmin4"], ["maskA"]))  # bool [1,1,16,16]
nodes.append(helper.make_node("Equal", ["g16","cmax4"], ["maskB"]))

# ===== solidity per mask via tiny reductions; occluder = solid one =====
add_init("axes_w", np.array([3], dtype=np.int64)); add_init("axes_h", np.array([2], dtype=np.int64))
def solidity(mask, tag):
    nodes.append(helper.make_node("Cast", [mask], [tag+"_f"], to=F16))  # [1,1,16,16] f16 =512B
    nodes.append(helper.make_node("ReduceMax", [tag+"_f","axes_w"], [tag+"_rmax"], keepdims=0))  # [1,1,16]
    nodes.append(helper.make_node("ReduceMax", [tag+"_f","axes_h"], [tag+"_cmax"], keepdims=0))  # [1,1,16]
    nodes.append(helper.make_node("ReduceSum", [tag+"_rmax","axis1v_t"], [tag+"_nr"], keepdims=1))  # [1,1]
    nodes.append(helper.make_node("ReduceSum", [tag+"_cmax","axis1v_t"], [tag+"_nc"], keepdims=1))  # [1,1]
    nodes.append(helper.make_node("Mul", [tag+"_nr", tag+"_nc"], [tag+"_area"]))  # [1,1]
    nodes.append(helper.make_node("ReduceSum", [tag+"_f","axes_hw"], [tag+"_cnt"], keepdims=1))  # [1,1]
    nodes.append(helper.make_node("Equal", [tag+"_cnt", tag+"_area"], [tag+"_solid"]))  # [1,1] bool
add_init("axis1v_t", np.array([2], dtype=np.int64))  # reduce axis for [1,1,16]
solidity("maskA","A")
solidity("maskB","B")
# occ_color = A_solid ? c_min : c_max ; shape_color = the other
nodes.append(helper.make_node("Where", ["A_solid","cmin4","cmax4"], ["occ4"]))   # [1,1,1,1]
nodes.append(helper.make_node("Where", ["A_solid","cmax4","cmin4"], ["shc4"]))
nodes.append(helper.make_node("Equal", ["g16","shc4"], ["shape_b"]))  # bool [1,1,16,16]
nodes.append(helper.make_node("Equal", ["g16","occ4"], ["occ_b"]))
# ===== conflict & overlap via Conv2D (reuse A_f/B_f f16 masks) =====
# obj_h = A_f + B_f (disjoint union of the two colors)
nodes.append(helper.make_node("Add", ["A_f","B_f"], ["obj_h"]))   # [1,1,16,16] f16
# shape_f = non-solid mask (A_solid -> B is shape, else A)
nodes.append(helper.make_node("Where", ["A_solid","B_f","A_f"], ["shape_h"]))  # [1,1,16,16] f16
# kernel = shape flipped along cols
add_init("flip_starts", np.array([N-1], dtype=np.int64))
add_init("flip_ends", np.array([-N-1], dtype=np.int64))
add_init("flip_axes", np.array([3], dtype=np.int64))
add_init("flip_steps", np.array([-1], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["shape_h","flip_starts","flip_ends","flip_axes","flip_steps"], ["kern"]))  # [1,1,16,16] f16
# Conv2D pad cols only -> output [1,1,1,31]; rows summed by full-height kernel.
# over[a] = sum shape[r,a-c]*obj[r,c]; total[a] = sum shape[r,a-c] (in-grid count) via ones input.
add_init("ones_in", np.ones((1,1,N,N), dtype=np.float16))
nodes.append(helper.make_node("Conv", ["ones_in","kern"], ["total_vec"], pads=[0,N-1,0,N-1], kernel_shape=[N,N]))  # [1,1,1,31] f16
nodes.append(helper.make_node("Conv", ["obj_h","kern"], ["over_vec"], pads=[0,N-1,0,N-1], kernel_shape=[N,N]))

# ===== best axis: conflict-free <=> over==total ; pick max overlap =====
nodes.append(helper.make_node("Equal", ["over_vec","total_vec"], ["cf_mask"]))  # bool [1,1,1,31]
add_init("neg1h", np.array(-1.0, dtype=np.float16))
nodes.append(helper.make_node("Where", ["cf_mask","over_vec","neg1h"], ["score"]))  # [1,1,31] f16
nodes.append(helper.make_node("ArgMax", ["score"], ["a_idx"], axis=3, keepdims=1))  # [1,1,1,1] int64
nodes.append(helper.make_node("Cast", ["a_idx"], ["a_idx32"], to=TensorProto.INT32))
add_init("a_reshape", np.array([1,1,1], dtype=np.int64))
nodes.append(helper.make_node("Reshape", ["a_idx32","a_reshape"], ["a_star_i"]))  # [1,1,1] int32

# ===== reflect shape about a* (int32 index math) =====
add_init("cvec", np.arange(N, dtype=np.int32).reshape(1,1,N))
nodes.append(helper.make_node("Sub", ["a_star_i","cvec"], ["srcidx"]))  # [1,1,N] int32
add_init("i0", np.array(0, dtype=np.int32)); add_init("iN", np.array(N, dtype=np.int32)); add_init("iNm1", np.array(N-1, dtype=np.int32))
nodes.append(helper.make_node("GreaterOrEqual", ["srcidx","i0"], ["ge0"]))
nodes.append(helper.make_node("Less", ["srcidx","iN"], ["ltN"]))
nodes.append(helper.make_node("And", ["ge0","ltN"], ["validcol"]))  # bool [1,1,N]
nodes.append(helper.make_node("Max", ["srcidx","i0"], ["sc_lo"]))
nodes.append(helper.make_node("Min", ["sc_lo","iNm1"], ["src_clamped"]))
add_init("flatN", np.array([N], dtype=np.int64))
nodes.append(helper.make_node("Reshape", ["src_clamped","flatN"], ["src_idx_flat"]))  # [N] int32
nodes.append(helper.make_node("Gather", ["shape_b","src_idx_flat"], ["refl_raw"], axis=3))  # bool [1,1,N,N]
add_init("vshape", np.array([1,1,1,N], dtype=np.int64))
nodes.append(helper.make_node("Reshape", ["validcol","vshape"], ["validcol4"]))
nodes.append(helper.make_node("And", ["refl_raw","validcol4"], ["refl"]))
nodes.append(helper.make_node("And", ["refl","occ_b"], ["added"]))
nodes.append(helper.make_node("Or", ["shape_b","added"], ["out_shape_b"]))  # bool [1,1,N,N]

# ===== output one-hot =====
nodes.append(helper.make_node("Cast", ["shc4"], ["shc_u8"], to=TensorProto.UINT8))
add_init("zero_u8", np.array(0, dtype=np.uint8))
nodes.append(helper.make_node("Where", ["out_shape_b","shc_u8","zero_u8"], ["gm16"]))  # uint8 [1,1,N,N]
add_init("pads3030", np.array([0,0,0,0, 0,0,30-N,30-N], dtype=np.int64))
add_init("sent_u8", np.array(255, dtype=np.uint8))
nodes.append(helper.make_node("Pad", ["gm16","pads3030","sent_u8"], ["gm30"], mode="constant"))  # uint8 [1,1,30,30]
add_init("colors_u8", np.arange(10, dtype=np.uint8).reshape(1,10,1,1))
nodes.append(helper.make_node("Equal", ["gm30","colors_u8"], ["output"]))  # bool [1,10,30,30]

graph = helper.make_graph(nodes, "task071",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])],
    [helper.make_tensor_value_info("output", TensorProto.BOOL, [1,10,30,30])], inits)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
onnx.checker.check_model(m)
onnx.save(m, "/tmp/b9_task071.onnx")
print("saved, nodes", len(nodes))
