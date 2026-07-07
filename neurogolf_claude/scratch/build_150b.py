"""task150 v2: width-aware horizontal flip via Gather on axis=3 (output excluded).
Goal: beat 668 bytes. Minimize number & dtype of [1,30] interior tensors.

Algorithm:
  - colcount = ReduceSum(input over channels+rows) -> per-column occupancy [1,30] (f32). occupied col -> H>0.
  - occ = Greater(colcount, 0) -> bool [1,30]  (which columns occupied)
  - W = ReduceSum(occ as int32) -> scalar [1,1]
  - rng = [0..29] int32 const
  - fi = (W-1) - rng        (flip index for occupied region)
  - idx = Where(rng < W, fi, rng)  -> int32 [30] (squeeze)
  - output = Gather(input, idx, axis=3)
Reductions in dtype tricks to shrink: cast occupancy to int8 early? ReduceSum needs numeric.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H = Wd = 30
N = 10


def const(name, arr):
    return numpy_helper.from_array(arr, name=name)


nodes = []
inits = []

# Sum input over channels (axis1) and rows (axis2) -> [1,1,30] then we want [1,30]
inits.append(const("ax12", np.array([1, 2], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["input", "ax12"], ["colsum"], name="colsum", keepdims=0))  # [1,30] f32

# occupied columns: colsum > 0
inits.append(const("zero_f", np.array(0.0, dtype=np.float32)))
nodes.append(helper.make_node("Greater", ["colsum", "zero_f"], ["occ"], name="occ"))  # bool [1,30]

# W = sum of occ. Cast bool->int32 then ReduceSum
nodes.append(helper.make_node("Cast", ["occ"], ["occi"], name="occi", to=TensorProto.INT32))  # [1,30] int32
inits.append(const("ax1", np.array([1], dtype=np.int64)))
nodes.append(helper.make_node("ReduceSum", ["occi", "ax1"], ["W"], name="W", keepdims=1))  # [1,1] int32

# rng [1,30] int32 const 0..29
rng = np.arange(Wd, dtype=np.int32).reshape(1, Wd)
inits.append(const("rng", rng))

# fi = (W-1) - rng
inits.append(const("one_i", np.array(1, dtype=np.int32)))
nodes.append(helper.make_node("Sub", ["W", "one_i"], ["Wm1"], name="Wm1"))  # [1,1] int32
nodes.append(helper.make_node("Sub", ["Wm1", "rng"], ["fi"], name="fi"))  # [1,30] int32

# mask rng < W
nodes.append(helper.make_node("Less", ["rng", "W"], ["inside"], name="inside"))  # bool [1,30]
nodes.append(helper.make_node("Where", ["inside", "fi", "rng"], ["i2d"], name="i2d"))  # [1,30] int32

# squeeze to [30]
inits.append(const("ax0", np.array([0], dtype=np.int64)))
nodes.append(helper.make_node("Squeeze", ["i2d", "ax0"], ["idx"], name="idx"))  # [30] int32

# gather along width -> output
nodes.append(helper.make_node("Gather", ["input", "idx"], ["output"], name="output", axis=3))

graph = helper.make_graph(
    nodes, "task150",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, N, H, Wd])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N, H, Wd])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b17_task150.onnx")
print("saved")
