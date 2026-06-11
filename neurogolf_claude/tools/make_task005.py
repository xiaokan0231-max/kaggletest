"""task005 ONNX generator.

Rule: a template shape (the color with the densest 3x3 window) plus partial
"marker" copies offset 4 cells away in up to 8 directions. Output repeats the
template along each marked direction (period 4, recolored to the marker
color, clipped at the grid border).

Fully one-hot pipeline, opset 10:
- template selector: AveragePool 3x3 -> spatial ReduceMax -> 1 + Sign(x - max)
- 8-direction shifts: one [8,1,9,9] delta Conv fan-out + depthwise cascades
- direction activation: MatMul(colors [9,900], shifts^T [900,8]) -> Sign
- stamping: MatMul(activation [9,8], shift-sums [8,900]), masked to the grid
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path("/Users/kanxiao/IdeaProjects/kaggletest")
OUT = ROOT / "neurogolf_claude" / "solutions" / "task005.onnx"

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def shift_kernel() -> np.ndarray:
    w = np.zeros((8, 1, 9, 9), dtype=np.float32)
    for d, (dr, dc) in enumerate(DIRS):
        w[d, 0, 4 - 4 * dr, 4 - 4 * dc] = 1.0
    return w


def build_model() -> onnx.ModelProto:
    pads = [4, 4, 4, 4]
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes1"], ["x19"]),
        helper.make_node("ReduceSum", ["input"], ["grid"], axes=[1], keepdims=1),
        helper.make_node("AveragePool", ["x19"], ["box"], kernel_shape=[3, 3]),
        helper.make_node("ReduceMax", ["box"], ["wscore"], axes=[2, 3], keepdims=1),
        helper.make_node("ReduceMax", ["wscore"], ["wmax"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["wscore", "wmax"], ["wdiff"]),
        helper.make_node("Sign", ["wdiff"], ["wsign"]),
        helper.make_node("Add", ["wsign", "one"], ["tsel"]),
        helper.make_node("Mul", ["x19", "tsel"], ["xt"]),
        helper.make_node("ReduceSum", ["xt"], ["tmpl"], axes=[1], keepdims=1),
        helper.make_node("Conv", ["tmpl", "wfan"], ["s1"], pads=pads),
        helper.make_node("Conv", ["s1", "wdw"], ["s2"], pads=pads, group=8),
        helper.make_node("Conv", ["s2", "wdw"], ["s3"], pads=pads, group=8),
        helper.make_node("Conv", ["s3", "wdw"], ["s4"], pads=pads, group=8),
        helper.make_node("Conv", ["s4", "wdw"], ["s5"], pads=pads, group=8),
        helper.make_node("Add", ["s1", "s2"], ["t12"]),
        helper.make_node("Add", ["s3", "s4"], ["t34"]),
        helper.make_node("Add", ["t12", "t34"], ["t14"]),
        helper.make_node("Add", ["t14", "s5"], ["tsum"]),
        helper.make_node("Reshape", ["x19", "shp_k"], ["kflat"]),
        helper.make_node("Reshape", ["s1", "shp_s"], ["s1flat"]),
        helper.make_node("Transpose", ["s1flat"], ["s1t"], perm=[1, 0]),
        helper.make_node("MatMul", ["kflat", "s1t"], ["ov"]),
        helper.make_node("Sign", ["ov"], ["act"]),
        helper.make_node("Reshape", ["tsum", "shp_s"], ["tsumflat"]),
        helper.make_node("MatMul", ["act", "tsumflat"], ["stamp"]),
        helper.make_node("Reshape", ["stamp", "shp_y"], ["ypre"]),
        helper.make_node("Mul", ["ypre", "grid"], ["ymask"]),
        helper.make_node("Mul", ["tmpl", "tsel"], ["ty"]),
        helper.make_node("Add", ["ymask", "ty"], ["y19"]),
        helper.make_node("ReduceMax", ["y19"], ["colored"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["grid", "colored"], ["bg"]),
        helper.make_node("Concat", ["bg", "y19"], ["output"], axis=1),
    ]

    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([10], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(shift_kernel(), "wfan"),
        numpy_helper.from_array(shift_kernel(), "wdw"),
        numpy_helper.from_array(np.array([9, 900], dtype=np.int64), "shp_k"),
        numpy_helper.from_array(np.array([8, 900], dtype=np.int64), "shp_s"),
        numpy_helper.from_array(np.array([1, 9, 30, 30], dtype=np.int64), "shp_y"),
    ]

    shape4 = [1, 10, 30, 30]
    graph = helper.make_graph(
        nodes,
        "task005_ray_stamp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape4)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape4)],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), OUT)
    print(OUT)


if __name__ == "__main__":
    main()
