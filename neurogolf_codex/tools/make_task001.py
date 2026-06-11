from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path("/Users/kanxiao/IdeaProjects/kaggletest")
OUT = ROOT / "neurogolf_codex" / "solutions" / "task001.onnx"


def flat_index(channel: int, row: int, col: int) -> int:
    return channel * 30 * 30 + row * 30 + col


def build_model() -> onnx.ModelProto:
    size = 1 * 10 * 30 * 30
    shape4 = [1, 10, 30, 30]

    pattern_idx = np.zeros(size, dtype=np.int64)
    zero_idx = np.zeros(size, dtype=np.int64)
    valid = np.zeros(size, dtype=np.float32)
    channel0 = np.zeros(size, dtype=np.float32)

    for ch in range(10):
        for row in range(30):
            for col in range(30):
                out_i = flat_index(ch, row, col)
                if row < 9 and col < 9:
                    src_row = row % 3
                    src_col = col % 3
                    block_row = row // 3
                    block_col = col // 3
                    pattern_idx[out_i] = flat_index(ch, src_row, src_col)
                    zero_idx[out_i] = flat_index(0, block_row, block_col)
                    valid[out_i] = 1.0
                    channel0[out_i] = 1.0 if ch == 0 else 0.0

    nodes = [
        helper.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
        helper.make_node("Gather", ["flat", "pattern_idx"], ["pattern"], axis=0),
        helper.make_node("Gather", ["flat", "zero_idx"], ["zero_at_block"], axis=0),
        helper.make_node("Sub", ["ones", "zero_at_block"], ["active_block"]),
        helper.make_node("Mul", ["active_block", "valid"], ["active_valid"]),
        helper.make_node("Mul", ["pattern", "active_valid"], ["colored"]),
        helper.make_node("Sub", ["valid", "active_valid"], ["inactive_valid"]),
        helper.make_node("Mul", ["inactive_valid", "channel0"], ["background"]),
        helper.make_node("Add", ["colored", "background"], ["flat_output"]),
        helper.make_node("Reshape", ["flat_output", "out_shape"], ["output"]),
    ]

    initializers = [
        numpy_helper.from_array(np.array([size], dtype=np.int64), "flat_shape"),
        numpy_helper.from_array(np.array(shape4, dtype=np.int64), "out_shape"),
        numpy_helper.from_array(pattern_idx, "pattern_idx"),
        numpy_helper.from_array(zero_idx, "zero_idx"),
        numpy_helper.from_array(valid, "valid"),
        numpy_helper.from_array(channel0, "channel0"),
        numpy_helper.from_array(np.ones(size, dtype=np.float32), "ones"),
    ]

    graph = helper.make_graph(
        nodes,
        "task001_kron_mask",
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
