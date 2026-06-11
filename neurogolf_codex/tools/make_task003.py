from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "neurogolf_codex" / "solutions" / "task003.onnx"


def flat_index(channel: int, row: int, col: int) -> int:
    return channel * 30 * 30 + row * 30 + col


def build_model() -> onnx.ModelProto:
    size = 1 * 10 * 30 * 30
    shape4 = [1, 10, 30, 30]

    gather_idx = np.zeros(size, dtype=np.int64)
    valid = np.zeros(size, dtype=np.float32)

    row_map = [0, 1, 2, 3, 4, 5, 2, 3, 4]

    for ch in range(10):
        for row in range(30):
            for col in range(30):
                out_i = flat_index(ch, row, col)
                if row < 9 and col < 3:
                    src_row = row_map[row]
                    if ch == 0:
                        gather_idx[out_i] = flat_index(0, src_row, col)
                        valid[out_i] = 1.0
                    elif ch == 2:
                        gather_idx[out_i] = flat_index(1, src_row, col)
                        valid[out_i] = 1.0

    nodes = [
        helper.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
        helper.make_node("Gather", ["flat", "gather_idx"], ["picked"], axis=0),
        helper.make_node("Mul", ["picked", "valid"], ["flat_output"]),
        helper.make_node("Reshape", ["flat_output", "out_shape"], ["output"]),
    ]

    initializers = [
        numpy_helper.from_array(np.array([size], dtype=np.int64), "flat_shape"),
        numpy_helper.from_array(np.array(shape4, dtype=np.int64), "out_shape"),
        numpy_helper.from_array(gather_idx, "gather_idx"),
        numpy_helper.from_array(valid, "valid"),
    ]

    graph = helper.make_graph(
        nodes,
        "task003_vertical_period_recolor",
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
