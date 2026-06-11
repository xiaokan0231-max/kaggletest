"""Reusable ONNX backend for SHRINK crop-and-compact solutions.

Convention: input/output are [1,10,30,30] one-hot. A SHRINK answer is a
rectangular crop of (some transform of) the input, compacted to the top-left
with all-zero cells elsewhere (so the verifier trims to the true H x W).

Core trick (from task014): given keep_row [1,1,30,1] and keep_col [1,1,1,30]
binary masks, build row/col permutation matrices Sr, T and compute
    output[c, i, j] = content[c, kept_row_i, kept_col_j]
via out = Sr @ content @ T  (MatMul broadcasts Sr/T across the 10 channels).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def base_inits():
    """Constants shared by the compaction backend."""
    return [
        const("ck_iota_r", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
        const("ck_iota_c", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
        const("ck_one", np.float32(1.0)),
        const("ck_ax3", np.array(3, np.int64)),
        const("ck_sh_w", np.array([1, 1, 1, 30], np.int64)),
        const("ck_sh_h", np.array([1, 1, 30, 1], np.int64)),
    ]


def emit_compaction(n, keep_row, keep_col, content, out_name):
    """Append nodes computing out = Sr(keep_row) @ content @ T(keep_col).

    keep_row: name of [1,1,30,1] binary tensor (1 = keep this row).
    keep_col: name of [1,1,1,30] binary tensor (1 = keep this col).
    content:  name of [1,K,30,30] tensor to crop (usually 'input').
    """
    p = out_name  # unique prefix to avoid node-name clashes across calls
    # Sr: row permutation gathering kept rows to the top
    n("Reshape", [keep_row, "ck_sh_w"], [f"{p}_krw"])           # [1,1,1,30]
    n("CumSum", [f"{p}_krw", "ck_ax3"], [f"{p}_pr0"])
    n("Sub", [f"{p}_pr0", "ck_one"], [f"{p}_pr"])
    n("Equal", ["ck_iota_r", f"{p}_pr"], [f"{p}_ErB"])          # [1,1,30,30]
    n("Cast", [f"{p}_ErB"], [f"{p}_ErF"], to=TensorProto.FLOAT)
    n("Mul", [f"{p}_ErF", f"{p}_krw"], [f"{p}_Sr"])
    # T: col permutation gathering kept cols to the left
    n("CumSum", [keep_col, "ck_ax3"], [f"{p}_pc0"])
    n("Sub", [f"{p}_pc0", "ck_one"], [f"{p}_pcw"])
    n("Reshape", [f"{p}_pcw", "ck_sh_h"], [f"{p}_pc"])          # [1,1,30,1]
    n("Reshape", [keep_col, "ck_sh_h"], [f"{p}_kc"])            # [1,1,30,1]
    n("Equal", [f"{p}_pc", "ck_iota_c"], [f"{p}_EcB"])         # [1,1,30,30]
    n("Cast", [f"{p}_EcB"], [f"{p}_EcF"], to=TensorProto.FLOAT)
    n("Mul", [f"{p}_EcF", f"{p}_kc"], [f"{p}_T"])
    # out = Sr @ content @ T
    n("MatMul", [f"{p}_Sr", content], [f"{p}_cc"])
    n("MatMul", [f"{p}_cc", f"{p}_T"], [out_name])


def finalize(nodes, inits, name, out_path, in_name="input", out_name="output"):
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print("saved", out_path, len(model.SerializeToString()), "bytes")
    return model
