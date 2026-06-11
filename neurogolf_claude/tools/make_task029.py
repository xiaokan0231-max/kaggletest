"""task029 ONNX generator.

Rule: input contains exactly one color forming a 1-cell-thick hollow rectangle
(that color appears nowhere else). Output = strict interior of the frame.

Detection is done purely on row/col marginals (no [10,30,30] outer products):
a channel mask equals its bbox perimeter iff
  row counts == (W, 2, 2, ..., 2, W) over the bbox rows, and
  col counts == (H, 2, 2, ..., 2, H) over the bbox cols.
Proof: col count H at the two edge columns forces a cell of every bbox row at
those columns; middle rows have count 2, so those are exactly their two cells;
first/last rows are full (count W). That is precisely the hollow frame.
Degenerate guard: strict interior must be non-empty in both axes (kills empty
channels, single-row/col channels, and 2-wide solid blocks).

Extraction: standard SHRINK permutation matrices built from keep indicators
via CumSum:  output = S @ mask @ T^t  (MatMul broadcasts over the 10 channels).

opset 11 (CumSum), ir_version 10. All Sign inputs are non-negative integers.
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path("/Users/kanxiao/IdeaProjects/kaggletest")
OUT = ROOT / "neurogolf_claude" / "solutions" / "task029.onnx"
RAW = ROOT / "neurogolf" / "data" / "raw" / "task029.json"


def axis_pipeline(nodes, any_name, suffix):
    """Emit cs/tot/aft/strict/inbb tensors for one axis. Returns names."""
    cs, tot, aft = f"cs_{suffix}", f"tot_{suffix}", f"aft_{suffix}"
    nodes += [
        helper.make_node("CumSum", [any_name, "ax1"], [cs]),
        helper.make_node("ReduceSum", [any_name], [tot], axes=[1], keepdims=1),
        helper.make_node("Sub", [tot, cs], [aft]),
        # strictly inside: rows after the first AND before the last
        helper.make_node("Sub", [cs, any_name], [f"bef_{suffix}"]),
        helper.make_node("Sign", [f"bef_{suffix}"], [f"sbef_{suffix}"]),
        helper.make_node("Sign", [aft], [f"saft_{suffix}"]),
        helper.make_node("Mul", [f"sbef_{suffix}", f"saft_{suffix}"],
                         [f"strict_{suffix}"]),
        # inside bbox (inclusive): cs>0 AND (aft+any)>0
        helper.make_node("Add", [aft, any_name], [f"hi_{suffix}"]),
        helper.make_node("Sign", [f"hi_{suffix}"], [f"shi_{suffix}"]),
        helper.make_node("Sign", [cs], [f"slo_{suffix}"]),
        helper.make_node("Mul", [f"slo_{suffix}", f"shi_{suffix}"],
                         [f"inbb_{suffix}"]),
    ]
    return f"strict_{suffix}", f"inbb_{suffix}"


def expected_counts(nodes, suffix, span_name):
    """exp = 2*strict + span*(inbb - strict); span is [10,1] (W for rows)."""
    nodes += [
        helper.make_node("Sub", [f"inbb_{suffix}", f"strict_{suffix}"],
                         [f"edge_{suffix}"]),
        helper.make_node("Mul", [f"edge_{suffix}", span_name],
                         [f"e1_{suffix}"]),
        helper.make_node("Mul", [f"strict_{suffix}", "two"],
                         [f"e2_{suffix}"]),
        helper.make_node("Add", [f"e1_{suffix}", f"e2_{suffix}"],
                         [f"exp_{suffix}"]),
    ]


def count_diff(nodes, cnt_name, suffix):
    nodes += [
        helper.make_node("Sub", [cnt_name, f"exp_{suffix}"], [f"d_{suffix}"]),
        helper.make_node("Abs", [f"d_{suffix}"], [f"ad_{suffix}"]),
        helper.make_node("ReduceSum", [f"ad_{suffix}"], [f"diff_{suffix}"],
                         axes=[1], keepdims=0),
    ]


def perm_matrix(nodes, keep_name, p_name, out_name, row_side):
    """Build permutation matrix from keep indicator [1,30] and p [1,30]."""
    if row_side:
        # S[i,r] = keep[r] * (iota_col[i] == p[r])
        nodes += [
            helper.make_node("Sub", ["iota_col", p_name], [f"pd_{out_name}"]),
            helper.make_node("Abs", [f"pd_{out_name}"], [f"pa_{out_name}"]),
            helper.make_node("Sign", [f"pa_{out_name}"], [f"ps_{out_name}"]),
            helper.make_node("Sub", ["one", f"ps_{out_name}"],
                             [f"pe_{out_name}"]),
            helper.make_node("Mul", [f"pe_{out_name}", keep_name], [out_name]),
        ]
    else:
        # Tt[c,j] = keep[c] * (p[c] == iota_row[j]); reshape inputs to [30,1]
        nodes += [
            helper.make_node("Reshape", [p_name, "shp_c31"],
                             [f"pT_{out_name}"]),
            helper.make_node("Reshape", [keep_name, "shp_c31"],
                             [f"kT_{out_name}"]),
            helper.make_node("Sub", [f"pT_{out_name}", "iota_row"],
                             [f"pd_{out_name}"]),
            helper.make_node("Abs", [f"pd_{out_name}"], [f"pa_{out_name}"]),
            helper.make_node("Sign", [f"pa_{out_name}"], [f"ps_{out_name}"]),
            helper.make_node("Sub", ["one", f"ps_{out_name}"],
                             [f"pe_{out_name}"]),
            helper.make_node("Mul", [f"pe_{out_name}", f"kT_{out_name}"],
                             [out_name]),
        ]


def build_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Reshape", ["input", "shp_m"], ["m"]),
        helper.make_node("ReduceMax", ["m"], ["row_any"], axes=[2],
                         keepdims=0),
        helper.make_node("ReduceMax", ["m"], ["col_any"], axes=[1],
                         keepdims=0),
        helper.make_node("ReduceSum", ["m"], ["row_cnt"], axes=[2],
                         keepdims=0),
        helper.make_node("ReduceSum", ["m"], ["col_cnt"], axes=[1],
                         keepdims=0),
    ]
    axis_pipeline(nodes, "row_any", "r")
    axis_pipeline(nodes, "col_any", "c")
    nodes += [
        helper.make_node("ReduceSum", ["inbb_r"], ["H"], axes=[1], keepdims=1),
        helper.make_node("ReduceSum", ["inbb_c"], ["W"], axes=[1], keepdims=1),
    ]
    expected_counts(nodes, "r", "W")
    expected_counts(nodes, "c", "H")
    count_diff(nodes, "row_cnt", "r")
    count_diff(nodes, "col_cnt", "c")
    nodes += [
        helper.make_node("Add", ["diff_r", "diff_c"], ["diff"]),
        helper.make_node("Sign", ["diff"], ["sdiff"]),
        helper.make_node("Sub", ["one", "sdiff"], ["fmatch"]),
        # guards: strict interior non-empty on both axes
        helper.make_node("ReduceSum", ["strict_r"], ["nr"], axes=[1],
                         keepdims=0),
        helper.make_node("ReduceSum", ["strict_c"], ["nc"], axes=[1],
                         keepdims=0),
        helper.make_node("Sign", ["nr"], ["snr"]),
        helper.make_node("Sign", ["nc"], ["snc"]),
        helper.make_node("Mul", ["fmatch", "snr"], ["f0"]),
        helper.make_node("Mul", ["f0", "snc"], ["f1"]),
        helper.make_node("Reshape", ["f1", "shp_f"], ["fsel"]),  # [1,10]
        # keep indicators of the frame channel
        helper.make_node("MatMul", ["fsel", "strict_r"], ["keep_r"]),  # [1,30]
        helper.make_node("MatMul", ["fsel", "strict_c"], ["keep_c"]),
        helper.make_node("CumSum", ["keep_r", "ax1"], ["ck_r"]),
        helper.make_node("Sub", ["ck_r", "one"], ["p_r"]),
        helper.make_node("CumSum", ["keep_c", "ax1"], ["ck_c"]),
        helper.make_node("Sub", ["ck_c", "one"], ["p_c"]),
    ]
    perm_matrix(nodes, "keep_r", "p_r", "S", row_side=True)
    perm_matrix(nodes, "keep_c", "p_c", "Tt", row_side=False)
    nodes += [
        helper.make_node("MatMul", ["S", "m"], ["sm"]),     # [10,30,30]
        helper.make_node("MatMul", ["sm", "Tt"], ["o3"]),   # [10,30,30]
        helper.make_node("Reshape", ["o3", "shp_out"], ["output"]),
    ]

    iota = np.arange(30, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(np.array([10, 30, 30], dtype=np.int64),
                                "shp_m"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "ax1"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two"),
        numpy_helper.from_array(iota.reshape(30, 1), "iota_col"),
        numpy_helper.from_array(iota.reshape(1, 30), "iota_row"),
        numpy_helper.from_array(np.array([1, 10], dtype=np.int64), "shp_f"),
        numpy_helper.from_array(np.array([30, 1], dtype=np.int64), "shp_c31"),
        numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64),
                                "shp_out"),
    ]

    shape4 = [1, 10, 30, 30]
    graph = helper.make_graph(
        nodes,
        "task029_frame_interior",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape4)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape4)],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    onnx.checker.check_model(model)
    return model


def to_np(grid):
    a = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            a[0, col, r, c] = 1.0
    return a


def main():
    model = build_model()
    onnx.save(model, OUT)
    print("saved", OUT, OUT.stat().st_size, "bytes")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT),
                                providers=["CPUExecutionProvider"])
    d = json.load(open(RAW))
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        ok = 0
        for i, ex in enumerate(d[split]):
            out = sess.run(["output"], {"input": to_np(ex["input"])})[0]
            got = (out > 0.0).astype(np.float32)
            want = to_np(ex["output"])
            if (got == want).all():
                ok += 1
            else:
                all_ok = False
                if ok + 5 > i:  # print first few failures only
                    print(f"FAIL {split}[{i}]")
        print(f"{split}: {ok}/{len(d[split])}")
    print("ALL PASS" if all_ok else "HAS FAILURES")


if __name__ == "__main__":
    main()
