"""Build ONNX for task083: output = 2Hx2W four-quadrant mirror of input.

All 266 examples are 3x4 -> 6x8, fully static.
Graph (4 nodes, opset 9 so Slice/Pad carry indices as attributes -> 0 extra params):
  input [1,10,30,30]
  -> Slice rows 0:3, cols 0:4              -> [1,10,3,4]   (480 B)
  -> MatMul R[6,3] @ .   (row mirror)      -> [1,10,6,4]   (960 B)
  -> . @ MatMul C[4,8]   (col mirror)      -> [1,10,6,8]   (1920 B)
  -> Pad to [1,10,30,30] (constant 0)      -> output (not counted)
Params: 18 + 32 = 50. Expected ~3360 bytes + 50 params.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task083.onnx"

H, W = 3, 4
ROWS = [0, 1, 2, 2, 1, 0]          # output row i <- input row ROWS[i]
COLS = [0, 1, 2, 3, 3, 2, 1, 0]    # output col j <- input col COLS[j]


def build():
    R = np.zeros((2 * H, H), dtype=np.float32)
    for i, r in enumerate(ROWS):
        R[i, r] = 1.0
    C = np.zeros((W, 2 * W), dtype=np.float32)
    for j, c in enumerate(COLS):
        C[c, j] = 1.0

    nodes = [
        helper.make_node("Slice", ["input"], ["t0"],
                         axes=[2, 3], starts=[0, 0], ends=[H, W]),
        helper.make_node("MatMul", ["R", "t0"], ["t1"]),
        helper.make_node("MatMul", ["t1", "C"], ["t2"]),
        helper.make_node("Pad", ["t2"], ["output"], mode="constant",
                         pads=[0, 0, 0, 0, 0, 0, 30 - 2 * H, 30 - 2 * W],
                         value=0.0),
    ]
    graph = helper.make_graph(
        nodes, "task083",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=[
            helper.make_tensor("R", TensorProto.FLOAT, R.shape, R.flatten()),
            helper.make_tensor("C", TensorProto.FLOAT, C.shape, C.flatten()),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 9)], producer_name="ng")
    model.ir_version = 10
    onnx.save(model, OUT)
    print("saved", OUT)


def selftest():
    import json
    import onnxruntime as ort

    def to_np(grid):
        a = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(grid):
            for c, col in enumerate(row):
                a[0, col, r, c] = 1.0
        return a

    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    d = json.load(open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task083.json"))
    all_ok = True
    for split in ["train", "test", "arc-gen"]:
        fails = []
        for i, ex in enumerate(d[split]):
            out = sess.run(["output"], {"input": to_np(ex["input"])})[0]
            if not np.array_equal((out > 0.0).astype(np.float32), to_np(ex["output"])):
                fails.append(i)
        print(f"{split}: {len(d[split]) - len(fails)}/{len(d[split])} fails={fails[:10]}")
        all_ok &= not fails
    print("SELFTEST ALL PASS" if all_ok else "SELFTEST FAILED")
    return all_ok


if __name__ == "__main__":
    build()
    selftest()
