import json
import numpy as np
import onnxruntime as ort

ONNX = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task022.onnx"
DATA = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task022.json"


def to_np(grid):
    a = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            a[0, col, r, c] = 1.0
    return a


sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
d = json.load(open(DATA))
total_fail = 0
for split in ("train", "test", "arc-gen"):
    ok = 0
    fails = []
    for i, ex in enumerate(d[split]):
        got = (sess.run(["output"], {"input": to_np(ex["input"])})[0] > 0.0).astype(
            np.float32
        )
        exp = to_np(ex["output"])
        if np.array_equal(got, exp):
            ok += 1
        else:
            fails.append(i)
    print(f"{split}: {ok}/{len(d[split])}", "FAIL:" if fails else "", fails[:10])
    total_fail += len(fails)
print("ALL PASS" if total_fail == 0 else f"{total_fail} FAILURES")
