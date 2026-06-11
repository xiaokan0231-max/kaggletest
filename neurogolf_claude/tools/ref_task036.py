"""task036 numpy reference.

Rule hypothesis:
- Input 30x30 contains scattered noise pixels of one or more colors, plus one
  dense cluster (a contiguous blob) of a single "target" color.
- Target color = the color whose cells achieve the maximum count inside any
  3x3 window (window-density argmax).
- Output = bounding box crop of the target color's cells; inside the bbox,
  cells equal to the target color keep it, everything else becomes 0.
"""
import json
import numpy as np


def solve(grid):
    a = np.array(grid)
    best_c, best_score = None, -1
    H, W = a.shape
    for c in range(1, 10):
        m = (a == c).astype(int)
        if m.sum() == 0:
            continue
        # max 3x3 window sum
        p = np.zeros((H + 2, W + 2), dtype=int)
        p[1:-1, 1:-1] = m
        win = sum(
            p[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        )
        s = win.max()
        if s > best_score:
            best_score, best_c = s, c
    m = a == best_c
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    sub = a[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    res = np.where(sub == best_c, best_c, 0)
    return res.tolist()


if __name__ == "__main__":
    d = json.load(open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task036.json"))
    total_fail = 0
    for split in ("train", "test", "arc-gen"):
        ok = 0
        fails = []
        for i, ex in enumerate(d[split]):
            got = solve(ex["input"])
            if got == ex["output"]:
                ok += 1
            else:
                fails.append(i)
        total_fail += len(fails)
        print(f"{split}: {ok}/{len(d[split])} fails={fails[:10]}")
    print("ALL PASS" if total_fail == 0 else "HAS FAILURES")
