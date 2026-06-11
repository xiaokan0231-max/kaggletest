"""Reference numpy implementation for task124.

Rule: input is the top H rows (H in 5..8) of a vertically periodic pattern on a
10-wide grid. Rows repeat with vertical period p and horizontal shift s:
    row[r+p][c] == row[r][c-s]  (zero fill where c-s out of [0,W))
Extend the pattern downward to a 10x10 output:
    out[r] = shift(out[r-p], s) for r >= H.
Detection: smallest p (1..H-1); for that p, the s (search order 0, +1, -1, +2,
-2, ...) consistent across every row pair (r, r+p) of the input.
"""
import numpy as np


def shift_row(row, s, W):
    out = np.zeros(W, dtype=row.dtype)
    if s >= 0:
        if s < W:
            out[s:] = row[: W - s]
    else:
        if -s < W:
            out[: W + s] = row[-s:]
    return out


def find_period(a):
    H, W = a.shape
    for p in range(1, H):
        for s in [0] + [x for k in range(1, W) for x in (k, -k)]:
            ok = True
            for r in range(H - p):
                if not np.array_equal(a[r + p], shift_row(a[r], s, W)):
                    ok = False
                    break
            if ok:
                return p, s
    return None


def solve(grid):
    a = np.array(grid, dtype=int)
    H, W = a.shape
    OUT_H = 10
    p, s = find_period(a)
    out = np.zeros((OUT_H, W), dtype=int)
    out[:H] = a
    for r in range(H, OUT_H):
        out[r] = shift_row(out[r - p], s, W)
    return out.tolist()


if __name__ == "__main__":
    import json

    d = json.load(open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task124.json"))
    total_fail = 0
    for split in ["train", "test", "arc-gen"]:
        n = len(d[split])
        fails = []
        for i, ex in enumerate(d[split]):
            got = solve(ex["input"])
            if got != ex["output"]:
                fails.append(i)
        total_fail += len(fails)
        print(f"{split}: {n - len(fails)}/{n} pass", ("FAIL idx: " + str(fails[:10])) if fails else "")
        for i in fails[:3]:
            ex = d[split][i]
            print("--- failing", split, i, "p,s =", find_period(np.array(ex["input"])))
            print("in:")
            for row in ex["input"]:
                print("".join(map(str, row)))
            print("expected:")
            for row in ex["output"]:
                print("".join(map(str, row)))
            print("got:")
            for row in solve(ex["input"]):
                print("".join(map(str, row)))
    print("ALL PASS" if total_fail == 0 else f"TOTAL FAILS: {total_fail}")
