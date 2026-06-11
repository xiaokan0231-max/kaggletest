"""Reference solution for task022.

Rule: Input 11x11 contains several small shapes, each anchored by a gray cell
(color 5). The output is a single 3x3 grid formed by overlaying, for every
5-cell in the input, the 3x3 neighborhood of the input centered at that 5
(out-of-bounds treated as 0). Non-zero cells from the different windows are
combined (they never conflict); the center is always 5.
"""
import numpy as np

def solve(grid):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = np.zeros((3, 3), dtype=int)
    rs, cs = np.where(g == 5)
    for r, c in zip(rs, cs):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and g[rr, cc] != 0:
                    out[1 + dr, 1 + dc] = max(out[1 + dr, 1 + dc], g[rr, cc])
    return out.tolist()

if __name__ == "__main__":
    import json
    d = json.load(open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task022.json"))
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
        print(f"{split}: {ok}/{len(d[split])} pass", "FAIL idx:" if fails else "", fails[:10])
        total_fail += len(fails)
        for i in fails[:3]:
            ex = d[split][i]
            print("--- fail", split, i)
            for row in ex["input"]: print("".join(map(str, row)))
            print("expected:")
            for row in ex["output"]: print("".join(map(str, row)))
            print("got:")
            for row in solve(ex["input"]): print("".join(map(str, row)))
    print("ALL PASS" if total_fail == 0 else f"{total_fail} FAILURES")
