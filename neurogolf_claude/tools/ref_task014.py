"""task014 numpy reference.

Rule: the input grid is tiled with rectangular blocks (separated by all-zero
rows/columns). All blocks but one are noisy patterns of a single common color;
exactly one block uses a different (unique) color. Output = the unique-color
block, cropped to its bounding box (internal zeros preserved).

Target color selection: among nonzero colors, the one with the smaller total
cell count (the unique color occupies a single block, the common color many).
"""
import json
import numpy as np

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task014.json"


def solve(grid):
    a = np.array(grid)
    colors, counts = np.unique(a[a > 0], return_counts=True)
    target = colors[np.argmin(counts)]
    mask = a == target
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].tolist()


def main():
    d = json.load(open(RAW))
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
    print("ALL PASS" if total_fail == 0 else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
