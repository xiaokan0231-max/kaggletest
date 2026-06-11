"""task006 reference: 3x7 input, column 3 is a separator of 5s.
Left half = cols 0-2 (colors 0/1), right half = cols 4-6 (colors 0/1).
Output 3x3: cell = 2 where BOTH halves are nonzero (logical AND), else 0.
"""
import json
import numpy as np

DATA = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task006.json"


def solve(grid):
    g = np.asarray(grid)
    left = g[:, 0:3] != 0
    right = g[:, 4:7] != 0
    return ((left & right) * 2).tolist()


def main():
    d = json.load(open(DATA))
    total_ok = True
    for split in ["train", "test", "arc-gen"]:
        ok = 0
        fails = []
        for i, ex in enumerate(d[split]):
            if solve(ex["input"]) == ex["output"]:
                ok += 1
            else:
                fails.append(i)
        print(f"{split}: {ok}/{len(d[split])} fails={fails[:10]}")
        if fails:
            total_ok = False
    print("ALL PASS" if total_ok else "FAILED")


if __name__ == "__main__":
    main()
