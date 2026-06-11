"""Numpy reference for task083 (EXPAND_MULTIPLIER family).

Rule: output is a 2H x 2W four-quadrant mirror of the input:
  top-left     = input
  top-right    = horizontal flip of input
  bottom-left  = vertical flip of input
  bottom-right = 180-degree rotation (both flips)
Equivalently: top half = [input | fliplr(input)], bottom half = flipud(top half).

Data note: every example in train/test/arc-gen has input exactly 3x4 -> output 6x8.
"""
import json
import numpy as np

DATA = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task083.json"


def solve(grid):
    a = np.array(grid)
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0).tolist()


def main():
    d = json.load(open(DATA))
    all_ok = True
    for split in ["train", "test", "arc-gen"]:
        ok, fails = 0, []
        for i, ex in enumerate(d[split]):
            if solve(ex["input"]) == ex["output"]:
                ok += 1
            else:
                fails.append(i)
        print(f"{split}: {ok}/{len(d[split])} fails={fails[:10]}")
        all_ok &= not fails
    print("ALL PASS" if all_ok else "FAILED")


if __name__ == "__main__":
    main()
