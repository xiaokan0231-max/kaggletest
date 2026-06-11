"""Reference numpy solution for task026.

Rule: input is 5x7. Column 3 is a separator of 1s (blue). Left half =
cols 0..2, right half = cols 4..6, both 5x3 grids of {0, 9}. Output is
5x3: cell = 8 where BOTH halves are 0 (logical NOR of the 9-masks),
else 0.
"""
import json
import numpy as np

DATA = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task026.json"


def solve(grid):
    a = np.array(grid)
    left = a[:, 0:3]
    right = a[:, 4:7]
    out = np.where((left == 0) & (right == 0), 8, 0)
    return out.tolist()


def main():
    d = json.load(open(DATA))
    total_fail = 0
    for split in ("train", "test", "arc-gen"):
        ok = 0
        for i, ex in enumerate(d[split]):
            got = solve(ex["input"])
            if got == ex["output"]:
                ok += 1
            else:
                total_fail += 1
                if total_fail <= 3:
                    print(f"FAIL {split}[{i}]")
                    print("input:")
                    for row in ex["input"]:
                        print("".join(map(str, row)))
                    print("expected:")
                    for row in ex["output"]:
                        print("".join(map(str, row)))
                    print("got:")
                    for row in got:
                        print("".join(map(str, row)))
        print(f"{split}: {ok}/{len(d[split])}")


if __name__ == "__main__":
    main()
