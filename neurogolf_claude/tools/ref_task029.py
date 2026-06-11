"""task029 numpy reference: extract interior of the rectangular frame.

Rule: the input contains exactly one hollow rectangle (1-cell-thick frame)
drawn in a color that appears nowhere else in the grid. Output = the cells
strictly inside that frame.
"""
import json
import numpy as np

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task029.json"


def find_frame(a):
    """Return (color, r0, r1, c0, c1) of the frame's bounding box (inclusive)."""
    for color in range(10):
        ys, xs = np.where(a == color)
        if len(ys) == 0:
            continue
        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        # build expected perimeter mask
        m = np.zeros_like(a, dtype=bool)
        m[r0:r1 + 1, c0] = True
        m[r0:r1 + 1, c1] = True
        m[r0, c0:c1 + 1] = True
        m[r1, c0:c1 + 1] = True
        actual = (a == color)
        if (actual == m).all():
            return color, r0, r1, c0, c1
    return None


def solve(grid):
    a = np.array(grid, dtype=int)
    f = find_frame(a)
    assert f is not None, "no frame found"
    _, r0, r1, c0, c1 = f
    return a[r0 + 1:r1, c0 + 1:c1].tolist()


def main():
    d = json.load(open(RAW))
    total_fail = 0
    for split in ("train", "test", "arc-gen"):
        ok = 0
        fails = []
        for i, ex in enumerate(d[split]):
            try:
                got = solve(ex["input"])
            except Exception as e:
                fails.append((i, repr(e)))
                continue
            if got == ex["output"]:
                ok += 1
            else:
                fails.append((i, "mismatch"))
        print(f"{split}: {ok}/{len(d[split])}")
        for i, why in fails[:5]:
            total_fail += 1
            print("  FAIL", split, i, why)
            ex = d[split][i]
            print("  IN:")
            for row in ex["input"]:
                print("   ", " ".join(map(str, row)))
            print("  EXPECTED:")
            for row in ex["output"]:
                print("   ", " ".join(map(str, row)))
            try:
                got = solve(ex["input"])
                print("  GOT:")
                for row in got:
                    print("   ", " ".join(map(str, row)))
            except Exception as e:
                print("  GOT: exception", e)
        total_fail += len(fails)
    if total_fail == 0:
        print("ALL PASS")


if __name__ == "__main__":
    main()
