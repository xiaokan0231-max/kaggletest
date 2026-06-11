"""Reference solution for task021.

Rule:
  Input contains exactly two colors: a background color and a separator color.
  The separator forms full horizontal rows and full vertical columns that
  partition the grid into rectangular bands (like a table).
  Output is an R x C grid filled entirely with the background color, where
    R = number of horizontal bands (maximal runs of non-separator rows)
    C = number of vertical bands  (maximal runs of non-separator columns).

  Separator identification: the color that forms at least one full uniform
  row (equivalently full uniform column). Background = the other color
  (also the majority color in every example).
"""
import json
import numpy as np


def solve(grid):
    g = np.array(grid)
    h, w = g.shape
    # separator = color of any fully-uniform row (always exists)
    sep = None
    for r in range(h):
        if len(set(g[r].tolist())) == 1:
            sep = int(g[r, 0])
            break
    if sep is None:  # fallback: uniform column
        for c in range(w):
            if len(set(g[:, c].tolist())) == 1:
                sep = int(g[0, c])
                break
    colors = set(g.flatten().tolist())
    bg = [c for c in colors if c != sep][0] if len(colors) == 2 else sep

    row_is_sep = np.all(g == sep, axis=1)  # full separator rows
    col_is_sep = np.all(g == sep, axis=0)  # full separator columns

    def count_bands(is_sep):
        bands = 0
        in_band = False
        for s in is_sep:
            if not s and not in_band:
                bands += 1
                in_band = True
            elif s:
                in_band = False
        return bands

    R = count_bands(row_is_sep)
    C = count_bands(col_is_sep)
    return [[bg] * C for _ in range(R)]


if __name__ == "__main__":
    d = json.load(open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task021.json"))
    total_fail = 0
    for split in ["train", "test", "arc-gen"]:
        ok = 0
        for i, ex in enumerate(d[split]):
            got = solve(ex["input"])
            if got == ex["output"]:
                ok += 1
            else:
                total_fail += 1
                if total_fail <= 5:
                    print(f"FAIL {split}[{i}]")
                    print("input:")
                    for row in ex["input"]:
                        print("".join(map(str, row)))
                    print("expected:", ex["output"])
                    print("got     :", got)
        print(f"{split}: {ok}/{len(d[split])}")
