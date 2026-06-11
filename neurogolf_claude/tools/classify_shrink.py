"""Battery of numpy hypotheses to classify each SHRINK task's rule.

Tests crop/quadrant/fold/downscale families against ALL examples (train+test+
all arc-gen) and prints the first rule that explains 100% of examples.
"""
import json
import sys
import numpy as np
from itertools import product

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"


def load(tid):
    d = json.load(open(f"{RAW}/{tid}.json"))
    exs = []
    for sp in ("train", "test", "arc-gen"):
        for ex in d.get(sp, []):
            i, o = np.array(ex["input"]), np.array(ex["output"])
            if max(i.shape) <= 30 and max(o.shape) <= 30:
                exs.append((i, o))
    return exs


def nz_bbox(a):
    m = a > 0
    if not m.any():
        return None
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return r[0], r[-1] + 1, c[0], c[-1] + 1


# ---- separator-based tiling ----
def split_tiles(a):
    """Split grid into tiles by separator lines (full rows/cols of a single uniform color).
    Returns (tiles 2d list, row_groups, col_groups, sep_color) or None."""
    H, W = a.shape
    # a separator row: all cells equal (uniform). candidate sep color = that value.
    def uniform_rows(a):
        return [r for r in range(a.shape[0]) if len(set(a[r])) == 1]
    urows = [r for r in range(H) if len(np.unique(a[r])) == 1]
    ucols = [c for c in range(W) if len(np.unique(a[:, c])) == 1]
    # group sep colors: pick the color that forms BOTH some full row and full col, or the most frequent uniform line color
    line_colors = [a[r][0] for r in urows] + [a[0][c] for c in ucols]
    if not line_colors:
        return None
    from collections import Counter
    sep = Counter(line_colors).most_common(1)[0][0]
    sr = [r for r in urows if a[r][0] == sep]
    sc = [c for c in ucols if a[0][c] == sep]
    if not sr and not sc:
        return None
    # row segments between sep rows
    def segments(n, seps):
        segs = []
        start = 0
        for s in sorted(seps) + [n]:
            if s > start:
                segs.append((start, s))
            start = s + 1
        return [seg for seg in segs if seg[1] > seg[0]]
    rsegs = segments(H, sr) or [(0, H)]
    csegs = segments(W, sc) or [(0, W)]
    tiles = [[a[r0:r1, c0:c1] for (c0, c1) in csegs] for (r0, r1) in rsegs]
    return tiles, rsegs, csegs, sep


def flat_tiles(tt):
    return [t for row in tt for t in row]


def test_odd_quadrant(exs):
    """output == the unique tile among separator-split tiles (all others equal)."""
    for i, o in exs:
        r = split_tiles(i)
        if not r:
            return False
        tiles = flat_tiles(r[0])
        if len(tiles) < 2:
            return False
        shapes = set(t.shape for t in tiles)
        if len(shapes) != 1:
            return False
        # find tile that differs from the majority
        keys = [t.tobytes() for t in tiles]
        from collections import Counter
        cnt = Counter(keys)
        if len(cnt) != 2:
            return False
        rare = min(cnt, key=lambda k: cnt[k])
        if cnt[rare] != 1:
            return False
        odd = tiles[keys.index(rare)]
        if not np.array_equal(odd, o):
            return False
    return True


def test_fold(exs, op):
    """output == elementwise fold (max/overlay) of all separator-split equal-shape tiles."""
    for i, o in exs:
        r = split_tiles(i)
        if not r:
            return False
        tiles = flat_tiles(r[0])
        if len(tiles) < 2 or len(set(t.shape for t in tiles)) != 1:
            return False
        st = np.stack(tiles)
        if op == "max":
            f = st.max(0)
        elif op == "overlay":  # nonzero wins; assume no conflict
            f = np.zeros_like(tiles[0])
            for t in tiles:
                f = np.where(t > 0, t, f)
        if not np.array_equal(f, o):
            return False
    return True


def test_halffold(exs, axis, op):
    """fold the two halves along axis (0=rows,1=cols)."""
    for i, o in exs:
        H, W = i.shape
        n = i.shape[axis]
        if n % 2 != 0:
            return False
        if axis == 0:
            a, b = i[:n // 2], i[n // 2:]
        else:
            a, b = i[:, :n // 2], i[:, n // 2:]
        if a.shape != b.shape:
            return False
        if op == "max":
            f = np.maximum(a, b)
        else:
            f = np.where(a > 0, a, b)
        if not np.array_equal(f, o):
            return False
    return True


def test_downscale(exs):
    """input = each output cell expanded to a KxK block (block-constant). Detect K."""
    for K in (2, 3):
        ok = True
        for i, o in exs:
            h, w = o.shape
            if i.shape != (h * K, w * K):
                ok = False; break
            red = i.reshape(h, K, w, K)
            # majority/any nonzero per block -> use the top-left as representative
            rep = red[:, 0, :, 0]
            if not np.array_equal(rep, o):
                # try nonzero-max per block
                rep2 = red.max(axis=(1, 3))
                if not np.array_equal(rep2, o):
                    ok = False; break
        if ok:
            return f"downscale:{K}"
    return None


def test_corner_bbox(exs):
    res = []
    # nz bbox exact
    if all(nz_bbox(i) and np.array_equal(i[slice(*nz_bbox(i)[:2]), slice(*nz_bbox(i)[2:])], o) for i, o in exs):
        res.append("NZBBOX")
    # nz bbox quarters
    for rr, cc in product(["fh", "lh"], ["fh", "lh"]):
        def q(i):
            bb = nz_bbox(i)
            if not bb:
                return None
            r0, r1, c0, c1 = bb
            hh, ww = r1 - r0, c1 - c0
            rs = slice(r0, r0 + hh // 2) if rr == "fh" else slice(r1 - (hh - hh // 2), r1)
            cs = slice(c0, c0 + ww // 2) if cc == "fh" else slice(c1 - (ww - ww // 2), c1)
            return i[rs, cs]
        if all((q(i) is not None and np.array_equal(q(i), o)) for i, o in exs):
            res.append(f"NZBBOX_Q:{rr}{cc}")
    return res


def classify(tid):
    exs = load(tid)
    hits = []
    hits += test_corner_bbox(exs)
    if test_odd_quadrant(exs):
        hits.append("ODD_TILE")
    if test_fold(exs, "max"):
        hits.append("FOLD_max")
    if test_fold(exs, "overlay"):
        hits.append("FOLD_overlay")
    for ax in (0, 1):
        for op in ("max", "overlay"):
            if test_halffold(exs, ax, op):
                hits.append(f"HALFFOLD:ax{ax}:{op}")
    d = test_downscale(exs)
    if d:
        hits.append(d)
    return tid, len(exs), hits


def main():
    for tid in sys.argv[1:]:
        try:
            tid, n, hits = classify(tid)
            print(f"{tid:9s} n={n:3d}  {hits if hits else '— none —'}")
        except Exception as e:
            print(f"{tid:9s} ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
