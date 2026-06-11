"""Probe 'select one tile from a tiling' tasks.

Infers tile layout from output size (with optional 1-cell separators), splits
the grid into a R x C tile grid, finds which tile equals the output per example,
then reports which selection criterion explains the choice across all examples.
"""
import json
import sys
from collections import Counter
import numpy as np

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"


def load(tid):
    d = json.load(open(f"{RAW}/{tid}.json"))
    out = []
    for sp in ("train", "test", "arc-gen"):
        for ex in d.get(sp, []):
            i, o = np.array(ex["input"]), np.array(ex["output"])
            if max(i.shape) <= 30 and max(o.shape) <= 30:
                out.append((i, o))
    return out


def tile_grid(i, h, w):
    """Yield (ri, ci, tile) for an R x C tiling of i into h x w tiles, allowing a
    uniform separator of s in {0,1} cells between tiles. Returns list or None."""
    H, W = i.shape
    for s in (0, 1):
        R = (H + s) // (h + s) if (h + s) else 0
        C = (W + s) // (w + s) if (w + s) else 0
        if R < 1 or C < 1:
            continue
        if R * h + (R - 1) * s != H or C * w + (C - 1) * s != W:
            continue
        tiles = []
        for ri in range(R):
            for ci in range(C):
                r0 = ri * (h + s); c0 = ci * (w + s)
                tiles.append((ri, ci, i[r0:r0 + h, c0:c0 + w]))
        return tiles, R, C, s
    return None


def features(tile):
    nz = tile[tile > 0]
    return {
        "density": int((tile > 0).sum()),
        "ncolors": len(set(nz.tolist())),
        "colorsum": int(nz.sum()),
    }


def probe(tid):
    exs = load(tid)
    # use first example to guess (h,w)
    rows = []
    layouts = set()
    for i, o in exs:
        h, w = o.shape
        tg = tile_grid(i, h, w)
        if not tg:
            rows.append(None); continue
        tiles, R, C, s = tg
        layouts.add((R, C, s))
        match = [k for k, (_, _, t) in enumerate(tiles) if np.array_equal(t, o)]
        rows.append((tiles, R, C, s, match))
    n_ok = sum(1 for r in rows if r and r[4])
    print(f"{tid}: {len(exs)} exs, layouts={sorted(layouts)[:6]}, tile-matches-output in {n_ok}/{len(exs)}")
    if n_ok < len(exs) * 0.9:
        return
    # which criterion picks the matched tile?
    crits = ["dense_max", "dense_min", "ncolors_max", "ncolors_min", "odd",
             "first", "last", "middle"]
    score = Counter()
    total = 0
    for r in rows:
        if not r or not r[4]:
            continue
        tiles, R, C, s, match = r
        total += 1
        ts = [t for _, _, t in tiles]
        feat = [features(t) for t in ts]
        idx = match[0]
        picks = {}
        picks["dense_max"] = max(range(len(ts)), key=lambda k: feat[k]["density"])
        picks["dense_min"] = min(range(len(ts)), key=lambda k: feat[k]["density"])
        picks["ncolors_max"] = max(range(len(ts)), key=lambda k: feat[k]["ncolors"])
        picks["ncolors_min"] = min(range(len(ts)), key=lambda k: feat[k]["ncolors"])
        keys = [t.tobytes() for t in ts]
        cnt = Counter(keys)
        odd = [k for k in range(len(ts)) if cnt[keys[k]] == 1]
        picks["odd"] = odd[0] if len(odd) == 1 else -1
        picks["first"] = 0
        picks["last"] = len(ts) - 1
        picks["middle"] = len(ts) // 2
        for c in crits:
            if picks[c] == idx:
                score[c] += 1
    print("   " + ", ".join(f"{c}={score[c]}/{total}" for c in crits if score[c]))


def main():
    for tid in sys.argv[1:]:
        try:
            probe(tid)
        except Exception as e:
            print(f"{tid}: ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
