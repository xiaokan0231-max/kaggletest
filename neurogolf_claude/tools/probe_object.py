"""Probe 'select one object, output its bbox crop' tasks.

Candidates = connected components (8-conn) of nonzero cells. For each, compute a
feature battery; find which argmax/argmin/unique-predicate selects the object
whose bbox crop equals the output, across all examples.
"""
import json
import sys
import numpy as np
from scipy import ndimage
from collections import Counter

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"


def load(tid):
    d = json.load(open(f"{RAW}/{tid}.json"))
    return [(np.array(e["input"]), np.array(e["output"]))
            for sp in ("train", "test", "arc-gen") for e in d.get(sp, [])]


def objects(i, conn=2):
    st = ndimage.generate_binary_structure(2, conn)
    lbl, nl = ndimage.label(i > 0, st)
    objs = []
    for k in range(1, nl + 1):
        m = lbl == k
        r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
        crop = i[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
        objs.append((m, crop, r[0], c[0]))
    return objs


def feats(i, m, crop):
    nz = i[m]
    cc, ct = np.unique(nz, return_counts=True)
    h, w = crop.shape
    bb = crop > 0
    return {
        "size": int(m.sum()), "bbox_area": h * w, "h": h, "w": w,
        "ncolors": len(cc), "maxc": int(cc.max()), "minc": int(cc.min()),
        "fill": int(m.sum()) / (h * w), "is_rect": int((bb.sum()) == h * w),
        "holes": int((crop == 0).sum()), "aspect": h / w if w else 0,
        "perim": int(h * 2 + w * 2),
    }


def probe(tid, conn=2):
    ex = load(tid)
    rows = []
    for i, o in ex:
        objs = objects(i, conn)
        if len(objs) < 2:
            continue
        idx = [k for k, (m, crop, r0, c0) in enumerate(objs) if np.array_equal(crop, o)]
        if len(idx) != 1:
            continue
        fl = [feats(i, m, crop) for (m, crop, r0, c0) in objs]
        rows.append((fl, idx[0]))
    if not rows:
        print(f"{tid} conn{conn}: no clean single-object match")
        return
    keys = list(rows[0][0][0].keys())
    best = []
    for k in keys:
        for mode in ("max", "min", "uniqmax", "uniqmin"):
            ok = 0
            for fl, idx in rows:
                vals = [f[k] for f in fl]
                if mode == "max":
                    pick = int(np.argmax(vals))
                elif mode == "min":
                    pick = int(np.argmin(vals))
                elif mode == "uniqmax":
                    m = max(vals); pick = vals.index(m) if vals.count(m) == 1 else -1
                else:
                    m = min(vals); pick = vals.index(m) if vals.count(m) == 1 else -1
                if pick == idx:
                    ok += 1
            best.append((ok / len(rows), k, mode))
    best.sort(reverse=True)
    print(f"{tid} conn{conn}: {len(rows)}/{len(ex)} clean. top:")
    for sc, k, m in best[:5]:
        print(f"   {sc:.1%}  {m}({k})")


if __name__ == "__main__":
    for tid in sys.argv[1:]:
        for conn in (2, 1):
            probe(tid, conn)
