"""Probe geometric-transform-of-nonzero-bbox rules across tasks."""
import json
import sys
import numpy as np

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"


def load(tid):
    d = json.load(open(f"{RAW}/{tid}.json"))
    return [(np.array(e["input"]), np.array(e["output"]))
            for sp in ("train", "test", "arc-gen") for e in d.get(sp, [])]


def bbox(a):
    m = a > 0
    if not m.any():
        return a
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]


def transforms(s):
    yield "identity", s
    yield "htile2", np.concatenate([s, s], 1)
    yield "vtile2", np.concatenate([s, s], 0)
    yield "up2", np.repeat(np.repeat(s, 2, 0), 2, 1)
    yield "hmirror_cat", np.concatenate([s, s[:, ::-1]], 1)
    yield "hmirror_cat_rev", np.concatenate([s[:, ::-1], s], 1)
    yield "vmirror_cat", np.concatenate([s, s[::-1]], 0)
    yield "transpose", s.T
    yield "rot90", np.rot90(s)
    yield "rot180", np.rot90(s, 2)
    yield "flipud", s[::-1]
    yield "fliplr", s[:, ::-1]
    # 4-fold
    top = np.concatenate([s, s[:, ::-1]], 1)
    yield "fourfold", np.concatenate([top, top[::-1]], 0)


def probe(tid):
    ex = load(tid)
    counts = {}
    for i, o in ex:
        s = bbox(i)
        for name, t in transforms(s):
            if t.shape == o.shape and np.array_equal(t, o):
                counts[name] = counts.get(name, 0) + 1
    good = {k: v for k, v in counts.items() if v >= len(ex) * 0.95}
    if good:
        print(f"{tid}: {good}  (n={len(ex)})")
    else:
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        print(f"{tid}: -- none>=95%; best={top} (n={len(ex)})")


if __name__ == "__main__":
    for tid in sys.argv[1:]:
        probe(tid)
