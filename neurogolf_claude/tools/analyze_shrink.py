"""Auto-categorize SHRINK tasks by detecting common crop/select rules.

For each task, load all examples (train+test+arc-gen sample) and test hypotheses:
  - output is a contiguous subgrid of input (find consistent offset / corner)
  - output == bounding-box crop of a specific color (rarest/most-common nonzero)
  - output == a fixed quadrant / half
  - output == dedup of a tiled/repeated pattern (unique tile)
Prints a one-line verdict per task so families can be batched.
"""
import json
import sys
import numpy as np

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"


def load(tid):
    d = json.load(open(f"{RAW}/{tid}.json"))
    exs = []
    for split in ("train", "test"):
        for ex in d.get(split, []):
            exs.append((np.array(ex["input"]), np.array(ex["output"])))
    # add a few arc-gen for robustness
    for ex in d.get("arc-gen", [])[:20]:
        exs.append((np.array(ex["input"]), np.array(ex["output"])))
    return exs


def find_subgrid_offset(inp, out):
    """Return list of (r,c) offsets where out appears as a subgrid of inp."""
    H, W = inp.shape
    h, w = out.shape
    hits = []
    if h > H or w > W:
        return hits
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if np.array_equal(inp[r:r + h, c:c + w], out):
                hits.append((r, c))
    return hits


def bbox_of(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def test_bbox_color(exs, mode):
    """mode: 'rarest'/'common'/'nonbg-unique' — crop = bbox of selected color."""
    for inp, out in exs:
        nz = inp[inp > 0]
        if len(nz) == 0:
            return False
        colors, counts = np.unique(nz, return_counts=True)
        if mode == "rarest":
            col = colors[np.argmin(counts)]
        elif mode == "common":
            col = colors[np.argmax(counts)]
        else:
            return False
        bb = bbox_of(inp == col)
        if bb is None:
            return False
        r0, r1, c0, c1 = bb
        if not np.array_equal(inp[r0:r1, c0:c1], out):
            return False
    return True


def test_fixed_corner(exs):
    """output is top-left/top-right/bottom-left/bottom-right crop with consistent corner."""
    corners = {"TL": True, "TR": True, "BL": True, "BR": True}
    for inp, out in exs:
        H, W = inp.shape
        h, w = out.shape
        if h > H or w > W:
            return None
        if not np.array_equal(inp[:h, :w], out):
            corners["TL"] = False
        if not np.array_equal(inp[:h, W - w:], out):
            corners["TR"] = False
        if not np.array_equal(inp[H - h:, :w], out):
            corners["BL"] = False
        if not np.array_equal(inp[H - h:, W - w:], out):
            corners["BR"] = False
    good = [k for k, v in corners.items() if v]
    return good or None


def test_shape_relation(exs):
    """Describe how (h,w) relates to (H,W)."""
    rels = []
    for inp, out in exs:
        H, W = inp.shape
        h, w = out.shape
        rels.append((H, W, h, w))
    return rels


def consistent_offset(exs):
    """Is there a single (r,c) corner offset that works for all (incl. relative-to-corner)?"""
    common = None
    for inp, out in exs:
        hits = set(find_subgrid_offset(inp, out))
        common = hits if common is None else (common & hits)
        if not common:
            break
    return common


def analyze(tid):
    exs = load(tid)
    shapes = test_shape_relation(exs)
    # shrink ratio summary
    ratios = set()
    for H, W, h, w in shapes:
        rh = H / h if h else 0
        rw = W / w if w else 0
        ratios.add((round(rh, 2), round(rw, 2)))
    verdicts = []
    if test_bbox_color(exs, "rarest"):
        verdicts.append("BBOX(rarest)")
    if test_bbox_color(exs, "common"):
        verdicts.append("BBOX(common)")
    corners = test_fixed_corner(exs)
    if corners:
        verdicts.append("CORNER:" + ",".join(corners))
    off = consistent_offset(exs)
    if off and not corners:
        verdicts.append(f"FIXED_OFFSET:{sorted(off)[:3]}")
    # is every output a subgrid at all?
    all_sub = all(find_subgrid_offset(i, o) for i, o in exs)
    sample = shapes[0]
    out_shapes = sorted(set((h, w) for _, _, h, w in shapes))
    return {
        "tid": tid,
        "n": len(exs),
        "in_shapes": sorted(set((H, W) for H, W, _, _ in shapes)),
        "out_shapes": out_shapes,
        "ratios": sorted(ratios),
        "all_subgrid": all_sub,
        "verdicts": verdicts or (["SUBGRID(varied)"] if all_sub else ["?UNKNOWN"]),
    }


def main():
    tasks = sys.argv[1:]
    for tid in tasks:
        try:
            r = analyze(tid)
            print(f"{r['tid']:9s} n={r['n']:2d} in={r['in_shapes']} out={r['out_shapes'][:4]} "
                  f"sub={r['all_subgrid']!s:5s} -> {' | '.join(r['verdicts'])}")
        except Exception as e:
            print(f"{tid:9s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
