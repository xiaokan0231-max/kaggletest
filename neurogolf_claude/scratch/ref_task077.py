"""
task077 reference (NOT ONNX-tractable; connected-component / rectangle-detection class).

RULE (color-invariant):
- Each grid has 0 (background), 2 ("marker"), and exactly ONE other nonzero "shape color".
- The shape-color cells + 2 cells together form solid rectangular blocks (every cell
  in the block's bbox is either 2 or shape-color; NO zeros inside).
- For every such solid rectangle that contains >=1 two, recolor its SHAPE-COLOR cells to 4.
  (2 cells stay 2; cells outside any such rectangle stay unchanged.)
- Equivalent local fact: with a 5x5 receptive field (canon 0/shape=1/two=2) the 4-label
  is a deterministic function (0 contradictions over all 266 cases) -- but the function is
  effectively "is this cell inside the stamped solid rectangle object that holds a 2",
  which is GLOBAL object identity, not a simple local geometric predicate.

Best pure-geometric reproduction below = 264/266 (all train+test pass; 2 arc-gen fail
on bridging ambiguities: arc-gen[53], arc-gen[243], where two distinct objects are joined
by an incidental solid bridge and greedy merge can't recover the generator's split).

This requires iterative rectangle merging (connected-component class) => NOT expressible
in compact ONNX (no Loop/Scan/If, no CC labeling). tractability=1.
"""
import numpy as np


def predict(inp):
    H, W = inp.shape
    is2 = (inp == 2)
    shape = (inp != 0) & (inp != 2)
    blk = is2 | shape
    ys, xs = np.where(is2)
    boxes = [[r, r, c, c] for r, c in zip(ys, xs)]

    def area(b):
        return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)

    changed = True
    while changed:
        changed = False
        best = None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                nb = [min(a[0], b[0]), max(a[1], b[1]),
                      min(a[2], b[2]), max(a[3], b[3])]
                if blk[nb[0]:nb[1] + 1, nb[2]:nb[3] + 1].all():
                    ar = area(nb)
                    if best is None or ar < best[0]:
                        best = (ar, i, j, nb)
        if best is not None:
            _, i, j, nb = best
            boxes = [boxes[k] for k in range(len(boxes))
                     if k != i and k != j] + [nb]
            changed = True

    res = np.zeros((H, W), bool)
    for r1, r2, c1, c2 in boxes:
        res[r1:r2 + 1, c1:c2 + 1] = True
    out = inp.copy()
    out[res & shape] = 4
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
    import neurogolf_utils as ng
    ng._NEUROGOLF_DIR = '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
    d = ng.load_examples(77)
    tot = ok = 0
    bad = []
    for k in ['train', 'test', 'arc-gen']:
        for jx, ex in enumerate(d[k]):
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            tot += 1
            if np.array_equal(predict(inp), out):
                ok += 1
            else:
                bad.append((k, jx))
    print(f"reproduces {ok}/{tot}; fails={bad}")
