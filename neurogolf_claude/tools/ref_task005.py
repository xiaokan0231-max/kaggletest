"""task005 numpy reference.

Hypothesis:
- Template color t = non-zero color with the most cells.
- M = (grid == t), kept as-is in the output.
- Every other color c marks one or more directions: c's cells always lie
  inside translate(M, 4*d) for exactly one unit direction d per marker group.
- Output = template + for each active (c, d): translate(M, 4*k*d) painted c
  for k = 1.. until the shifted mask is empty (clipped at borders).
"""
import json
import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
PERIOD = 4


def shift(mask, dr, dc):
    out = np.zeros_like(mask)
    h, w = mask.shape
    sr0, sr1 = max(0, dr), min(h, h + dr)
    tr0, tr1 = max(0, -dr), min(h, h - dr)
    out[sr0:sr1, max(0, dc):min(w, w + dc)] = mask[tr0:tr1, max(0, -dc):min(w, w - dc)]
    return out


def window_score(mask):
    """Max number of mask cells inside any 3x3 window."""
    h, w = mask.shape
    best = 0
    for r in range(h - 2):
        for c in range(w - 2):
            best = max(best, mask[r:r + 3, c:c + 3].sum())
    return best


def solve(grid):
    g = np.array(grid)
    scores = [window_score(g == c) for c in range(10)]
    t = int(np.argmax(scores[1:])) + 1
    M = (g == t)
    out = np.where(M, t, 0)
    for c in range(1, 10):
        if c == t:
            continue
        K = (g == c)
        if not K.any():
            continue
        for dr, dc in DIRS:
            if not (K & shift(M, PERIOD * dr, PERIOD * dc)).any():
                continue
            k = 1
            while True:
                S = shift(M, PERIOD * k * dr, PERIOD * k * dc)
                if not S.any():
                    break
                out[S & (out == 0)] = c
                conflict = S & (out != 0) & (out != c)
                if conflict.any():
                    print(f'  CONFLICT color {c} dir {(dr, dc)} k={k}')
                k += 1
    return out


def main():
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task005.json') as f:
        d = json.load(f)
    for split in ['train', 'test', 'arc-gen']:
        ok = bad = 0
        for i, ex in enumerate(d[split]):
            pred = solve(ex['input'])
            if np.array_equal(pred, np.array(ex['output'])):
                ok += 1
            else:
                bad += 1
                if bad <= 3:
                    print(f'{split}[{i}] MISMATCH')
                    diff = (pred != np.array(ex['output']))
                    print('  diff cells:', np.argwhere(diff)[:10].tolist())
        print(f'{split}: {ok} pass, {bad} fail')


if __name__ == '__main__':
    main()
