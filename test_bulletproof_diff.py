import json
import numpy as np
from scipy.ndimage import label

def transform(x, k):
    if k == 0: return x
    if k == 2: return np.rot90(x, 2)
    if k == 4: return np.flip(x, 1)
    if k == 6: return np.flip(x, 0)
    return x

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

dirs = [
    (-1, 0, 6), (1, 0, 6), (0, -1, 4), (0, 1, 4),
    (-1, -1, 2), (-1, 1, 2), (1, -1, 2), (1, 1, 2)
]

ex = data['train'][0]
inp = np.array(ex['input'])
out = np.array(ex['output'])
counts = np.bincount(inp.flatten())
bg = np.argmax(counts)
out_pred = np.zeros_like(inp)
out_pred[inp == bg] = bg

for c in range(10):
    if c == bg: continue
    mask = (inp == c)
    labeled, num = label(mask)
    for j in range(1, num+1):
        B_mask = (labeled == j)
        if B_mask.sum() <= 1: continue
        r_idx, c_idx = np.where(B_mask)
        rm, rM = r_idx.min(), r_idx.max()
        cm, cM = c_idx.min(), c_idx.max()
        kH = rM - rm + 1
        kW = cM - cm + 1
        B_crop = B_mask[rm:rM+1, cm:cM+1]
        out_pred[B_mask] = c
        for dr, dc, k in dirs:
            B_shifted = np.roll(B_mask, shift=(dr, dc), axis=(0, 1))
            if dr > 0: B_shifted[:dr, :] = False
            elif dr < 0: B_shifted[dr:, :] = False
            if dc > 0: B_shifted[:, :dc] = False
            elif dc < 0: B_shifted[:, dc:] = False
            S = B_shifted & (inp != bg) & ~mask
            if S.sum() > 0:
                ac_colors = np.unique(inp[S])
                for ac in ac_colors:
                    # check if the specific color is exactly in S
                    if (inp == ac)[S].sum() > 0:
                        tr_rm = rm + dr * kH
                        tr_cm = cm + dc * kW
                        trans_crop = transform(B_crop, k)
                        for rr in range(kH):
                            for cc in range(kW):
                                if trans_crop[rr, cc]:
                                    if 0 <= tr_rm + rr < inp.shape[0] and 0 <= tr_cm + cc < inp.shape[1]:
                                        out_pred[tr_rm + rr, tr_cm + cc] = ac

diff = (out_pred != out)
for r, c in zip(*np.where(diff)):
    print(f"Diff at ({r}, {c}): Pred {out_pred[r, c]}, True {out[r, c]}")
