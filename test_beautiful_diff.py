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
        B = (labeled == j)
        if B.sum() <= 1: continue
        out_pred[B] = c
        for dr, dc, k in dirs:
            B_shifted = np.roll(B, shift=(dr, dc), axis=(0, 1))
            if dr > 0: B_shifted[:dr, :] = False
            elif dr < 0: B_shifted[dr:, :] = False
            if dc > 0: B_shifted[:, :dc] = False
            elif dc < 0: B_shifted[:, dc:] = False
            
            S = B_shifted & (inp != bg) & ~mask
            adj_colors = np.unique(inp[S])
            for ac in adj_colors:
                ac_mask = (inp == ac) & S
                if ac_mask.sum() > 0:
                    B_trans = transform(B, k)
                    B_stamp = np.roll(B_trans, shift=(dr*3, dc*3), axis=(0, 1))
                    if dr*3 > 0: B_stamp[:dr*3, :] = False
                    elif dr*3 < 0: B_stamp[dr*3:, :] = False
                    if dc*3 > 0: B_stamp[:, :dc*3] = False
                    elif dc*3 < 0: B_stamp[:, dc*3:] = False
                    out_pred[B_stamp] = ac

diff = (out_pred != out)
print("Rows changed:", np.unique(np.where(diff)[0]))
print("Cols changed:", np.unique(np.where(diff)[1]))

for r, c in zip(*np.where(diff)):
    print(f"Diff at ({r}, {c}): Pred {out_pred[r, c]}, True {out[r, c]}")

