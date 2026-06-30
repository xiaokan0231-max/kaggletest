import torch
import torch.nn.functional as F
import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

ex = data['train'][0]
inp = torch.tensor(ex['input'])
out = torch.tensor(ex['output'])

orig_H, orig_W = inp.shape

MAX_SIZE = 30
padded_inp = torch.zeros(MAX_SIZE, MAX_SIZE)
padded_inp[:orig_H, :orig_W] = inp

# anchors are colors 1, 3, 4
A = torch.zeros(10, MAX_SIZE, MAX_SIZE)
for c in [1, 3, 4]:
    A[c] = (padded_inp == c).float()

# structure is color 8
S = (padded_inp == 8).float().unsqueeze(0)

A = A.unsqueeze(0) # 1, 10, MAX_SIZE, MAX_SIZE
S = S.unsqueeze(0) # 1, 1, MAX_SIZE, MAX_SIZE

out_S = torch.zeros(1, 1, MAX_SIZE, MAX_SIZE)

def apply_sym(x, sym):
    if sym >= 4:
        x = x.flip(-1)
    k = sym % 4
    x = torch.rot90(x, k, dims=[-2, -1])
    return x

for sym in range(8):
    A_g = apply_sym(A, sym)
    S_g = apply_sym(S, sym)
    
    weight = A_g
    
    # Pad A so that we get all shifts
    # A has shape 30x30. weight is 30x30.
    # To get shift v_y in [-29, 29], we pad A by 29 on all sides.
    A_pad = F.pad(A, (29, 29, 29, 29))
    
    C = F.conv2d(A_pad, weight) # shape 1, 1, 59, 59
    
    M = (C >= 2.99).float()
    
    for dy in range(59):
        for dx in range(59):
            if M[0, 0, dy, dx] > 0:
                v_y = dy - 29
                v_x = dx - 29
                
                for y in range(MAX_SIZE):
                    for x in range(MAX_SIZE):
                        sy = y - v_y
                        sx = x - v_x
                        if 0 <= sy < MAX_SIZE and 0 <= sx < MAX_SIZE:
                            out_S[0, 0, y, x] = max(out_S[0, 0, y, x].item(), S_g[0, 0, sy, sx].item())

pred = out_S[0, 0, :orig_H, :orig_W]
print("Predicted matches True:", torch.allclose(pred, (out == 8).float()))

diff = pred - (out == 8).float()
print("Diff (pred - true):")
r, c = torch.where(diff != 0)
for rr, cc in zip(r, c):
    print(f"  ({rr}, {cc}): {diff[rr, cc]}")
