import torch
import torch.nn.functional as F
import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

for ex_idx, ex in enumerate(data['train'] + data['test']):
    inp = torch.tensor(ex['input']).long()
    out = torch.tensor(ex['output']).long()

    orig_H, orig_W = inp.shape
    MAX_SIZE = 30
    padded_inp = torch.zeros(MAX_SIZE, MAX_SIZE, dtype=torch.long)
    padded_inp[:orig_H, :orig_W] = inp

    unique_colors, counts = torch.unique(padded_inp[:orig_H, :orig_W], return_counts=True)
    mask = unique_colors != 0
    unique_colors = unique_colors[mask]
    counts = counts[mask]
    
    sorted_counts, indices = torch.sort(counts)
    anchor_colors = unique_colors[indices[:3]].tolist()
    structure_color = unique_colors[indices[-1]].item()

    A = torch.zeros(10, MAX_SIZE, MAX_SIZE)
    for c in anchor_colors:
        A[c] = (padded_inp == c).float()

    S = (padded_inp == structure_color).float().unsqueeze(0).unsqueeze(0)
    
    S_dilated = F.max_pool2d(S, kernel_size=3, stride=1, padding=1)
    
    A_comp = A * S_dilated[0]
    A_naked = A * (1 - S_dilated[0])
    
    A_comp = A_comp.unsqueeze(0)
    A_naked = A_naked.unsqueeze(0)

    out_S = torch.zeros(1, 1, MAX_SIZE, MAX_SIZE)

    def apply_sym(x, sym):
        if sym >= 4:
            x = x.flip(-1)
        k = sym % 4
        x = torch.rot90(x, k, dims=[-2, -1])
        return x

    for sym in range(8):
        A_comp_g = apply_sym(A_comp, sym)
        S_g = apply_sym(S, sym)
        
        weight = A_comp_g
        A_pad = F.pad(A_naked, (29, 29, 29, 29))
        
        C = F.conv2d(A_pad, weight)
        M = (C >= 2.99).float()
        
        for dy in range(59):
            for dx in range(59):
                if M[0, 0, dy, dx] > 0:
                    v_y = dy - 29
                    v_x = dx - 29
                    
                    # Compute Match
                    Match = torch.zeros(1, 1, MAX_SIZE, MAX_SIZE)
                    S_shift = torch.zeros(1, 1, MAX_SIZE, MAX_SIZE)
                    for y in range(MAX_SIZE):
                        for x in range(MAX_SIZE):
                            sy = y - v_y
                            sx = x - v_x
                            if 0 <= sy < MAX_SIZE and 0 <= sx < MAX_SIZE:
                                S_shift[0, 0, y, x] = S_g[0, 0, sy, sx]
                                match_val = (A_comp_g[0, :, sy, sx] * A_naked[0, :, y, x]).sum()
                                Match[0, 0, y, x] = match_val
                    
                    # Connected components of S_shift starting from Match
                    Comp = (Match > 0).float()
                    # We also need to include the structure itself
                    for _ in range(30):
                        Comp = F.max_pool2d(Comp, kernel_size=3, stride=1, padding=1) * S_shift
                        
                    out_S = torch.max(out_S, Comp)

    pred_S = out_S[0, 0, :orig_H, :orig_W]
    true_S = (out == structure_color).float()
    
    match = torch.allclose(pred_S, true_S)
    print(f"Ex {ex_idx} structure match: {match}")
    if not match:
        diff = pred_S - true_S
        print("  False positives:", (diff == 1).sum().item())
        print("  False negatives:", (diff == -1).sum().item())

