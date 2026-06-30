import torch
import torch.nn.functional as F
import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

for ex_idx, ex in enumerate(data['train'] + data['test']):
    inp = torch.tensor(ex['input'])
    out = torch.tensor(ex['output'])

    orig_H, orig_W = inp.shape
    MAX_SIZE = 30
    padded_inp = torch.zeros(MAX_SIZE, MAX_SIZE)
    padded_inp[:orig_H, :orig_W] = inp

    # Find unique colors
    unique_colors, counts = torch.unique(padded_inp[:orig_H, :orig_W], return_counts=True)
    
    # Exclude 0
    mask = unique_colors != 0
    unique_colors = unique_colors[mask]
    counts = counts[mask]
    
    # In Train0, counts are [4, 4, 4, 15] for 1,3,4,8.
    # So anchors are colors with the minimum count?
    # Wait, the number of configurations is `num_configs`.
    # Let's say anchors are the 3 colors that have exactly the same count.
    # Actually, simpler: anchors are the 3 colors with smallest counts.
    sorted_counts, indices = torch.sort(counts)
    anchor_colors = unique_colors[indices[:3]].tolist()
    structure_color = unique_colors[indices[-1]].item()

    A = torch.zeros(10, MAX_SIZE, MAX_SIZE)
    for c in anchor_colors:
        A[c] = (padded_inp == c).float()

    S = (padded_inp == structure_color).float().unsqueeze(0).unsqueeze(0)
    A = A.unsqueeze(0)

    out_S = torch.zeros(1, 1, MAX_SIZE, MAX_SIZE)
    out_A = torch.zeros(1, 10, MAX_SIZE, MAX_SIZE) # We also need to copy the anchors!

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
        A_pad = F.pad(A, (29, 29, 29, 29))
        
        C = F.conv2d(A_pad, weight)
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
                                # We also want to copy the anchors
                                for c in range(10):
                                    out_A[0, c, y, x] = max(out_A[0, c, y, x].item(), A_g[0, c, sy, sx].item())

    # out_S has all structures. We want only the naked ones.
    # The naked ones are where out_S is 1 but S is 0?
    naked_S = F.relu(out_S - S)
    
    # We also need to keep the anchors for the naked ones!
    # How to know which anchors belong to the naked ones?
    # If an anchor is close to naked_S ?
    # Alternatively, the original anchors A are everywhere. 
    # The completed anchors are the ones close to S.
    # If we dilate S by, say, 5 pixels, we can mask out the completed anchors!
    # Wait, even simpler:
    # A naked configuration is one where (out_S - S) matches the structure!
    # But how to extract the anchors?
    # We want to output ONLY the naked objects.
    # We can just construct the output image.
    pred_out = torch.zeros(MAX_SIZE, MAX_SIZE)
    pred_out[naked_S[0,0] > 0] = structure_color
    
    # For anchors, we only want those that are NOT near original S!
    # Actually, the completed anchors are exactly `A` where `C(v) >= 3` with `v` such that `S_g` shifted matches `S`.
    # Wait, we know `S`. We want to delete `S` AND the anchors of the completed object!
    # A completed anchor is an anchor pixel that is touching or near `S`.
    # Since the structure is bounded, we can just dilate S!
    # Let's see if dilating S by a small window covers the completed anchors.
    # How far can an anchor be from S?
    
    print(f"--- Example {ex_idx} ---")
    print("Anchor colors:", anchor_colors)
    print("Structure color:", structure_color)
