import torch
import torch.nn.functional as F
import json

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
    A = A.unsqueeze(0)

    pred_out = torch.zeros(10, MAX_SIZE, MAX_SIZE)

    def apply_sym(x, sym):
        if sym >= 4:
            x = x.flip(-1)
        k = sym % 4
        x = torch.rot90(x, k, dims=[-2, -1])
        return x

    for sym in range(8):
        A_g = apply_sym(A, sym)
        S_g = apply_sym(S, sym)
        
        A_pad = F.pad(A, (29, 29, 29, 29))
        C = F.conv2d(A_pad, A_g)
        M = (C >= 2.99).float()
        
        for dy in range(59):
            for dx in range(59):
                if M[0, 0, dy, dx] > 0:
                    v_y = dy - 29
                    v_x = dx - 29
                    
                    # Compute S_map
                    S_map = torch.zeros(MAX_SIZE, MAX_SIZE)
                    for y in range(MAX_SIZE):
                        for x in range(MAX_SIZE):
                            sy = y - v_y
                            sx = x - v_x
                            if 0 <= sy < MAX_SIZE and 0 <= sx < MAX_SIZE:
                                S_map[y, x] = S_g[0, 0, sy, sx]
                                
                    overlap = (S_map * S[0, 0]).sum().item()
                    
                    # Check if it's a naked configuration
                    # A completed config will have overlap > 0.
                    # Wait, if S_g has NO structure pixels, overlap is 0. 
                    # But S_g comes from S, so S_g has structure pixels!
                    # What if a naked config maps to a naked config? Then S_map will be 0 everywhere?
                    # No, S_g is the mapped ENTIRE S grid!
                    # Wait! S_g contains ALL structure pixels of the input.
                    # If we shift S_g by v_y, v_x, we are shifting ALL structure pixels.
                    # This means S_map will contain the structure pixels of the completed object, placed at the naked object!
                    # BUT it will ALSO contain other structure pixels shifted by the same amount, which might fall out of bounds or randomly hit something.
                    # For the specific completed object that matches the naked object, its structure pixels in S_map will fall on the naked object.
                    # Since the naked object has NO structure pixels in S, those specific pixels will have 0 overlap with S.
                    # What about the other structure pixels in S_g? Will they overlap with S?
                    # It's possible but unlikely. To be safe, we should ONLY look at the structure pixels belonging to the MATCHED configuration!
                    # Wait, how to isolate the structure pixels of ONE completed object?
                    
                    pass

