import json
import torch
import numpy as np

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task066.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train']):
    if i != 2: continue
    inp = torch.tensor(ex['input'], dtype=torch.long)
    H, W = inp.shape
    start_c = 3
    
    sy, sx = torch.where(inp == start_c)
    tips = []
    for y, x in zip(sy, sx):
        neighbors = []
        for dy, dx, d_idx in [(-1, 0, 3), (1, 0, 1), (0, -1, 2), (0, 1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and inp[ny, nx] == start_c:
                neighbors.append(d_idx)
        if len(neighbors) == 1:
            out_dir = (neighbors[0] + 2) % 4
            tips.append((y.item(), x.item(), out_dir))
            
    print(f"Ex {i} tips: {tips}")
