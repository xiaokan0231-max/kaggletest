import torch
import torch.nn.functional as F
import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    H, W = inp.shape
    x = torch.zeros(1, 10, H, W)
    for c in range(10):
        x[0, c] = torch.tensor(inp == c, dtype=torch.float32)
        
    counts = torch.sum(x, dim=(2,3))[0]
    bg_c = torch.argmax(counts).item()
    
    k_neighbors = torch.ones(1, 1, 3, 3)
    k_neighbors[0, 0, 1, 1] = 0
    
    for c in range(10):
        if c == bg_c: continue
        mask = x[:, c:c+1]
        neighbors = F.conv2d(mask, k_neighbors, padding=1)
        
        single_pixels = mask * (neighbors == 0)
        base_shapes = mask * (neighbors > 0)
        
        if torch.sum(single_pixels) > 0:
            print(f"Train {i}, Color {c}: {int(torch.sum(single_pixels).item())} single pixels, {int(torch.sum(base_shapes).item())} base pixels")

