import torch
import torch.nn.functional as F
import json
import numpy as np

with open('neurogolf/data/raw/task158.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    H, W = inp.shape
    x = torch.zeros(1, 10, H, W)
    for c in range(10):
        x[0, c] = torch.tensor(inp == c, dtype=torch.float32)
        
    counts = torch.sum(x, dim=(2,3))[0]
    bg_c = torch.argmax(counts).item()
    minority_cs = [c for c in range(10) if counts[c] > 0 and c != bg_c]
    
    c_path = -1
    
    k_dil = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=x.device)
    
    for c in minority_cs:
        mask = x[:, c:c+1]
        
        flat_mask = mask.view(-1)
        idx = torch.argmax(flat_mask)
        seed = torch.zeros_like(flat_mask)
        seed[idx] = 1.0
        seed = seed.view(1, 1, H, W) * mask
        
        for _ in range(30):
            dilated = torch.clamp(F.conv2d(seed, k_dil, padding=1), 0, 1)
            seed = torch.clamp(dilated * mask, 0, 1)
            
        if torch.sum(seed) == torch.sum(mask):
            c_path = c
            print(f"Train {i}: c_path = {c}")
            break

