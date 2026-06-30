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
    
    print(f"Train {i}: bg={bg_c}, min={minority_cs}")
    
    k_dil = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32, device=x.device)
    
    for c in minority_cs:
        mask_c = x[:, c:c+1]
        dilated_c = torch.clamp(F.conv2d(mask_c, k_dil, padding=1), 0, 1)
        touches = []
        for other_c in minority_cs:
            if other_c != c:
                mask_other = x[:, other_c:other_c+1]
                if torch.sum(dilated_c * mask_other) > 0:
                    touches.append(other_c)
        print(f"  Color {c} touches {touches}")

