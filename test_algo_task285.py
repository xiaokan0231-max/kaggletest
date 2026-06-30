import torch
import torch.nn.functional as F
import json
import numpy as np

def transform(x, k):
    if k == 0: return x
    if k == 2: return torch.rot90(x, 2, [2, 3])
    if k == 4: return torch.flip(x, [3])
    if k == 6: return torch.flip(x, [2])
    return x

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    H, W = inp.shape
    x = torch.zeros(1, 10, H, W)
    for c in range(10):
        x[0, c] = torch.tensor(inp == c, dtype=torch.float32)
        
    counts = torch.sum(x, dim=(2,3))[0]
    bg_c = torch.argmax(counts).item()
    
    out_pred = torch.zeros_like(x)
    out_pred[0, bg_c] = 1.0
    
    # We can process each color channel as a base!
    # Wait, the component is connected.
    # To find the base, we can just find connected components.
    # But in ONNX we can't easily find connected components!
    # CAN WE JUST DO THIS PER COLOR CHANNEL GLOBALLY?
    # What if we assume EVERY pixel of color C is the base shape for color C?
    # Then `B_c = x[:, c]`.
    # But `B_c` might have multiple disconnected components!
    # If we just use `B_c` as the base shape, it's a global mask!
    pass
