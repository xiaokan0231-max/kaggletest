import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskNet066(nn.Module):
    def forward(self, x):
        m3 = x[:, 3:4, :, :]
        m2 = x[:, 2:3, :, :]
        m8 = x[:, 8:9, :, :]
        
        # Base cost is 1.0
        cost = torch.ones_like(m3)
        # Hugging cost is 0.99
        adj_8 = F.max_pool2d(m8, kernel_size=3, stride=1, padding=1)
        cost = torch.where(adj_8 > 0.5, torch.full_like(cost, 0.99), cost)
        
        dist = torch.full_like(m3, 999.0)
        dist = torch.where(m3 > 0.5, torch.zeros_like(dist), dist)
        
        for _ in range(45):
            up = F.pad(dist, (0, 0, 1, 0), value=999.0)[:, :, :-1, :]
            down = F.pad(dist, (0, 0, 0, 1), value=999.0)[:, :, 1:, :]
            left = F.pad(dist, (1, 0, 0, 0), value=999.0)[:, :, :, :-1]
            right = F.pad(dist, (0, 1, 0, 0), value=999.0)[:, :, :, 1:]
            
            min_n = torch.min(torch.min(up, down), torch.min(left, right))
            new_dist = min_n + cost
            dist = torch.min(dist, new_dist)
            dist = torch.where(m8 > 0.5, torch.full_like(dist, 999.0), dist)
            
        dist2 = torch.full_like(m2, 999.0)
        dist2 = torch.where(m2 > 0.5, torch.zeros_like(dist2), dist2)
        
        for _ in range(45):
            up = F.pad(dist2, (0, 0, 1, 0), value=999.0)[:, :, :-1, :]
            down = F.pad(dist2, (0, 0, 0, 1), value=999.0)[:, :, 1:, :]
            left = F.pad(dist2, (1, 0, 0, 0), value=999.0)[:, :, :, :-1]
            right = F.pad(dist2, (0, 1, 0, 0), value=999.0)[:, :, :, 1:]
            
            min_n = torch.min(torch.min(up, down), torch.min(left, right))
            new_dist = min_n + cost
            dist2 = torch.min(dist2, new_dist)
            dist2 = torch.where(m8 > 0.5, torch.full_like(dist2, 999.0), dist2)
            
        total_dist = dist + dist2
        min_dist = torch.min(total_dist.view(x.shape[0], -1), dim=1)[0].view(-1, 1, 1, 1)
        
        path = (torch.abs(total_dist - min_dist) < 0.005).float()
        
        out_colors = torch.zeros_like(x)
        out_colors[:, 3:4, :, :] = path
        
        mask_0 = x[:, 0:1, :, :]
        overwrite = path * mask_0
        
        out_colors[:, 3:4, :, :] = overwrite
        out_colors = out_colors + x * (1.0 - overwrite)
        
        return out_colors

def test():
    with open("neurogolf/data/raw/task066.json") as f:
        data = json.load(f)
    
    net = TaskNet066()
    net.eval()
    
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        
        h, w = inp.shape
        x = torch.zeros((1, 10, h, w))
        for r in range(h):
            for c in range(w):
                x[0, int(inp[r,c]), r, c] = 1.0
                
        pred = net(x)
        pred_grid = torch.argmax(pred, dim=1)[0].numpy()
        
        match = np.array_equal(pred_grid, out)
        print(f"Train {i}: {match}")
        if not match:
            diff = (pred_grid != out)
            print(np.argwhere(diff))

if __name__ == "__main__":
    test()
