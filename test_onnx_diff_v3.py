import torch
import torch.nn.functional as F
import json
import numpy as np

def roll2d(x, shifts):
    return torch.roll(x, shifts=shifts, dims=(0, 1))

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

ex = data['train'][0]
inp = np.array(ex['input'])
out = np.array(ex['output'])

H, W = inp.shape
x_ts = torch.tensor(inp, dtype=torch.float32)
counts = torch.bincount(x_ts.flatten().long())
bg_c = torch.argmax(counts).item()

out_pred = torch.zeros((H, W), dtype=torch.float32)
out_pred[x_ts == bg_c] = bg_c

R = torch.arange(H).view(H, 1).expand(H, W).float()
C = torch.arange(W).view(1, W).expand(H, W).float()
k_8 = torch.ones(1, 1, 3, 3)
k_8[0, 0, 1, 1] = 0

for c in range(10):
    if c == bg_c: continue
    
    mask_c = (x_ts == c).float()
    has_neighbor = F.conv2d(mask_c.view(1, 1, H, W), k_8, padding=1).view(H, W) > 0.5
    B_c = mask_c * has_neighbor.float()
    if B_c.sum() <= 1: continue
    
    rm = torch.min(R * B_c + 1000 * (1 - B_c))
    rM = torch.max(R * B_c)
    cm = torch.min(C * B_c + 1000 * (1 - B_c))
    cM = torch.max(C * B_c)
    kH = rM - rm + 1
    kW = cM - cm + 1
    
    dr_grid = torch.floor((R - rm) / kH)
    dc_grid = torch.floor((C - cm) / kW)
    flip_r = (dr_grid != 0).float()
    flip_c = (dc_grid != 0).float()
    
    r_loc = R - rm - dr_grid * kH
    c_loc = C - cm - dc_grid * kW
    r_src_loc = r_loc * (1 - flip_r) + (kH - 1 - r_loc) * flip_r
    c_src_loc = c_loc * (1 - flip_c) + (kW - 1 - c_loc) * flip_c
    r_src = rm + r_src_loc
    c_src = cm + c_src_loc
    
    idx = (r_src * W + c_src).long()
    idx = torch.clamp(idx, 0, H * W - 1)
    B_flat = B_c.flatten()
    B_tiled = B_flat[idx.flatten()].view(H, W)
    
    S_mask = torch.zeros_like(x_ts)
    for c_any in range(10):
        if c_any == bg_c: continue
        m_any = (x_ts == c_any).float()
        hn_any = F.conv2d(m_any.view(1, 1, H, W), k_8, padding=1).view(H, W) > 0.5
        single_any = m_any * (1 - hn_any.float())
        S_mask += single_any * c_any
        
    S_matched = S_mask * B_tiled
    S_matched = torch.where(B_c > 0.5, torch.tensor(c, dtype=torch.float32), S_matched)
    
    color_mask = S_matched
    for _ in range(15):
        new_color = color_mask.clone()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                shifted_color = roll2d(color_mask, (dr, dc))
                shifted_dr = roll2d(dr_grid, (dr, dc))
                shifted_dc = roll2d(dc_grid, (dr, dc))
                valid = (dr_grid == shifted_dr) & (dc_grid == shifted_dc) & (B_tiled > 0.5)
                if dr == 1: valid[0, :] = False
                if dr == -1: valid[-1, :] = False
                if dc == 1: valid[:, 0] = False
                if dc == -1: valid[:, -1] = False
                new_color = torch.max(new_color, shifted_color * valid.float())
        color_mask = new_color
        
    out_pred = torch.where((B_tiled > 0.5) & (color_mask > 0.5), color_mask, out_pred)

diff = (out_pred != torch.tensor(out, dtype=torch.float32))
for r, c_idx in zip(*torch.where(diff)):
    print(f"Diff at ({r}, {c_idx}): Pred {out_pred[r, c_idx]}, True {out[r, c_idx]}")
