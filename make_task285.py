import torch
import torch.nn as nn
import torch.nn.functional as F

class Task285(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        
        # bg color
        counts = x.sum(dim=(0, 2, 3), keepdim=True)
        max_counts = torch.max(counts, dim=1, keepdim=True).values
        bg_c_mask = (counts == max_counts).float()
        
        out = out + bg_c_mask
        
        R = torch.arange(H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W).float()
        C_grid = torch.arange(W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W).float()
        
        k_8 = torch.ones(1, 1, 3, 3, device=x.device)
        k_8[0, 0, 1, 1] = 0
        k_dilate = torch.ones(1, 1, 3, 3, device=x.device)
        
        S_mask = torch.zeros_like(x[:, 0:1])
        for c_any in range(10):
            is_not_bg = 1.0 - bg_c_mask[:, c_any:c_any+1]
            m_any = x[:, c_any:c_any+1]
            hn_any = F.conv2d(m_any, k_8, padding=1) > 0.5
            single_any = m_any * (1.0 - hn_any.float()) * is_not_bg
            S_mask = S_mask + single_any * c_any
            
        for c in range(10):
            is_not_bg = 1.0 - bg_c_mask[:, c:c+1]
            mask_c = x[:, c:c+1]
            has_neighbor = F.conv2d(mask_c, k_8, padding=1) > 0.5
            B_c = mask_c * has_neighbor.float() * is_not_bg
            
            has_base = (B_c.sum(dim=(1,2,3), keepdim=True) > 1.5).float()
            
            B_dilated = F.conv2d(B_c, k_dilate, padding=1) > 0.5
            S_adjacent = S_mask * B_dilated.float()
            
            rm = torch.min(R * B_c + 1000.0 * (1.0 - B_c))
            rM = torch.max(R * B_c)
            cm = torch.min(C_grid * B_c + 1000.0 * (1.0 - B_c))
            cM = torch.max(C_grid * B_c)
            
            kH = rM - rm + 1.0
            kW = cM - cm + 1.0
            
            dr_grid = torch.floor((R - rm) / kH)
            dc_grid = torch.floor((C_grid - cm) / kW)
            
            flip_r = (dr_grid != 0.0).float()
            flip_c = (dc_grid != 0.0).float()
            
            r_loc = R - rm - dr_grid * kH
            c_loc = C_grid - cm - dc_grid * kW
            
            r_src_loc = r_loc * (1.0 - flip_r) + (kH - 1.0 - r_loc) * flip_r
            c_src_loc = c_loc * (1.0 - flip_c) + (kW - 1.0 - c_loc) * flip_c
            
            r_src = rm + r_src_loc
            c_src = cm + c_src_loc
            
            idx = (r_src * W + c_src).long()
            idx = torch.clamp(idx, 0, H * W - 1)
            B_flat = B_c.view(-1)
            B_tiled = torch.gather(B_flat, 0, idx.view(-1)).view(B, 1, H, W)
            
            S_matched = S_adjacent * B_tiled
            S_matched = torch.where(B_c > 0.5, torch.tensor(c, dtype=torch.float32, device=x.device), S_matched)
            
            color_mask = S_matched
            for _ in range(15):
                new_color = color_mask.clone()
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        shifted_color = torch.roll(color_mask, shifts=(dr, dc), dims=(2, 3))
                        shifted_dr = torch.roll(dr_grid, shifts=(dr, dc), dims=(2, 3))
                        shifted_dc = torch.roll(dc_grid, shifts=(dr, dc), dims=(2, 3))
                        
                        valid = (dr_grid == shifted_dr) & (dc_grid == shifted_dc) & (B_tiled > 0.5)
                        if dr == 1: valid[:, :, 0, :] = False
                        if dr == -1: valid[:, :, -1, :] = False
                        if dc == 1: valid[:, :, :, 0] = False
                        if dc == -1: valid[:, :, :, -1] = False
                        
                        new_color = torch.max(new_color, shifted_color * valid.float())
                color_mask = new_color
                
            update_mask = (B_tiled > 0.5) & (color_mask > 0.5) & (has_base > 0.5)
            for c_out in range(10):
                is_c_out = (torch.abs(color_mask - c_out) < 0.1) & update_mask
                out[:, c_out] = torch.where(is_c_out, torch.tensor(1.0, device=x.device), out[:, c_out])
                for c_other in range(10):
                    if c_other != c_out:
                        out[:, c_other] = torch.where(is_c_out, torch.tensor(0.0, device=x.device), out[:, c_other])

        return out

if __name__ == '__main__':
    import json
    import numpy as np
    
    with open('neurogolf/data/raw/task285.json') as f:
        data = json.load(f)
        
    model = Task285()
    
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        
        H, W = inp.shape
        x_ts = torch.zeros(1, 10, H, W)
        for c in range(10):
            x_ts[0, c] = torch.tensor(inp == c, dtype=torch.float32)
            
        out_pred = model(x_ts)
        out_pred_label = torch.argmax(out_pred[0], dim=0).numpy()
        
        match = np.array_equal(out_pred_label, out)
        print(f"Train {i}: {match}")
        if not match:
            diff = (out_pred_label != out)
            for r, c in zip(*np.where(diff)):
                print(f"Diff at ({r}, {c}): Pred {out_pred_label[r, c]}, True {out[r, c]}")

