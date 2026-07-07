import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Task285(nn.Module):
    def __init__(self, max_iter=4):
        super().__init__()
        self.max_iter = max_iter
        self.register_buffer('cross_kernel', torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, grid):
        B, C, H, W = grid.shape
        y_grid = torch.arange(H, dtype=torch.float32, device=grid.device).view(-1, 1).expand(H, W)
        x_grid = torch.arange(W, dtype=torch.float32, device=grid.device).view(1, -1).expand(H, W)
        
        out_grid = grid.clone().float()
        
        for _ in range(self.max_iter):
            # Compute one-hot
            # shape: (1, 10, H, W)
            one_hot = (out_grid.long() == torch.arange(10, device=grid.device).view(1, 10, 1, 1)).float()
            
            # Find neighbors of same color
            # We can do this per color
            has_neighbor = F.conv2d(one_hot.view(10, 1, H, W), self.cross_kernel, padding=1).view(1, 10, H, W) > 0.5
            is_source_pixel = (one_hot > 0.5) & has_neighbor
            is_marker_pixel = (one_hot > 0.5) & (~has_neighbor)
            
            # We only care about colors 1..9
            is_source_pixel[:, 0] = False
            is_marker_pixel[:, 0] = False
            
            # All source pixels mask (regardless of color)
            S_mask_all = is_source_pixel.sum(dim=1, keepdim=True) > 0.5
            
            # All marker pixels mask
            M_mask_all = is_marker_pixel.sum(dim=1, keepdim=True) > 0.5
            
            # We process up to 15 markers one by one
            current_markers = M_mask_all.clone()
            
            for _m in range(15):
                if current_markers.sum() < 0.5:
                    break
                    
                # Isolate first marker
                flat_markers = current_markers.view(-1)
                cumsum = torch.cumsum(flat_markers, dim=0)
                first_marker = (current_markers > 0.5) & (cumsum.view(1, 1, H, W) == 1)
                
                # Remove it from current_markers
                current_markers = current_markers & (~first_marker)
                
                # Find its color
                c_m_tensor = (out_grid * first_marker.float()).sum()
                if c_m_tensor < 0.5: continue
                
                # Dilate it to find adjacent source
                dilated_m = F.conv2d(first_marker.float(), self.cross_kernel, padding=1) > 0.5
                
                # Adjacent source pixel mask
                A = dilated_m & S_mask_all
                if A.sum() < 0.5:
                    continue # Not adjacent to any source yet
                
                # Which color is the source?
                c_s_tensor = (out_grid * A.float()).sum() / A.sum() # average in case of multiple, but should be 1
                
                M_s = is_source_pixel[0, int(c_s_tensor.item())].view(1, 1, H, W)
                
                mx = (first_marker.float().squeeze() * x_grid).sum()
                my = (first_marker.float().squeeze() * y_grid).sum()
                sx = (A.float().squeeze() * x_grid).sum() / A.sum()
                sy = (A.float().squeeze() * y_grid).sum() / A.sum()
                
                is_horizontal = (torch.abs(mx - sx) > 0.5).float()
                is_vertical = (torch.abs(my - sy) > 0.5).float()
                
                x_old = is_horizontal * (sx + mx - x_grid) + (1 - is_horizontal) * x_grid
                y_old = is_vertical * (sy + my - y_grid) + (1 - is_vertical) * y_grid
                
                norm_x = x_old / (W - 1) * 2 - 1
                norm_y = y_old / (H - 1) * 2 - 1
                
                sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
                M_new = F.grid_sample(M_s.float(), sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)
                M_new = M_new > 0.5
                
                out_grid = torch.where(M_new, c_m_tensor, out_grid)

        return out_grid

model = Task285()
inp = np.array([
    [0,0,0,0,0,0,0,8,8,0,0,0,0,0],
    [0,0,0,0,0,0,0,8,8,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,8,4,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,6,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,2,4,0,0,0,0,0,0,0,0],
    [0,2,2,2,2,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,0,0,0,0,0,0,0,0,0,0]
], dtype=np.int64)
H, W = inp.shape
inp_t = torch.from_numpy(inp).view(1, 1, H, W).float()
out_t = model(inp_t)
print(out_t.squeeze().numpy())
