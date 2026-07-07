import json
import numpy as np
import torch
import torch.nn.functional as F

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task285.json") as f:
    task_data = json.load(f)

from make_task285 import Task285
model = Task285(max_iter=4)

ex = task_data['train'][1]
inp = np.array(ex['input'], dtype=np.float32).reshape(1, 1, len(ex['input']), len(ex['input'][0]))
inp_t = torch.from_numpy(inp)

# We will patch forward to print
old_forward = model.forward
def new_forward(self, grid):
    B, C, H, W = grid.shape
    y_grid = torch.arange(H, dtype=torch.float32, device=grid.device).view(-1, 1).expand(H, W)
    x_grid = torch.arange(W, dtype=torch.float32, device=grid.device).view(1, -1).expand(H, W)
    out_grid = grid.clone().float()
    for _ in range(self.max_iter):
        one_hot = (out_grid.long() == torch.arange(10, device=grid.device).view(1, 10, 1, 1)).float()
        has_neighbor = F.conv2d(one_hot.view(10, 1, H, W), self.moore_kernel, padding=1).view(1, 10, H, W) > 0.5
        is_source_pixel = (one_hot > 0.5) & has_neighbor
        is_marker_pixel = (one_hot > 0.5) & (~has_neighbor)
        is_source_pixel = is_source_pixel * (torch.arange(10, device=grid.device).view(1, 10, 1, 1) > 0.5)
        is_marker_pixel = is_marker_pixel * (torch.arange(10, device=grid.device).view(1, 10, 1, 1) > 0.5)
        S_mask_all = is_source_pixel.sum(dim=1, keepdim=True) > 0.5
        M_mask_all = is_marker_pixel.sum(dim=1, keepdim=True) > 0.5
        current_markers = M_mask_all.clone()
        for _m in range(10):
            flat_markers = current_markers.view(-1)
            cumsum = torch.cumsum(flat_markers, dim=0)
            first_marker = (current_markers > 0.5) & (cumsum.view(1, 1, H, W) == 1)
            current_markers = current_markers & (~first_marker)
            valid_m = first_marker.sum() > 0.5
            c_m_tensor = (out_grid * first_marker.float()).sum()
            dilated_m = F.conv2d(first_marker.float(), self.cross_kernel, padding=1) > 0.5
            A = dilated_m & S_mask_all
            valid_A = A.sum() > 0.5
            if valid_m and valid_A:
                safe_A_sum = torch.clamp(A.sum(), min=1.0)
                c_s_tensor = (out_grid * A.float()).sum() / safe_A_sum
                color_match = (torch.arange(10, device=grid.device).view(1, 10, 1, 1).float() == c_s_tensor)
                M_s = (is_source_pixel.float() * color_match).sum(dim=1, keepdim=True) > 0.5
                mx = (first_marker.float().squeeze() * x_grid).sum()
                my = (first_marker.float().squeeze() * y_grid).sum()
                sx = (A.float().squeeze() * x_grid).sum() / safe_A_sum
                sy = (A.float().squeeze() * y_grid).sum() / safe_A_sum
                
                print(f"Iter {_}, marker {_m}: color {c_m_tensor.item()} at ({my.item()}, {mx.item()}) adjacent to {c_s_tensor.item()} at ({sy.item()}, {sx.item()})")
                is_horizontal = (torch.abs(mx - sx) > 0.5).float()
                is_vertical = (torch.abs(my - sy) > 0.5).float()
                x_old = is_horizontal * (sx + mx - x_grid) + (1 - is_horizontal) * x_grid
                y_old = is_vertical * (sy + my - y_grid) + (1 - is_vertical) * y_grid
                norm_x = x_old / (W - 1) * 2 - 1
                norm_y = y_old / (H - 1) * 2 - 1
                sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
                M_new = F.grid_sample(M_s.float(), sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)
                valid_update = valid_m & valid_A
                M_new = (M_new > 0.5) & valid_update.view(1, 1, 1, 1)
                print(f"M_new sum: {M_new.sum().item()}")
                out_grid = torch.where(M_new, c_m_tensor, out_grid)
    return out_grid

model.forward = new_forward.__get__(model)
out_t = model(inp_t)
