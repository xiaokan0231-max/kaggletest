from make_task285 import Task285
import numpy as np
import torch
import torch.nn.functional as F

class Task285Debug(Task285):
    def forward(self, grid):
        B, C, H, W = grid.shape
        y_grid = torch.arange(H, dtype=torch.float32, device=grid.device).view(-1, 1).expand(H, W)
        x_grid = torch.arange(W, dtype=torch.float32, device=grid.device).view(1, -1).expand(H, W)
        out_grid = grid.clone()
        for i in range(1):
            for c_m in range(1, 10):
                M_m = (out_grid == c_m).float()
                area_m = M_m.sum()
                is_marker = (area_m > 0.5) & (area_m < 1.5)
                dilated_m = F.conv2d(M_m, self.cross_kernel, padding=1) > 0.5
                for c_s in range(1, 10):
                    if c_s == c_m: continue
                    M_s = (out_grid == c_s).float()
                    area_s = M_s.sum()
                    is_source = (area_s > 1.5)
                    A = (dilated_m.float() * M_s)
                    has_adj = A.sum() > 0.5
                    valid = is_marker & is_source & has_adj
                    if valid.item():
                        print(f"Match: marker {c_m}, source {c_s}")
                        mx = (M_m.squeeze() * x_grid).sum()
                        my = (M_m.squeeze() * y_grid).sum()
                        sx = (A.squeeze() * x_grid).sum()
                        sy = (A.squeeze() * y_grid).sum()
                        print(f"mx={mx.item()}, my={my.item()}, sx={sx.item()}, sy={sy.item()}")
                        is_horizontal = (torch.abs(mx - sx) > 0.5).float()
                        is_vertical = (torch.abs(my - sy) > 0.5).float()
                        x_old = is_horizontal * (sx + mx - x_grid) + (1 - is_horizontal) * x_grid
                        y_old = is_vertical * (sy + my - y_grid) + (1 - is_vertical) * y_grid
                        norm_x = x_old / (W - 1) * 2 - 1
                        norm_y = y_old / (H - 1) * 2 - 1
                        sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
                        M_new = F.grid_sample(M_s, sample_grid, mode='nearest', padding_mode='zeros')
                        M_new = (M_new > 0.5) & valid.view(1, 1, 1, 1)
                        print(f"M_new sum: {M_new.sum().item()}")
                        out_grid = torch.where(M_new, torch.tensor(c_m, dtype=grid.dtype, device=grid.device), out_grid)
        return out_grid

model = Task285Debug()
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
inp_t = torch.from_numpy(inp).view(1, 1, H, W)
out_t = model(inp_t)
print(out_t.squeeze().numpy())
