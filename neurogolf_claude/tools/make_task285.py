import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import sys
import os

class Task285(nn.Module):
    def __init__(self, max_iter=4):
        super().__init__()
        self.max_iter = max_iter
        self.register_buffer('cross_kernel', torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('moore_kernel', torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, grid):
        B, C, H, W = grid.shape
        y_grid = torch.arange(H, dtype=torch.float32, device=grid.device).view(-1, 1).expand(H, W)
        x_grid = torch.arange(W, dtype=torch.float32, device=grid.device).view(1, -1).expand(H, W)
        
        out_grid = grid.clone().float()
        
        for _ in range(self.max_iter):
            one_hot = (out_grid.long() == torch.arange(10, device=grid.device).view(1, 10, 1, 1)).float()
            
            has_neighbor = F.conv2d(one_hot.view(10, 1, H, W), self.moore_kernel, padding=1).view(1, 10, H, W) > 0.5
            is_source_pixel = (one_hot > 0.5) & has_neighbor
            is_marker_pixel = (one_hot > 0.5) & (~has_neighbor)
            
            # Mask out color 0
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
                
                # Isolate exactly one adjacent source pixel
                flat_A = A.view(-1)
                cumsum_A = torch.cumsum(flat_A, dim=0)
                first_A = A & (cumsum_A.view(1, 1, H, W) == 1)
                
                safe_A_sum = torch.clamp(first_A.sum(), min=1.0)
                c_s_tensor = (out_grid * first_A.float()).sum() / safe_A_sum
                
                color_match = (torch.arange(10, device=grid.device).view(1, 10, 1, 1).float() == c_s_tensor)
                valid_source_pixels = is_source_pixel.float() * color_match
                
                # Flood fill from first_A to get M_s
                component = first_A.clone().float()
                valid_mask = valid_source_pixels.sum(dim=1, keepdim=True) > 0.5
                for _flood in range(15):
                    dilated_comp = F.conv2d(component, self.moore_kernel, padding=1)
                    component = (dilated_comp > 0.5).float() * valid_mask.float()
                M_s = component > 0.5
                
                mx = (first_marker.float().squeeze() * x_grid).sum()
                my = (first_marker.float().squeeze() * y_grid).sum()
                sx = (first_A.float().squeeze() * x_grid).sum() / safe_A_sum
                sy = (first_A.float().squeeze() * y_grid).sum() / safe_A_sum
                
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
                
                out_grid = torch.where(M_new, c_m_tensor, out_grid)

        return out_grid

if __name__ == "__main__":
    import numpy as np
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
    print("Python Output:")
    print(out_t.squeeze().numpy())
    
    # Export ONNX
    dummy = torch.zeros(1, 1, 30, 30, dtype=torch.float32)
    torch.onnx.export(
        model, 
        dummy, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task285.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=17
    )
    print("Exported task285.onnx")
