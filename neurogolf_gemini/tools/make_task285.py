import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

class Task285(nn.Module):
    def __init__(self):
        super().__init__()
        self.moore_kernel = nn.Parameter(torch.tensor([
            [[[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]]]
        ]), requires_grad=False)
        self.cross_kernel = nn.Parameter(torch.tensor([
            [[[0., 1., 0.],
              [1., 1., 1.],
              [0., 1., 0.]]]
        ]), requires_grad=False)
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def forward(self, x):
        # Convert from 10-channel one-hot back to integer grid
        grid = torch.argmax(x, dim=1, keepdim=True).float()
        
        _, _, H, W = grid.shape
        out_grid = grid.clone()
        
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=grid.device), torch.arange(W, device=grid.device), indexing='ij')
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).float()
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).float()
        
        # S_mask_all selection
        is_source_pixel = torch.zeros(1, 10, H, W, device=grid.device)
        for d in self.dirs:
            shifted = F.pad(grid, (max(0, -d[1]), max(0, d[1]), max(0, -d[0]), max(0, d[0])))
            if d[1] < 0: shifted = shifted[:, :, :, :d[1]]
            elif d[1] > 0: shifted = shifted[:, :, :, d[1]:]
            if d[0] < 0: shifted = shifted[:, :, :d[0], :]
            elif d[0] > 0: shifted = shifted[:, :, d[0]:, :]
            
            same_color = (shifted == grid).float()
            zero_mask = (grid == 0).float()
            
            color_indices = grid.long()
            match_mask = (same_color * (1 - zero_mask))
            is_source_pixel.scatter_add_(1, color_indices, match_mask)
            
        S_mask_all = is_source_pixel.sum(dim=1, keepdim=True) > 0.5
        
        for _m in range(10):
            m_mask = (grid == _m).float()
            for _flood in range(5):
                m_mask = F.conv2d(m_mask, self.moore_kernel, padding=1) > 0.5
                m_mask = m_mask.float() * (grid == _m).float()
                
            dilated_m = F.conv2d(m_mask, self.moore_kernel, padding=1)
            A = (dilated_m > 0.5) & S_mask_all
            
            valid_A = A.sum() > 0.5
            
            # Get the first pixel in A
            cumsum_A = torch.cumsum(A.view(-1).float(), dim=0)
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
            
            mx = (m_mask.float() * x_grid).sum() / torch.clamp(m_mask.sum(), min=1.0)
            my = (m_mask.float() * y_grid).sum() / torch.clamp(m_mask.sum(), min=1.0)
            sx = (first_A.float() * x_grid).sum() / safe_A_sum
            sy = (first_A.float() * y_grid).sum() / safe_A_sum
            
            is_horizontal = (torch.abs(my - sy) < 0.5).float()
            is_vertical = (torch.abs(mx - sx) < 0.5).float()
            
            x_new = is_horizontal * (sx + mx - x_grid) + (1 - is_horizontal) * x_grid
            y_new = is_vertical * (sy + my - y_grid) + (1 - is_vertical) * y_grid
            
            sample_grid = torch.stack([x_new / (W - 1) * 2 - 1, y_new / (H - 1) * 2 - 1], dim=-1).squeeze(1)
            grid_new = F.grid_sample(
                M_s.float(), 
                sample_grid,
                mode='nearest', padding_mode='zeros', align_corners=True
            )
            
            valid_update = valid_A.view(1, 1, 1, 1)
            M_new = (grid_new > 0.5) & valid_update
            out_grid = torch.where(M_new, torch.tensor(float(_m), device=grid.device), out_grid)
                
        # Convert back to one-hot
        out_one_hot = (torch.arange(10, device=x.device).view(1, 10, 1, 1).float() == out_grid).float()
        return out_one_hot

if __name__ == "__main__":
    with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task285.json") as f:
        task_data = json.load(f)
        
    ex = task_data['train'][0]
    inp = np.array(ex['input'], dtype=np.float32).reshape(1, 1, len(ex['input']), len(ex['input'][0]))
    
    # Run torch model
    model = Task285()
    model.eval()
    
    # Input must be one-hot 10-channel
    inp_t = torch.from_numpy(inp).long().squeeze()
    inp_onehot = F.one_hot(inp_t, num_classes=10).permute(2, 0, 1).unsqueeze(0).float()
    
    out_t = model(inp_onehot)
    print("Python Output:")
    print(torch.argmax(out_t, dim=1).squeeze().numpy())
    
    # Export ONNX
    dummy = torch.randint(0, 10, (1, 10, 10, 14), dtype=torch.float32)
    torch.onnx.export(
        model, 
        dummy, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task285.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        opset_version=17
    )
    print("Exported task285.onnx")
