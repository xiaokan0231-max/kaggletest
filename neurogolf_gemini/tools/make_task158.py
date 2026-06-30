import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

class Task158(nn.Module):
    def __init__(self):
        super().__init__()
        self.moore_kernel = nn.Parameter(torch.tensor([
            [[[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]]]
        ]), requires_grad=False)

    def shift_mask(self, m, dy, dx):
        _, _, H, W = m.shape
        y_grid = (torch.cumsum(torch.ones(1, 1, H, W, device=m.device), dim=2) - 1).float()
        x_grid = (torch.cumsum(torch.ones(1, 1, H, W, device=m.device), dim=3) - 1).float()
        
        x_new = x_grid - dx.view(1, 1, 1, 1)
        y_new = y_grid - dy.view(1, 1, 1, 1)
        
        norm_x = x_new / (W - 1) * 2 - 1
        norm_y = y_new / (H - 1) * 2 - 1
        
        sample_grid = torch.stack([norm_x, norm_y], dim=-1).squeeze(1)
        return F.grid_sample(m, sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)

    def forward(self, x):
        _, _, orig_H, orig_W = x.shape
        x = F.pad(x, (0, 30 - orig_W, 0, 30 - orig_H))
        grid = torch.argmax(x, dim=1, keepdim=True).float()
        H, W = 30, 30
        
        pixel_counts = x.sum(dim=(2,3), keepdim=True)
        bg_color = torch.argmax(pixel_counts.view(1, 10), dim=1)
        
        is_present = (pixel_counts > 0.5).float().view(1, 10)
        is_present = is_present * (torch.arange(10, device=x.device).unsqueeze(0) != bg_color.unsqueeze(1)).float()
        
        dilated_x = F.conv2d(x, self.moore_kernel.expand(10, 1, 3, 3), padding=1, groups=10) > 0.5
        dilated_x = dilated_x.float()
        
        overlap = torch.einsum('bchw,bdhw->bcd', dilated_x, x) > 0.5
        overlap = overlap.float() * is_present.unsqueeze(2) * is_present.unsqueeze(1)
        diag_mask = 1 - torch.eye(10, device=x.device).unsqueeze(0)
        overlap = overlap * diag_mask
        
        num_touches = overlap.sum(dim=2)
        is_conn = (num_touches > 1.5).float()
        conn_idx = torch.argmax(is_conn, dim=1)
        
        is_endpoint = is_present - is_conn
        cumsum_endpoints = torch.cumsum(is_endpoint, dim=1)
        is_c1 = (is_endpoint > 0.5) & (cumsum_endpoints == 1.0)
        is_c2 = (is_endpoint > 0.5) & (cumsum_endpoints == 2.0)
        
        c1_idx = torch.argmax(is_c1.float(), dim=1)
        c2_idx = torch.argmax(is_c2.float(), dim=1)
        
        batch_indices = torch.arange(1, device=x.device)
        # Using indexing: x[0, c1_idx[0]:c1_idx[0]+1] doesn't export well sometimes, but we can do it via gather or just multiplying
        # Let's use multiplication:
        c1_mask = (x * is_c1.view(1, 10, 1, 1)).sum(dim=1, keepdim=True)
        c2_mask = (x * is_c2.view(1, 10, 1, 1)).sum(dim=1, keepdim=True)
        conn_mask = (x * is_conn.view(1, 10, 1, 1)).sum(dim=1, keepdim=True)
        
        dilated_conn = F.conv2d(conn_mask, self.moore_kernel, padding=1) > 0.5
        dilated_conn = dilated_conn.float()
        
        template_c1_mask = c1_mask * dilated_conn
        template_c2_mask = c2_mask * dilated_conn
        
        y_grid = (torch.cumsum(torch.ones(1, 1, H, W, device=x.device), dim=2) - 1).float()
        x_grid = (torch.cumsum(torch.ones(1, 1, H, W, device=x.device), dim=3) - 1).float()
        
        sum_c1 = torch.clamp(template_c1_mask.sum(), min=1.0)
        y1_t = (template_c1_mask * y_grid).sum() / sum_c1
        x1_t = (template_c1_mask * x_grid).sum() / sum_c1
        
        sum_c2 = torch.clamp(template_c2_mask.sum(), min=1.0)
        y2_t = (template_c2_mask * y_grid).sum() / sum_c2
        x2_t = (template_c2_mask * x_grid).sum() / sum_c2
        
        dy_t = torch.round(y2_t - y1_t)
        dx_t = torch.round(x2_t - x1_t)
        
        out_mask = torch.zeros_like(grid)
        
        ones_k = torch.ones(1, 1, 2*H - 1, 2*W - 1, device=x.device)
        k_y = (torch.cumsum(ones_k, dim=2) - H).float()
        k_x = (torch.cumsum(ones_k, dim=3) - W).float()
        
        for S in range(1, 16):
            pool_kernel = torch.ones(1, 1, S, S, device=grid.device)
            for sign_y in [-1, 1]:
                for sign_x in [-1, 1]:
                    DY = dy_t * sign_y * S
                    DX = dx_t * sign_x * S
                    
                    shifted_c2 = self.shift_mask(c2_mask, -DY, -DX)
                    base_c1 = c1_mask * shifted_c2
                    
                    base_c1_pad = F.pad(base_c1, (0, S-1, 0, S-1))
                    conv_counts = F.conv2d(base_c1_pad, pool_kernel, padding=0)
                    valid_centers = (conv_counts > S*S - 0.5).float()
                    
                    y_in = y1_t + torch.floor(k_y / S) * sign_y
                    x_in = x1_t + torch.floor(k_x / S) * sign_x
                    
                    norm_x = x_in / (W - 1) * 2 - 1
                    norm_y = y_in / (H - 1) * 2 - 1
                    sample_grid = torch.stack([norm_x, norm_y], dim=-1).squeeze(1)
                    
                    kernel_mask = F.grid_sample(conn_mask, sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)
                    
                    drawn = F.conv2d(valid_centers, torch.flip(kernel_mask, [2, 3]), padding=(H-1, W-1))
                    out_mask = torch.max(out_mask, drawn)
                    
        out_grid = torch.where(out_mask > 0.5, conn_idx.float().view(1, 1, 1, 1), grid)
        out_one_hot = (torch.arange(10, device=x.device).view(1, 10, 1, 1).float() == out_grid).float()
        return out_one_hot[:, :, :orig_H, :orig_W]

if __name__ == "__main__":
    with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
        task_data = json.load(f)
        
    ex = task_data['train'][0]
    inp = np.array(ex['input'], dtype=np.int64).reshape(1, len(ex['input']), len(ex['input'][0]))
    
    # Run torch model
    model = Task158()
    model.eval()
    
    inp_t = torch.from_numpy(inp).long()
    inp_onehot = F.one_hot(inp_t, num_classes=10).permute(0, 3, 1, 2).float()
    
    out_t = model(inp_onehot)
    print("Python Output:")
    print(torch.argmax(out_t, dim=1).squeeze().numpy())
    
    # Export ONNX
    dummy = torch.randint(0, 10, (1, 1, 10, 14), dtype=torch.int64).squeeze(1)
    dummy_onehot = F.one_hot(dummy, num_classes=10).permute(0, 3, 1, 2).float()
    
    torch.onnx.export(
        model, 
        dummy_onehot, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task158.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        opset_version=17
    )
    print("Exported task158.onnx")
