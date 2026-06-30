import json
import numpy as np
import torch
import torch.nn.functional as F

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    x = torch.from_numpy(inp).long().unsqueeze(0).unsqueeze(0)
    x = F.one_hot(x.squeeze(0).squeeze(0), num_classes=10).permute(2, 0, 1).unsqueeze(0).float()
    
    grid = torch.argmax(x, dim=1, keepdim=True).float()
    
    pixel_counts = x.sum(dim=(2,3), keepdim=True)
    bg_color = torch.argmax(pixel_counts.view(1, 10), dim=1)
    
    is_present = (pixel_counts > 0.5).float().view(1, 10)
    is_present = is_present * (torch.arange(10) != bg_color.unsqueeze(1)).float()
    
    moore_kernel = torch.tensor([[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]])
    dilated_x = F.conv2d(x, moore_kernel.expand(10, 1, 3, 3), padding=1, groups=10) > 0.5
    dilated_x = dilated_x.float()
    
    overlap = torch.einsum('bchw,bdhw->bcd', dilated_x, x) > 0.5
    overlap = overlap.float() * is_present.unsqueeze(2) * is_present.unsqueeze(1)
    diag_mask = 1 - torch.eye(10).unsqueeze(0)
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
    
    c1_mask = x[:, c1_idx[0]:c1_idx[0]+1]
    c2_mask = x[:, c2_idx[0]:c2_idx[0]+1]
    conn_mask = x[:, conn_idx[0]:conn_idx[0]+1]
    
    dilated_conn = F.conv2d(conn_mask, moore_kernel, padding=1) > 0.5
    dilated_conn = dilated_conn.float()
    
    template_c1_mask = c1_mask * dilated_conn
    template_c2_mask = c2_mask * dilated_conn
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)
    
    sum_c1 = torch.clamp(template_c1_mask.sum(), min=1.0)
    y1_t = (template_c1_mask * y_grid).sum() / sum_c1
    x1_t = (template_c1_mask * x_grid).sum() / sum_c1
    
    sum_c2 = torch.clamp(template_c2_mask.sum(), min=1.0)
    y2_t = (template_c2_mask * y_grid).sum() / sum_c2
    x2_t = (template_c2_mask * x_grid).sum() / sum_c2
    
    dy_t = torch.round(y2_t - y1_t)
    dx_t = torch.round(x2_t - x1_t)
    
    out_mask = torch.zeros_like(grid)
    
    def shift_mask(m, dy, dx):
        y_new = y_grid - dy
        x_new = x_grid - dx
        norm_x = x_new / (W - 1) * 2 - 1
        norm_y = y_new / (H - 1) * 2 - 1
        sample_grid = torch.stack([norm_x, norm_y], dim=-1).squeeze(1)
        return F.grid_sample(m, sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)
        
    k_y, k_x = torch.meshgrid(torch.arange(-(H-1), H).float(), torch.arange(-(W-1), W).float(), indexing='ij')
    
    for S in range(1, 16):
        pool_kernel = torch.ones(1, 1, S, S)
        for sign_y in [-1, 1]:
            for sign_x in [-1, 1]:
                DY = dy_t * sign_y * S
                DX = dx_t * sign_x * S
                
                shifted_c2 = shift_mask(c2_mask, DY, DX) # Wait, my manual code shifted by -DY, -DX. grid_sample shifts by positive? 
                # Let's check: x_new = x_grid - dx. If we want shifted_c2 to be c2 shifted by -DY, we should pass dy=-DY?
                # Actually, if we pass DY to shift_mask, x_new = x_grid - DY. The pixels at x_grid will fetch from x_grid - DY. 
                # Which means the image is shifted by +DY.
                # So we want to shift c2 by -DY!
                shifted_c2 = shift_mask(c2_mask, -DY, -DX)
                
                base_c1 = c1_mask * shifted_c2
                
                conv_counts = F.conv2d(base_c1, pool_kernel, padding=0)
                valid_centers = (conv_counts > S*S - 0.5).float()
                
                if valid_centers.sum() > 0.5:
                    y_in = y1_t + torch.floor(k_y / S) * sign_y
                    x_in = x1_t + torch.floor(k_x / S) * sign_x
                    
                    norm_x = x_in / (W - 1) * 2 - 1
                    norm_y = y_in / (H - 1) * 2 - 1
                    sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
                    
                    kernel_mask = F.grid_sample(conn_mask, sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)
                    
                    # Pad valid_centers to match conv2d
                    # We want to stamp kernel_mask at every valid_center.
                    # F.conv2d(valid_centers, torch.flip(kernel_mask, [2, 3]), padding=(H-1, W-1))
                    valid_centers = F.pad(valid_centers, (0, S-1, 0, S-1))
                    drawn = F.conv2d(valid_centers, torch.flip(kernel_mask, [2, 3]), padding=(H-1, W-1))
                    out_mask = torch.max(out_mask, drawn)
                    
    out_pred = grid.clone()
    out_pred = torch.where(out_mask > 0.5, conn_idx[0].float(), out_pred)
    print(f"Ex {i} PyTorch Match: {np.array_equal(out_pred.squeeze().numpy(), out_expected)}")
