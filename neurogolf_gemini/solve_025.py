import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task025Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        x_colors = x[:, 1:, :, :] # (1, 9, H, W)
        
        ones_H = torch.ones_like(x_colors[:, 0:1, :, 0:1]) # (1, 1, H, 1)
        ones_W = torch.ones_like(x_colors[:, 0:1, 0:1, :]) # (1, 1, 1, W)
        
        true_H_tensor = torch.max(valid_mask, dim=3, keepdim=True)[0].sum(dim=2, keepdim=True)
        true_W_tensor = torch.max(valid_mask, dim=2, keepdim=True)[0].sum(dim=3, keepdim=True)
        
        row_counts = x_colors.sum(dim=3, keepdim=True)
        col_counts = x_colors.sum(dim=2, keepdim=True)
        
        is_hline_mask = (row_counts > true_W_tensor / 2.0).float() * torch.max(valid_mask, dim=3, keepdim=True)[0]
        is_vline_mask = (col_counts > true_H_tensor / 2.0).float() * torch.max(valid_mask, dim=2, keepdim=True)[0]
        
        # hline = is_hline_mask expanded to W
        hline = is_hline_mask * ones_W
        vline = is_vline_mask * ones_H
        
        line_M = torch.clamp(hline + vline, 0.0, 1.0) * x_colors
        stray_M = x_colors - line_M
        
        has_hline = torch.max(is_hline_mask, dim=2, keepdim=True)[0]
        has_vline = torch.max(is_vline_mask, dim=3, keepdim=True)[0]
        
        y_indices = torch.cumsum(ones_H, dim=2) - 1.0 # (1, 1, H, 1)
        L_y_idx = torch.argmax(is_hline_mask, dim=2, keepdim=True).float()
        
        is_above = (y_indices < L_y_idx).float() * has_hline
        is_below = (y_indices > L_y_idx).float() * has_hline
        
        stray_above_cols = torch.max(stray_M * is_above, dim=2, keepdim=True)[0]
        stray_below_cols = torch.max(stray_M * is_below, dim=2, keepdim=True)[0]
        
        out_above = (y_indices == L_y_idx - 1.0).float() * stray_above_cols
        out_below = (y_indices == L_y_idx + 1.0).float() * stray_below_cols
        
        x_indices = torch.cumsum(ones_W, dim=3) - 1.0 # (1, 1, 1, W)
        L_x_idx = torch.argmax(is_vline_mask, dim=3, keepdim=True).float()
        
        is_left = (x_indices < L_x_idx).float() * has_vline
        is_right = (x_indices > L_x_idx).float() * has_vline
        
        stray_left_rows = torch.max(stray_M * is_left, dim=3, keepdim=True)[0]
        stray_right_rows = torch.max(stray_M * is_right, dim=3, keepdim=True)[0]
        
        out_left = (x_indices == L_x_idx - 1.0).float() * stray_left_rows
        out_right = (x_indices == L_x_idx + 1.0).float() * stray_right_rows
        
        new_colors = torch.clamp(line_M + out_above + out_below + out_left + out_right, 0.0, 1.0)
        
        out = torch.zeros_like(x)
        out[:, 1:, :, :] = new_colors
        max_color = torch.max(new_colors, dim=1, keepdim=True)[0]
        out[:, 0:1, :, :] = 1.0 - torch.clamp(max_color, 0.0, 1.0)
        
        return out * valid_mask

def export_onnx():
    net = Task025Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task025.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}},
        dynamo=False
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
