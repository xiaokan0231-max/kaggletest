import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task004Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        x_colors = x[:, 1:, :, :] # (1, 9, 30, 30)
        
        B, C, H, W = x_colors.shape
        
        row_indices = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
        col_indices = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
        
        # Find R_max for each color channel
        row_has_color = torch.max(x_colors, dim=3, keepdim=True)[0] # (1, C, H, 1)
        R_max = torch.max(row_has_color * row_indices, dim=2, keepdim=True)[0] # (1, C, 1, 1)
        
        # Find max_c_per_row for each color channel
        max_c_per_row = torch.max(x_colors * col_indices, dim=3, keepdim=True)[0] # (1, C, H, 1)
        is_rightmost = (col_indices == max_c_per_row) * x_colors # (1, C, H, W)
        
        # keep_mask
        is_R_max = (row_indices == R_max).float()
        is_R_max_minus_1 = (row_indices == R_max - 1.0).float()
        
        keep_mask = torch.clamp(is_R_max + is_R_max_minus_1 * is_rightmost, 0.0, 1.0)
        
        stay_pixels = x_colors * keep_mask
        move_pixels = x_colors * (1.0 - keep_mask)
        
        # Shift move_pixels right by 1
        moved_pixels = F.pad(move_pixels, (1, 0, 0, 0))[:, :, :, :-1]
        
        new_colors = torch.clamp(stay_pixels + moved_pixels, 0.0, 1.0)
        
        out = torch.zeros_like(x)
        out[:, 1:, :, :] = new_colors
        max_color = torch.max(new_colors, dim=1, keepdim=True)[0]
        out[:, 0:1, :, :] = 1.0 - torch.clamp(max_color, 0.0, 1.0)
        
        return out * valid_mask

def export_onnx():
    net = Task004Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task004.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
