import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task011Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        is_color = torch.sum(x[:, 1:5, :, :], dim=1, keepdim=True) + \
                   torch.sum(x[:, 6:10, :, :], dim=1, keepdim=True)
                   
        counts = F.avg_pool2d(is_color, kernel_size=3, stride=4, padding=0) * 9
        counts = torch.round(counts)
        
        valid_counts = F.avg_pool2d(valid_mask, kernel_size=3, stride=4, padding=0) * 9
        counts = counts + (1 - (valid_counts > 0).float()) * 100
        
        min_count = torch.min(counts)
        is_special = (counts == min_count).float() # (1, 1, 7, 7)
        
        expanded_special = F.interpolate(is_special, scale_factor=4, mode='nearest') # (1, 1, 28, 28)
        expanded_special = F.pad(expanded_special, (0, 2, 0, 2)) # (1, 1, 30, 30)
        
        special_block = x * expanded_special
        special_block[:, 0:1, :, :] = 0.0
        special_block[:, 5:6, :, :] = 0.0
        
        map_3x3 = torch.zeros(1, 10, 3, 3, device=x.device)
        for i in range(7):
            for j in range(7):
                map_3x3 = map_3x3 + special_block[:, :, i*4:i*4+3, j*4:j*4+3]
                
        out_colors = F.interpolate(map_3x3, scale_factor=4, mode='nearest') # (1, 10, 12, 12)
        out_colors = F.pad(out_colors, (0, 18, 0, 18)) # (1, 10, 30, 30)
        
        grid_5 = x[:, 5:6, :, :]
        
        out_colors = out_colors * (1 - grid_5)
        
        out = out_colors
        out[:, 5:6, :, :] = grid_5
        
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        return out * valid_mask

def export_onnx():
    net = Task011Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task011.onnx"
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
