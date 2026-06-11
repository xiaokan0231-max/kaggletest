import torch
import torch.nn as nn
import onnx

class Task052Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        # Valid mask: 1 where any color (including 0) is present?
        # Wait, the input has a one-hot vector for each valid pixel.
        # But padded pixels are ALL ZEROS across all 10 channels.
        valid_mask = torch.max(x, dim=1, keepdim=True)[0] # (B, 1, H, W)
        
        # True width per row: (B, 1, H, 1)
        true_W = torch.sum(valid_mask, dim=3, keepdim=True)
        
        # For each color 1..9, count pixels in each row
        x_colors = x[:, 1:, :, :] # (B, 9, H, W)
        color_counts = torch.sum(x_colors, dim=3, keepdim=True) # (B, 9, H, 1)
        
        # A row is uniform if color_counts == true_W and true_W > 0
        is_uniform = (color_counts == true_W) & (true_W > 0) # (B, 9, H, 1)
        
        # If ANY color is uniform in this row
        row_has_uniform = torch.max(is_uniform.float(), dim=1, keepdim=True)[0] # (B, 1, H, 1)
        
        # The output should have 5s where row_has_uniform is 1 (and pixel is valid)
        # All other pixels are 0
        
        out_channels = []
        # Channel 0 (background)
        bg = 1.0 - row_has_uniform
        out_channels.append(bg)
        
        # Channels 1 to 9
        for c in range(1, 10):
            if c == 5:
                out_channels.append(row_has_uniform)
            else:
                out_channels.append(torch.zeros_like(row_has_uniform))
                
        out = torch.cat(out_channels, dim=1)
        
        # Apply valid mask
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task052Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task052.onnx"
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
