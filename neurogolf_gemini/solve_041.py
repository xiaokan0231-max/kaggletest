import torch
import torch.nn as nn
import onnx

class Task041Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        # We only process channels 1 to 9
        x_colors = x[:, 1:, :, :] # (B, 9, H, W)
        
        prefix_sum = torch.cumsum(x_colors, dim=3)
        suffix_sum = torch.cumsum(torch.flip(x_colors, dims=[3]), dim=3)
        suffix_sum = torch.flip(suffix_sum, dims=[3])
        
        fill_mask = (prefix_sum > 0) & (suffix_sum > 0)
        
        # Combine back
        filled_colors = fill_mask.float()
        
        # Background channel is 1 where all other channels are 0
        bg = 1.0 - torch.max(filled_colors, dim=1, keepdim=True)[0]
        
        out = torch.cat([bg, filled_colors], dim=1)
        
        # Mask with valid_mask
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task041Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task041.onnx"
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
