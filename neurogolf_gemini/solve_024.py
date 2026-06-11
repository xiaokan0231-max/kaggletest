import torch
import torch.nn as nn
import onnx

class Task024Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        # Color 2 makes vertical lines
        c2 = x[:, 2:3, :, :]
        m2 = torch.max(c2, dim=2, keepdim=True)[0] # (B, 1, 1, W)
        # broadcast to H
        m2 = m2.expand_as(c2) # (B, 1, H, W)
        
        # Color 1 makes horizontal lines
        c1 = x[:, 1:2, :, :]
        m1 = torch.max(c1, dim=3, keepdim=True)[0] # (B, 1, H, 1)
        m1 = m1.expand_as(c1) # (B, 1, H, W)
        
        # Color 3 makes horizontal lines
        c3 = x[:, 3:4, :, :]
        m3 = torch.max(c3, dim=3, keepdim=True)[0] # (B, 1, H, 1)
        m3 = m3.expand_as(c3) # (B, 1, H, W)
        
        # Priority: 1 and 3 overwrite 2
        # m1 and m3 shouldn't overlap, but let's just make them exclusive
        out_1 = m1
        out_3 = m3 * (1.0 - m1)
        out_2 = m2 * (1.0 - m1) * (1.0 - m3)
        
        bg = 1.0 - torch.max(torch.max(out_1, out_2), out_3)
        
        out_channels = [bg, out_1, out_2, out_3]
        for c in range(4, 10):
            out_channels.append(torch.zeros_like(bg))
            
        out = torch.cat(out_channels, dim=1)
        
        # Apply valid mask
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task024Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task024.onnx"
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
