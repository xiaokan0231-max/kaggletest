import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task015Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        c2 = x[:, 2:3, :, :] # (B, 1, H, W)
        c1 = x[:, 1:2, :, :] # (B, 1, H, W)
        
        padded_c2 = F.pad(c2, (1, 1, 1, 1), value=0.0)
        padded_c1 = F.pad(c1, (1, 1, 1, 1), value=0.0)
        
        H = x.shape[2]
        W = x.shape[3]
        
        # Color 4 mask (corners of 2)
        m4 = torch.zeros_like(c2)
        for dy in [-1, 1]:
            for dx in [-1, 1]:
                m4 = torch.max(m4, padded_c2[:, :, 1-dy:H+1-dy, 1-dx:W+1-dx])
                
        # Color 7 mask (cross of 1)
        m7 = torch.zeros_like(c1)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            m7 = torch.max(m7, padded_c1[:, :, 1-dy:H+1-dy, 1-dx:W+1-dx])
            
        out_channels = []
        # Channel 0
        bg = 1.0 - torch.max(torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0], torch.max(m4, m7))
        out_channels.append(bg)
        
        for c in range(1, 10):
            if c == 4:
                # original 4 + m4
                out_channels.append(torch.max(x[:, 4:5, :, :], m4))
            elif c == 7:
                # original 7 + m7
                out_channels.append(torch.max(x[:, 7:8, :, :], m7))
            else:
                out_channels.append(x[:, c:c+1, :, :])
                
        out = torch.cat(out_channels, dim=1)
        
        # Apply valid mask
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task015Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task015.onnx"
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
