import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task075Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        anchor = x[:, 1:2, :, :] # Anchor is color 1
        padded_anchor = F.pad(anchor, (1, 1, 1, 1), value=0.0)
        
        H = x.shape[2]
        W = x.shape[3]
        
        # Start with original image but erase color 1
        out_colors = x[:, 1:, :, :].clone()
        # Eraser: color 1 is channel 0 of out_colors
        out_colors[:, 0:1, :, :] = 0.0
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                shifted_anchor = padded_anchor[:, :, 1-dy:H+1-dy, 1-dx:W+1-dx]
                color_val = x[:, 1:, dy+1:dy+2, dx+1:dx+2] # (B, 9, 1, 1)
                
                # Erase color 1 from color_val just in case?
                # The 3x3 pattern shouldn't have color 1, but let's be safe
                cv = color_val.clone()
                cv[:, 0:1, :, :] = 0.0
                
                out_colors = torch.max(out_colors, shifted_anchor * cv)
                
        # Background is where out_colors is 0
        bg = 1.0 - torch.max(out_colors, dim=1, keepdim=True)[0]
        out = torch.cat([bg, out_colors], dim=1)
        
        # Apply valid mask
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task075Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task075.onnx"
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
