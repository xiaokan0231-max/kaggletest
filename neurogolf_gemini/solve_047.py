import torch
import torch.nn as nn
import onnx

class Task047Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        crosses = []
        for c in range(1, 10):
            x_c = x[:, c:c+1, :, :] # (B, 1, H, W)
            has_row = torch.max(x_c, dim=3, keepdim=True)[0] # (B, 1, H, 1)
            has_col = torch.max(x_c, dim=2, keepdim=True)[0] # (B, 1, 1, W)
            cross_c = torch.max(has_row, has_col) # (B, 1, H, W)
            crosses.append(cross_c)
            
        cross_tensor = torch.cat(crosses, dim=1) # (B, 9, H, W)
        num_crosses = torch.sum(cross_tensor, dim=1, keepdim=True) # (B, 1, H, W)
        
        is_intersection = (num_crosses >= 2).float() # (B, 1, H, W)
        
        out_channels = []
        # Channel 0 (Background)
        bg = (num_crosses == 0).float()
        out_channels.append(bg)
        
        # Channels 1..9
        for c in range(1, 10):
            if c == 2:
                # Color 2 is intersection PLUS original cross_2 if any
                c_mask = is_intersection + cross_tensor[:, 1:2, :, :] * (1.0 - is_intersection)
                out_channels.append(c_mask)
            else:
                # Other colors are cross_c EXCEPT where intersection
                c_mask = cross_tensor[:, c-1:c, :, :] * (1.0 - is_intersection)
                out_channels.append(c_mask)
                
        out = torch.cat(out_channels, dim=1)
        
        # Apply valid mask
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task047Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task047.onnx"
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
