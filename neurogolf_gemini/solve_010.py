import torch
import torch.nn as nn
import onnx

class Task010Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # x is (1, 10, H, W)
        x_any = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0] # (1, 1, H, W)
        
        # Compute height of each column
        H_c = torch.sum(x_any, dim=2, keepdim=True) # (1, 1, 1, W)
        
        # Compute rank of each column
        # rank[c] = sum_c' (H_c[c'] > H_c[c]) + 1
        # To do this, we compare H_c with itself
        H_c_expand1 = H_c.unsqueeze(4) # (1, 1, 1, W, 1)
        H_c_expand2 = H_c.unsqueeze(3) # (1, 1, 1, 1, W)
        
        # is_taller[c, c'] = 1 if H_c[c'] > H_c[c]
        is_taller = (H_c_expand2 > H_c_expand1).float() # (1, 1, 1, W, W)
        
        # Count taller columns for each column
        rank_c = torch.sum(is_taller, dim=4) + 1 # (1, 1, 1, W)
        
        # Apply rank to colors
        out = torch.zeros_like(x)
        
        for c in range(1, 10):
            # A pixel gets color c if x_any > 0 and rank_c == c
            mask_c = (x_any > 0).float() * (rank_c == c).float()
            out[:, c:c+1, :, :] = mask_c
            
        # Background
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        return out * valid_mask

def export_onnx():
    net = Task010Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task010.onnx"
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
