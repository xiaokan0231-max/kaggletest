import torch
import torch.nn as nn
import onnx

class Task007Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        idx_r = torch.arange(30, dtype=torch.float32, device=x.device).view(1, 1, 30, 1)
        idx_c = torch.arange(30, dtype=torch.float32, device=x.device).view(1, 1, 1, 30)
        
        mod_val = (idx_r + idx_c) % 3
        
        mask_0 = (mod_val == 0).float() * valid_mask
        mask_1 = (mod_val == 1).float() * valid_mask
        mask_2 = (mod_val == 2).float() * valid_mask
        
        out = torch.zeros_like(x)
        
        for c in range(1, 10):
            xc = x[:, c:c+1, :, :]
            
            is_0 = torch.max(torch.max(xc * mask_0, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            is_1 = torch.max(torch.max(xc * mask_1, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            is_2 = torch.max(torch.max(xc * mask_2, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            
            out_c = is_0 * mask_0 + is_1 * mask_1 + is_2 * mask_2
            out[:, c:c+1, :, :] = out_c
            
        # The background should be empty since the 3 colors fill the entire grid
        out[:, 0:1, :, :] = 0.0
        
        return out * valid_mask

def export_onnx():
    net = Task007Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task007.onnx"
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
