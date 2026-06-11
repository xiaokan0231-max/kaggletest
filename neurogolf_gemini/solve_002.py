import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task002Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0] # (1, 1, 30, 30)
        
        # We only care about x[:, 3:4, :, :], which forms the boundaries
        x3 = x[:, 3:4, :, :]
        
        is_zero = (torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0] == 0).float()
        
        # Initialize M_out with 1s on the boundaries where is_zero is true
        # The borders of the image (or bounding box if padded)
        # But wait! x might be padded to 30x30!
        # The true borders are where valid_mask has its borders!
        
        # So we can initialize M_out outside the valid_mask!
        # Actually, if we just start M_out = 1 - valid_mask, and then dilate!
        K_cross = torch.tensor([[[[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]]]], dtype=torch.float32, device=x.device)
        
        idx_r = torch.arange(30, dtype=torch.float32, device=x.device).view(1, 1, 30, 1)
        idx_c = torch.arange(30, dtype=torch.float32, device=x.device).view(1, 1, 1, 30)
        
        M_out = 1.0 - valid_mask
        M_out = torch.max(M_out, (idx_r == 0).float())
        M_out = torch.max(M_out, (idx_c == 0).float())
        M_out = M_out * (1 - x3)
        
        for _ in range(60):
            M_out_dilated = (F.conv2d(M_out, K_cross, padding=1) > 0).float()
            M_out = M_out_dilated * (1 - x3)
            
        # Now M_out contains all "outside" pixels
        # The "inside" pixels are those that are 0 (or valid) but NOT in M_out, and NOT 3!
        # Wait, if a pixel is 0 and NOT in M_out, it's an enclosed hole!
        
        inside_holes = is_zero * valid_mask * (1 - M_out)
        
        out = x.clone()
        # Add inside_holes to color 4
        out[:, 4:5, :, :] = torch.max(out[:, 4:5, :, :], inside_holes)
        
        # Remove from background
        bg = out[:, 0:1, :, :]
        out[:, 0:1, :, :] = torch.clamp(bg - inside_holes, 0, 1)
        
        out = out * valid_mask
        return out

def export_onnx():
    net = Task002Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task002.onnx"
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
