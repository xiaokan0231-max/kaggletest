import torch
import torch.nn as nn
import os

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mask = x[:, 1:10, :, 0:1].sum(dim=1, keepdim=True)
        
        c_left = x[:, :, :, 0:1].clone()
        c_left[:, 0, :, :] = 0.0
        
        c_right = torch.zeros_like(c_left)
        c_right[:, 1:10, :, :] = x[:, 1:10, :, 1:].sum(dim=3, keepdim=True)
        
        c_mid = torch.zeros_like(c_left)
        c_mid[:, 5, :, :] = 1.0
        
        grid_mask = x.sum(dim=1, keepdim=True)
        width = grid_mask[0:1, 0:1, 0:1, :].sum(dim=3, keepdim=True)
        center_idx = (width - 1.0) / 2.0
        
        col_idx = torch.arange(30, device=x.device).view(1, 1, 1, 30).float()
        
        is_left = (col_idx < center_idx - 0.1).float() * grid_mask
        is_right = (col_idx > center_idx + 0.1).float() * grid_mask
        is_mid = ((col_idx >= center_idx - 0.1) & (col_idx <= center_idx + 0.1)).float() * grid_mask
        
        out_modified = is_left * c_left + is_right * c_right + is_mid * c_mid
        out = mask * out_modified + (1.0 - mask) * x
        
        return out

if __name__ == "__main__":
    model = Model()
    model.eval()
    dummy_input = torch.zeros(1, 10, 30, 30, dtype=torch.float32)
    out_dir = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "task060.onnx")
    # Export without dynamic_axes
    torch.onnx.export(
        model, 
        dummy_input, 
        out_path,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        fallback=True
    )
    print(f"Exported to {out_path}")
