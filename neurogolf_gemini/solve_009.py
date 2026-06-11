import torch
import torch.nn as nn
import onnx

class Task009Net(nn.Module):
    def forward(self, x):
        # x: (1, 10, H, W) where H=W=30 max
        # The pad is 0s
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # Grid color is the most frequent non-zero color
        counts = torch.sum(x[:, 1:, :, :], dim=(2, 3)) # (1, 9)
        # Find index of max. Add 1 to get color ID
        G_idx = torch.argmax(counts, dim=1) + 1 # (1,)
        
        # Prepare out_new
        new_pixels = torch.zeros_like(x)
        
        for c in range(1, 10):
            xc = x[:, c:c+1, :, :]
            
            # Row connections (horizontal lines)
            LR = torch.cumsum(xc, dim=3) > 0
            RL = torch.cumsum(xc.flip(3), dim=3).flip(3) > 0
            row_conn = LR & RL
            
            # Col connections (vertical lines)
            UD = torch.cumsum(xc, dim=2) > 0
            DU = torch.cumsum(xc.flip(2), dim=2).flip(2) > 0
            col_conn = UD & DU
            
            # Combine
            conn = row_conn | col_conn
            
            # Ignore grid color
            is_not_G = (torch.tensor([c], dtype=torch.float32) != G_idx.float()).view(1, 1, 1, 1)
            
            conn = (conn.float() * is_not_G) > 0
            
            new_pixels[:, c:c+1, :, :] = conn.float()
            
        # We only overwrite background (color 0)
        is_bg = (x[:, 0:1, :, :] > 0).float()
        out_new = new_pixels * is_bg
        
        # Combine with original x
        # out_new has at most 1 in the channel dimension (assuming no overlapping paths)
        # If paths overlap, max will preserve both, which is invalid one-hot.
        # But we don't expect overlapping paths.
        out = torch.max(x, out_new)
        
        # Zero out channel 0 so it doesn't win argmax
        out[:, 0:1, :, :] = 0
        
        # Ensure it's valid one-hot (argmax then one-hot)
        # In ONNX, we can just do:
        out_idx = torch.argmax(out, dim=1, keepdim=True)
        
        c_idx = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1)
        out_onehot = (out_idx == c_idx).float()
        
        out_onehot = out_onehot * valid_mask
        
        return out_onehot

def export_onnx():
    net = Task009Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task009.onnx"
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
