import torch
import torch.nn as nn
import onnx

class RollNetDynamic(nn.Module):
    def forward(self, x, shift_y, shift_x):
        iota_r = torch.arange(30, device=x.device, dtype=torch.float32).view(1, 1, 30, 1)
        target_idx_r = torch.remainder(iota_r - shift_y.float(), 30.0)
        iota_c = torch.arange(30, device=x.device, dtype=torch.float32).view(1, 1, 1, 30)
        P_r = (iota_c == target_idx_r).float() # [1, 1, 30, 30]
        
        target_idx_c = torch.remainder(iota_c - shift_x.float(), 30.0)
        P_c = (iota_r == target_idx_c).float() # [1, 1, 30, 30]
        
        return torch.matmul(torch.matmul(P_r, x), P_c)

net = RollNetDynamic()
x = torch.zeros(1, 1, 30, 30)
x[0, 0, 2, 3] = 1.0

shift_y = torch.tensor([5], dtype=torch.float32).view(1, 1, 1, 1)
shift_x = torch.tensor([-2], dtype=torch.float32).view(1, 1, 1, 1)

out = net(x, shift_y, shift_x)
idx = torch.argmax(out).item()
print("New position:", idx // 30, idx % 30)
