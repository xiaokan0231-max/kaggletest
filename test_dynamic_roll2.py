import torch
import torch.nn as nn
import onnx

class RollNet(nn.Module):
    def forward(self, x, shifts):
        # shifts is a 1D tensor of size 2
        return torch.roll(x, shifts=shifts.tolist(), dims=(2, 3)) # tolist() also traces as constant!

class RollNetDynamic(nn.Module):
    def forward(self, x, shift_y, shift_x):
        # Can we create a dynamic permutation matrix?
        # iota = [0, 1, ..., 29]
        # target_idx = (iota - shift_y) % 30
        # Then create one-hot matrix and multiply!
        iota_r = torch.arange(30, device=x.device, dtype=torch.float32).view(1, 1, 30, 1)
        target_idx = torch.remainder(iota_r - shift_y.float(), 30.0)
        
        # P = (iota_r.view(1, 1, 1, 30) == target_idx)
        iota_c = torch.arange(30, device=x.device, dtype=torch.float32).view(1, 1, 1, 30)
        P = (iota_c == target_idx).float()
        
        return torch.matmul(P, x)

net = RollNetDynamic()
x = torch.arange(25).view(1, 1, 5, 5).float()
# change to 30x30 for test
x = torch.zeros(1, 1, 30, 30)
x[0, 0, 2, 3] = 1.0

shift_y = torch.tensor([5], dtype=torch.float32)
shift_x = torch.tensor([0], dtype=torch.float32)

out = net(x, shift_y, shift_x)
print(torch.argmax(out).item() // 30, torch.argmax(out).item() % 30)
