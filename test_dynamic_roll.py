import torch
import torch.nn as nn
import onnx

class RollNet(nn.Module):
    def forward(self, x, dy, dx):
        # dy and dx are 1D tensors of size 1
        return torch.roll(x, shifts=(dy.item(), dx.item()), dims=(2, 3))

net = RollNet()
x = torch.zeros(1, 1, 5, 5)
dy = torch.tensor([1], dtype=torch.int64)
dx = torch.tensor([2], dtype=torch.int64)

# Wait, dy.item() is evaluated at trace time, so it becomes a static constant!
# To make it dynamic, we must pass it to torch.roll without .item()
